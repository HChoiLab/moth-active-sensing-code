from __future__ import annotations
import os
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw

try:
    import imageio.v3 as iio
except Exception:
    iio = None

Color = Tuple[int, int, int]

def process_excel_files(data_dir, save_dir):
    """
    Processes Excel files to extract sequences of length 1000 from multiple videos within each file
    and saves them as torch files.

    Args:
        data_dir (str): Directory containing the Excel files.
        save_dir (str): Directory to save the processed torch files.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_list = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') or f.endswith('.xls') or f.endswith('.csv')]

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, header=None)
        else:
            df = pd.read_csv(file_path, header=None)

        data = df.values  # Shape: (num_rows, num_columns)
        num_rows, num_columns = data.shape

        # Each video corresponds to 4 columns
        num_videos = num_columns // 4

        all_sequences = []

        for vid_idx in range(num_videos):
            start_col = vid_idx * 4
            end_col = start_col + 4
            video_data = data[:, start_col:end_col]  # Shape: (num_rows, 4)

            # Check if the video has any NaN values and remove them
            video_data = video_data[~np.isnan(video_data).any(axis=1)]

            video_length = video_data.shape[0]

            # Split video into sequences of length 1000
            num_sequences = video_length // 1000
            remainder = video_length % 1000

            sequences = []

            for seq_idx in range(num_sequences):
                start_idx = seq_idx * 1000
                end_idx = start_idx + 1000
                sequence = video_data[start_idx:end_idx, :].T  # Transpose to shape (4, 1000)
                sequences.append(sequence)

            if remainder > 0:
                # Process the last sequence, pad if necessary
                start_idx = num_sequences * 1000
                sequence = video_data[start_idx:, :].T  # Shape: (4, remainder)
                # Pad the sequence to length 1000
                padding = np.zeros((4, 1000 - sequence.shape[1]))
                sequence_padded = np.concatenate((sequence, padding), axis=1)
                sequences.append(sequence_padded)

            all_sequences.extend(sequences)

        if all_sequences:
            # Stack all sequences from all videos in this Excel file
            sequences_tensor = torch.tensor(np.stack(all_sequences), dtype=torch.float32)
            # Save the tensor to a torch file
            base_name = os.path.splitext(file_name)[0]
            save_path = os.path.join(save_dir, base_name + '.pt')
            torch.save(sequences_tensor, save_path)
            print(f"Processed and saved: {save_path}")
        else:
            print(f"No valid data found in: {file_name}")

def random_sum_of_sines(N, T, n_sinusoids=3, freq_range=(0.1, 2.0), amplitude_range=(0.5, 1.0)):
    """
    Creates N random sequences, each T timesteps long, where each sequence is a
    sum of 'n_sinusoids' sinusoids with random frequencies, phases, and amplitudes.
    The resulting sequences are scaled to be within [0, 1].

    Parameters:
    -----------
    N : int
        Number of sequences to generate.
    T : int
        Number of timesteps in each sequence.
    n_sinusoids : int, optional
        How many individual sinusoids will be summed to form each sequence.
    freq_range : tuple (float, float), optional
        The minimum and maximum frequencies for the sinusoids.
    amplitude_range : tuple (float, float), optional
        The minimum and maximum amplitudes for the sinusoids.

    Returns:
    --------
    sequences : np.ndarray of shape (N, T)
        The generated sequences, each bounded between 0 and 1.
    """

    # Time values from 0 to 2π (to get a typical sine-wave feel over T points)
    t = np.linspace(0, 2 * np.pi, T)

    # Initialize an array to store all sequences
    sequences = np.zeros((N, T))

    for i in range(N):
        # Start with a zero array; will sum multiple sinusoids
        seq = np.zeros(T)
        
        for _ in range(n_sinusoids):
            # Random frequency
            freq = np.random.uniform(*freq_range)
            # Random phase
            phase = np.random.uniform(0, 2 * np.pi)
            # Random amplitude
            amplitude = np.random.uniform(*amplitude_range)
            
            # Add this sinusoid to the sequence
            seq += amplitude * np.sin(freq * t + phase)
        
        # Scale sequence to [0, 1] to ensure bounding
        min_val, max_val = seq.min(), seq.max()
        if max_val > min_val:
            seq = (seq - min_val) / (max_val - min_val)
        else:
            # Edge case: if the entire sequence is constant (extremely unlikely),
            # just set it to zero.
            seq = np.zeros_like(seq)

        sequences[i] = seq

    return sequences

def ramp_schedule(n_steps, start_step, stop_step=None, start=0.0, stop=1.0):
    if stop_step is None:
        stop_step = n_steps
    L = np.ones(n_steps) * start
    eps = np.arange(start_step, stop_step+1) - start_step
    L[int(start_step - 1):stop_step] = start + (eps / (stop_step - start_step)) * (stop - start)
    if stop_step != n_steps:
        L[stop_step:] = stop
    return L


@dataclass
class MothViewRenderer:
    world_size_px: Tuple[int, int] = (512, 512)
    crop_size_px: Tuple[int, int] = (224, 224)
    flower_radius_px: int = 20
    flower_fill: Color = (255, 255, 255)
    flower_outline: Optional[Color] = (255, 255, 255) #(220, 235, 255)
    flower_outline_width: int = 2
    agent_color: Color = (18, 158, 33)
    agent_plus_size_px: int = 18
    agent_plus_thickness_px: int = 4
    grayscale: bool = False

    def _to_px(self, x: float, y: float) -> Tuple[int, int]:
        W, H = self.world_size_px
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        return int(round(x * (W - 1))), int(round(y * (H - 1)))

    def _draw_flower(self, im: Image.Image, center: Tuple[int, int]) -> None:
        draw = ImageDraw.Draw(im)
        cx, cy = center
        r = self.flower_radius_px
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=self.flower_fill, outline=self.flower_outline, width=self.flower_outline_width)

    def _draw_agent_plus(self, im: Image.Image, center: Tuple[int, int]) -> None:
        draw = ImageDraw.Draw(im)
        cx, cy = center
        L = self.agent_plus_size_px
        t = self.agent_plus_thickness_px
        draw.rectangle([cx - t // 2, cy - L, cx + (t - 1) // 2, cy + L], fill=self.agent_color)
        draw.rectangle([cx - L, cy - t // 2, cx + L, cy + (t - 1) // 2], fill=self.agent_color)

    def _safe_crop(self, im: Image.Image, center: Tuple[int, int], crop_size: Tuple[int, int]) -> Image.Image:
        W, H = im.size
        cw, ch = crop_size
        pad_x, pad_y = cw // 2, ch // 2
        padded = Image.new("RGB", (W + 2 * pad_x, H + 2 * pad_y), (0, 0, 0))
        padded.paste(im, (pad_x, pad_y))
        cx, cy = center
        cx_p, cy_p = cx + pad_x, cy + pad_y
        left, top = cx_p - cw // 2, cy_p - ch // 2
        return padded.crop((left, top, left + cw, top + ch))

    def _render_world(self, flower_xy: Tuple[float, float], moth_xy: Tuple[float, float]) -> Image.Image:
        W, H = self.world_size_px
        im = Image.new("RGB", (W, H), (0, 0, 0))
        self._draw_flower(im, self._to_px(*flower_xy))
        self._draw_agent_plus(im, self._to_px(*moth_xy))
        return im

    def render_frames(self, flower_traj: np.ndarray, moth_traj: np.ndarray) -> List[np.ndarray]:
        sensory_frames = []
        for f, m in zip(flower_traj, moth_traj):
            full_world = Image.new("RGB", self.world_size_px, (0, 0, 0))
            self._draw_flower(full_world, self._to_px(*f))
            moth_px = self._to_px(*m)
            crop = self._safe_crop(full_world, moth_px, self.crop_size_px)
            if self.grayscale:
                crop = crop.convert('L')
            sensory_frames.append(np.array(crop, dtype=np.uint8))
        return sensory_frames

    def visualize_full_world(self, flower_traj: np.ndarray, moth_traj: np.ndarray) -> List[np.ndarray]:
        frames = []
        for f, m in zip(flower_traj, moth_traj):
            frame = self._render_world(tuple(f), tuple(m))
            frames.append(np.array(frame, dtype=np.uint8))
        return frames

    def save_video(self, frames: List[np.ndarray], out_path: str, fps: int = 30) -> None:
        if iio is None:
            raise ImportError("Install imageio[ffmpeg] to save videos.")
        iio.imwrite(out_path, np.stack(frames), fps=fps)


# Example
if __name__ == "__main__":
    T = 120
    t = np.linspace(0, 2 * np.pi, T)
    flower_xy = np.stack([0.5 + 0.3 * np.cos(t), 0.5 + 0.3 * np.sin(t)], 1)
    moth_xy = np.stack([0.1 + 0.4 * (1 - np.cos(t)) / 2, 0.5 + 0.2 * np.sin(2 * t)], 1)

    renderer = MothViewRenderer(world_size_px=(640, 640), crop_size_px=(256, 256), flower_radius_px=25)

    sensory_frames = renderer.render_frames(flower_xy, moth_xy)
    viz_frames = renderer.visualize_full_world(flower_xy, moth_xy)

    renderer.save_video(viz_frames, "world_view.mp4")
    renderer.save_video(sensory_frames, "sensory_view.mp4")

    print("Done")