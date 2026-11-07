import torch
from torch.utils.data import Dataset
import os
import math
import random
import numpy as np
import glob
import pandas as pd
from utils.data import random_sum_of_sines, ramp_schedule
from observation_models import generate_proboscis_observations_1d
from utils.data import MothViewRenderer

def normalize_seq(seq, dim=-1):
    return (seq - torch.min(seq, dim=dim, keepdim=True)[0]) / (torch.max(seq, dim=dim, keepdim=True)[0] - torch.min(seq, dim=dim, keepdim=True)[0])

def compute_visual_obs(s_t, a_t, sigma=0.05, nu=0.1, A=1.):
        """
        Compute x_t based on s_t and a_t
        s_t and a_t are tensors of length T (here T=1000)
        """

        delta = torch.abs(s_t - a_t)
        # Compute μ_x using the Gaussian PDF
        mu_x = A * (1 / math.sqrt(2 * torch.pi * nu**2)) * torch.exp(- (delta ** 2) / (2 * nu ** 2))

        # Generate x_t ~ N(μ_x, σ^2)
        x_t = mu_x + torch.randn_like(mu_x) * sigma

        return x_t

####################################################################################################################################
######################################################## MOTH DATASETS #############################################################
####################################################################################################################################

class MothDataset(Dataset):

    """ This is the default mode dataset """

    def __init__(self, data_dir, filenames=None, nu=1.0, sigma=0.1, A=1.0, normalize=True, seq_len=1000, mechanosensory=False, mech_obs_params: dict = None):
        """
        Args:
            data_dir (string): Directory with all the torch files.
            nu (float): Hyperparameter ν^2 for the observation model.
            sigma (float): Hyperparameter σ^2 for the observation model.
        """
        self.data_dir = data_dir
        self.nu = nu
        self.sigma = sigma
        self.A = A
        self.seq_len = seq_len
        self.normalized = normalize
        self.mechanosensory = mechanosensory
        self.mech_obs_params = mech_obs_params

        # Get list of torch files
        if filenames is None:
            filenames = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.file_names = filenames
        self.file_list = [os.path.join(data_dir, f) for f in self.file_names if f.endswith('.pt')]
        self.s, self.a, self.x = [], [], []
        self.seq_files = []

        self._load_data(normalize=normalize)

    def _load_data(self, normalize=True):
        for file_path, fname in zip(self.file_list, self.file_names):
            # Load the tensor
            sequences_tensor = torch.load(file_path)  # Shape: (num_sequences, 4, 1000)
            num_sequences = sequences_tensor.shape[0]
            for i in range(num_sequences):
                sequence = sequences_tensor[i]  # Shape: (4, 1000)
            
                # Extract s_t and a_t
                s_t = sequence[1, :]  # Flower y position (2nd column)
                a_t = sequence[3, :]  # Moth y position (4th column)

                for t in range(1000 // self.seq_len):

                    s_t_ = s_t[self.seq_len * t: self.seq_len * (t + 1)]
                    a_t_ = a_t[self.seq_len * t: self.seq_len * (t + 1)]
                    
                    # center around the mean of the flower position
                    stmean = s_t_.mean(-1, keepdim=True)
                    s_t_ = s_t_ - stmean
                    a_t_ = a_t_ - stmean

                    self.s += [s_t_]
                    self.a += [a_t_]
                    self.seq_files += [fname]

        self.s, self.a = torch.stack(self.s), torch.stack(self.a)

        # we only normalize the positions (flower and moth) to
        # be between 0 and 1
        # observations are not normalized
        if normalize:
            
            min_s, max_s = self.s.min(), self.s.max()
            min_a, max_a = self.a.min(), self.a.max()
            min_pos = min(min_s.item(), min_a.item())
            max_pos = max(max_s.item(), max_a.item())

            self.s = (self.s - min_pos) / (max_pos - min_pos)
            self.a = (self.a - min_pos) / (max_pos - min_pos)

        visual_x = compute_visual_obs(self.s, self.a, sigma=self.sigma, nu=self.nu, A=self.A) 
        if not self.mechanosensory:
            self.x = visual_x
        else:
            mech_x = generate_proboscis_observations_1d(self.s.cpu().numpy(), self.a.cpu().numpy(),
                                                        dt=1e-2, **self.mech_obs_params)
            mech_x = torch.tensor(mech_x).to(self.s)
            self.x_mech = mech_x
            self.x = torch.stack([visual_x, mech_x], dim=-1)
    
    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.x[idx]
    
class ShuffledMothDataset(MothDataset):
    """ This is the dataset where flower and moth trajectories are shuffled """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shuffle_trajectories()
    
    def shuffle_trajectories(self, seed=0):
        """
        Shuffles the data so that moth trajectories do not correspond to flower trajectories. Sensor observations
        are then generated based on the shuffled trajectories
        """

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # shuffle action trajectories
        a_t_shuffled = self.a[torch.randperm(len(self.a))]

        # recompute observations
        x_t_shuffled = compute_visual_obs(self.s, a_t_shuffled, self.sigma, self.nu, self.A)

        if self.mechanosensory:
            mech_x = generate_proboscis_observations_1d(self.s.cpu().numpy(), a_t_shuffled.cpu().numpy(),
                                                        dt=1e-2, **self.mech_obs_params)
            mech_x = torch.tensor(mech_x).to(self.s)
            self.x_mech = mech_x
            x_t_shuffled = torch.stack([x_t_shuffled, mech_x], dim=-1)

        self.a = a_t_shuffled
        self.x = x_t_shuffled

class DrunkMothDataset(MothDataset):
    """ This is a dataset where the action trajectories (moth locations) are generated as a random sum of sines """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.create_random_action_trajectories()
    
    def create_random_action_trajectories(self):
        
        # generate random action trajectories
        N, T = self.a.shape
        random_at = random_sum_of_sines(N, T, n_sinusoids=10, freq_range=(2., 4.))
        random_at = torch.tensor(random_at).to(self.a)

        # recompute observations
        random_xt = compute_visual_obs(self.s, random_at, self.sigma, self.nu, self.A)
        
        if self.mechanosensory:
            mech_x = generate_proboscis_observations_1d(self.s.cpu().numpy(), random_at.cpu().numpy(),
                                                        dt=1e-2, **self.mech_obs_params)
            mech_x = torch.tensor(mech_x).to(self.s)
            self.x_mech = mech_x
            random_xt = torch.stack([random_xt, mech_x], dim=-1)

        self.a = random_at
        self.x = random_xt

class DistractorMothDataset(MothDataset):
    """ 
    This is a dataset where there is a distractor moving in the environment and contributing
    to the visual observations that the moth gets    
    """

    def __init__(self, data_dir, filenames=None,
                 nu=1, sigma=0.1, A=1,
                 normalize=True,
                 seq_len=1000,
                 mechanosensory=False, mech_obs_params: dict = None,
                 dist_nu=1, dist_sigma=0.1, dist_A=1,):
        super().__init__(data_dir, filenames, nu, sigma, A, normalize, seq_len, mechanosensory, mech_obs_params)

        self.dist_sigma = dist_sigma
        self.dist_nu = dist_nu
        self.dist_A = dist_A

        self.create_distractor_data()
    
    def create_distractor_data(self, distactor_sigma=None, distractor_nu=None, distractor_A=None):

        if distactor_sigma is None:
            distactor_sigma = self.dist_sigma
        if distractor_A is None:
            distractor_A = self.dist_A
        if distractor_nu is None:
            distractor_nu = self.dist_nu

        distractor_s = torch.zeros_like(self.s)

        for seq in range(len(distractor_s)):
            start_val = self.s[seq].min().item() - 0.2
            stop_val = self.s[seq].max().item() + 0.2
            distractor_traj = ramp_schedule(self.seq_len,
                                           start_step=self.seq_len//2-25,
                                           stop_step=self.seq_len//2+25,
                                           start=start_val,
                                           stop=stop_val)
            distractor_s[seq] = torch.tensor(distractor_traj)
        
        self.distractor_s = distractor_s.to(self.s)
        
        self.distractor_x = compute_visual_obs(self.distractor_s, self.a,
                                               sigma=distactor_sigma,
                                               nu=distractor_nu,
                                               A=distractor_A)
        
        # in case of mechanosensory observations, distractor x needs to be 2D
        # but the mechanosensory signal for the distractor is always set to 0
        # since the moth's proboscis is not attached to it
        if self.mechanosensory:
            self.distractor_x = torch.stack([self.distractor_x,
                                           torch.zeros_like(self.distractor_x)], dim=-1)

        self.x = self.x + self.distractor_x
    
    def __getitem__(self, idx):
        return (self.s[idx], self.distractor_s[idx]), self.a[idx], self.x[idx]

class MothDatasetByCondition(MothDataset): # TODO: add mechanosensory case here 
    """ This is a dataset where trajectories are separated by species and light condition """

    def __init__(self, data_dir, filenames=None, nu=1, sigma=0.1, A=1, normalize=True, seq_len=1000,
                 condition_file_names=None, condition_sigmas=None,
                 mechanosensory=False, mech_obs_params=None):
        super().__init__(data_dir, filenames, nu, sigma, A, normalize, seq_len,
                         mechanosensory=mechanosensory, mech_obs_params=mech_obs_params)

        self.separate_by_condition(condition_file_names, condition_sigmas)
    
    def separate_by_condition(self, condition_file_names, condition_sigmas):
        """
        - condition_file_names correspond to the datasets from different species and light conditions
        - condition_sigmas correspond to the sigmas associated with different light conditions
        """

        fnms = np.array(self.seq_files)
        s_by_condition = {k1: {k2: self.s[fnms == condition_file_names[k1][k2]] for k2 in condition_file_names[k1].keys()}
                          for k1 in condition_file_names.keys()}
        
        a_by_condition = {
            k1: {k2: self.a[fnms == condition_file_names[k1][k2]] for k2 in condition_file_names[k1].keys()}
            for k1 in condition_file_names.keys()
        }

        x_by_condition = {k1: {} for k1 in condition_file_names.keys()}

        for k1 in condition_file_names.keys():
            for k2 in condition_file_names[k1].keys():
                x = compute_visual_obs(s_by_condition[k1][k2],
                                        a_by_condition[k1][k2],
                                        sigma=condition_sigmas[k1],
                                        nu=self.nu,
                                        A=self.A)
                if self.mechanosensory:
                    mech_x = generate_proboscis_observations_1d(s_by_condition[k1][k2].cpu().numpy(),
                                                                a_by_condition[k1][k2].cpu().numpy(),
                                                                dt=1e-2,
                                                                **self.mech_obs_params)
                    mech_x = torch.tensor(mech_x).to(x)
                    x = torch.stack([x, mech_x], dim=-1)
                
                x_by_condition[k1][k2] = x

        self.s, self.a, self.x = s_by_condition, a_by_condition, x_by_condition

####################################################################################################################################
#################################################### 2D IMAGE DATASET ##############################################################
####################################################################################################################################
class ImageMothDataset:
    """
    This will wrap a base moth dataset
    """

    def __init__(self, base_dataset, **renderer_kwargs):
        super().__init__()

        self.renderer = MothViewRenderer(**renderer_kwargs)

        self.base_dataset = base_dataset

        self.generate_sensory_frames()
    
    def generate_sensory_frames(self):

        # generate the observation frames
        flower_xy = [np.stack([si.numpy(), 0.5 * np.ones(si.shape)], axis=-1) for si in self.base_dataset.s]
        moth_xy = [np.stack([ai.numpy(), 0.5 * np.ones(ai.shape)], axis=-1) for ai in self.base_dataset.a]
        sensory_frames = np.array([self.renderer.render_frames(fxy, mxy) for fxy, mxy in zip(flower_xy, moth_xy)])
        sensory_frames = torch.Tensor(sensory_frames)
        x_frames = sensory_frames.flatten(start_dim=-2) / 255.       # normalized pixels

        if self.base_dataset.mechanosensory:
            self.x = torch.concatenate([x_frames,
                                        self.base_dataset.x_mech.unsqueeze(-1)], dim=-1)
        else:
            self.x = x_frames

    def shuffle_trajectories(self, seed=0):
        if isinstance(self.base_dataset, ShuffledMothDataset):
            self.base_dataset.shuffle_trajectories(seed=seed)
            self.generate_sensory_frames()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset.s[idx], self.base_dataset.a[idx], self.x[idx]


####################################################################################################################################
#################################################### ELECTRIC FISH DATASET #########################################################
####################################################################################################################################

class ElectricFishDataset(Dataset):
    """
    A PyTorch Dataset to load fish and shuttle trajectory data from
    "DarkData100fps" and "LightData100fps" directories.

    Attributes:
        fish_data (torch.Tensor): Shape [N, T, 3], storing (x, y, rotation) for fish.
        shuttle_data (torch.Tensor): Shape [N, T, 3], storing (x, y, rotation) for shuttle.
        subjects (list): List of length N with subject names corresponding to each trial.
        light_conditions (list): List of length N with either 'dark' or 'light' for each trial.
    """
    
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to the folder containing 'DarkData100fps' and 'LightData100fps'.
        """
        super().__init__()
        
        self.root_dir = root_dir
        self.fish_data_list = []
        self.shuttle_data_list = []
        self.subjects = []
        self.light_conditions = []
        
        # We expect two main directories: 'DarkData100fps' and 'LightData100fps'
        condition_dirs = ["DarkData100fps", "LightData100fps"]
        
        for condition_dir in condition_dirs:
            # Determine if this is dark or light
            light_label = "dark" if "Dark" in condition_dir.lower() else "light"
            
            full_cond_path = os.path.join(self.root_dir, condition_dir)
            
            # For each subject folder in that condition directory
            # e.g., Finn, Hope, Kun, Len, Ruby
            subject_folders = sorted(
                [d for d in os.listdir(full_cond_path) 
                 if os.path.isdir(os.path.join(full_cond_path, d))]
            )
            
            for subject in subject_folders:
                subject_path = os.path.join(full_cond_path, subject)
                
                # For each trial folder in the subject folder
                # e.g. trial01, trial02, ...
                trial_folders = sorted(
                    [d for d in os.listdir(subject_path) 
                     if os.path.isdir(os.path.join(subject_path, d))]
                )
                
                for trial_folder in trial_folders:
                    trial_path = os.path.join(subject_path, trial_folder)
                    
                    # Find fish and shuttle CSVs
                    fish_csv = glob.glob(os.path.join(trial_path, "fish_*_40s.csv"))
                    shuttle_csv = glob.glob(os.path.join(trial_path, "shuttle_*_40s.csv"))
                    
                    # We only proceed if both fish and shuttle CSV exist
                    if len(fish_csv) == 1 and len(shuttle_csv) == 1:
                        fish_csv_path = fish_csv[0]
                        shuttle_csv_path = shuttle_csv[0]
                        
                        # Load CSV data
                        fish_df = pd.read_csv(fish_csv_path, header=None)
                        shuttle_df = pd.read_csv(shuttle_csv_path, header=None)

                        # We want columns: x=column2, y=column3, rotation=column4
                        # (0-based indexing => columns: 2,3,4)
                        fish_values = fish_df.iloc[:, 2:5].values  # shape [T, 3]
                        shuttle_values = shuttle_df.iloc[:, 2:5].values  # shape [T, 3]

                        # Load the pixel-to-meter conversion factor
                        cv_factor_csv = os.path.join(trial_path, "pixel_size.csv")
                        conversion_factor = pd.read_csv(cv_factor_csv, header=None).to_numpy().item()
                        
                        # Convert x-y positions from pixel to meter
                        fish_values[:, 0] = fish_values[:, 0] * conversion_factor
                        fish_values[:, 1] = fish_values[:, 1] * conversion_factor
                        shuttle_values[:, 0] = shuttle_values[:, 0] * conversion_factor
                        shuttle_values[:, 1] = shuttle_values[:, 1] * conversion_factor
                        
                        # Append to our lists
                        self.fish_data_list.append(fish_values)
                        self.shuttle_data_list.append(shuttle_values)
                        self.subjects.append(subject)
                        self.light_conditions.append(light_label)
        
        # Convert to Tensors
        # shape => [N, T, 3]
        self.fish_data = torch.tensor(self.fish_data_list, dtype=torch.float32)
        self.shuttle_data = torch.tensor(self.shuttle_data_list, dtype=torch.float32)

    def __len__(self):
        """
        Number of trials in the dataset.
        """
        return len(self.fish_data)

    def __getitem__(self, idx):
        """
        Returns a tuple of (fish_trajectory, shuttle_trajectory, subject, light_condition)
        for the trial at index `idx`.
        """
        fish_traj = self.fish_data[idx]        # shape [T, 3]
        shuttle_traj = self.shuttle_data[idx]  # shape [T, 3]
        subject = self.subjects[idx]
        light_cond = self.light_conditions[idx]
        
        return fish_traj, shuttle_traj, subject, light_cond
