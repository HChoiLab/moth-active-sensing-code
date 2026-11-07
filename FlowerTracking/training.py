import torch
from torch.utils.data import DataLoader
from task import MothDataset, ShuffledMothDataset, ImageMothDataset
from model import AtemporalModel, RNNModel
from utils.data import random_sum_of_sines
from observation_models import generate_proboscis_observations_1d
from task import compute_visual_obs

def train_rnn_model(num_epochs=10, batch_size=4, learning_rate=1e-3, data_dir="data",
                    dataset=None, model=None, synthetic_action=False,
                    train_action_net=False, ur_weight=0.5,
                    shuffle_every_epoch=False):

    # Initialize dataset and DataLoader
    if dataset is None:
        dataset = MothDataset(data_dir=data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    if model is None:
        model = RNNModel(obs_dim=1, action_dim=1, latent_dim=1, hidden_dim=32)
    optimizer = torch.optim.Adam(model.perception_params, lr=learning_rate)
    action_optimizer = torch.optim.Adam(model.action_params, lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_div = 0

        if shuffle_every_epoch and (isinstance(dataset, ShuffledMothDataset) 
                                    or isinstance(dataset, ImageMothDataset)):
            dataset.shuffle_trajectories(seed=epoch)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for s_t_batch, a_t_batch, x_t_batch in data_loader:
                            
            if synthetic_action:
                # create synthetic action trajectory
                N, T = x_t_batch.shape[0], x_t_batch.shape[1]
                random_at = random_sum_of_sines(N, T, n_sinusoids=10, freq_range=(2., 4.))
                random_at = torch.tensor(random_at).to(x_t_batch)

                # recompute observations
                random_xt = compute_visual_obs(s_t_batch, random_at, dataset.sigma, dataset.nu, dataset.A)
                
                if dataset.mechanosensory:
                    mech_x = generate_proboscis_observations_1d(s_t_batch.cpu().numpy(), random_at.cpu().numpy(),
                                                                dt=1e-2, **dataset.mech_obs_params)
                    mech_x = torch.tensor(mech_x).to(s_t_batch)
                    random_xt = torch.stack([random_xt, mech_x], dim=-1)

                a_t_batch = random_at
                x_t_batch = random_xt

                if dataset.normalized:
                    x_t_batch = (x_t_batch - x_t_batch.min()) / (x_t_batch.max() - x_t_batch.min())
            
            # Move data to device if using GPU
            a_t_batch = a_t_batch.to(device)
            x_t_batch = x_t_batch.to(device)
            s_t_batch = s_t_batch.to(device)

            # Forward pass
            responses, losses = model(
                x_t_batch, a_t_batch
            )

            # Compute loss
            loss, recon_loss, kl_div = losses['total'], losses['mse'], losses['kl']

            # Action network training (this must happen before backwarding the perception loss)
            track_err = loss.detach() * 0.
            expected_ur = loss.detach() * 0.
            if train_action_net:
                # tracking loss
                track_err = ((responses['a_hat'].squeeze() - s_t_batch)**2).mean()

                # expected UR
                expected_ur = model.compute_expected_UR(responses['a_hat'], responses['h'])

                action_loss = (1 - ur_weight) * track_err - ur_weight * expected_ur

                action_optimizer.zero_grad()
                action_loss.backward(retain_graph=True)
                

            # Backpropagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_action_net:
                action_optimizer.step()            

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()

        avg_loss = total_loss / len(dataset)
        avg_recon_loss = total_recon_loss / len(dataset)
        avg_kl_div = total_kl_div / len(dataset)

        print(
            f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, '
            f'Recon Loss: {avg_recon_loss:.4f}, KL Div: {avg_kl_div:.4f}, Expected UR: {expected_ur:0.4f}, Tracking error: {track_err:0.4f}'
        )
    
    return dataset, model

def train_atemporal_model(num_epochs=10, batch_size=4, learning_rate=1e-3, data_dir="data", dataset=None, model=None):

    # Initialize dataset and DataLoader
    if dataset is None:
        dataset = MothDataset(data_dir=data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    if model is None:
        model = AtemporalModel(input_dim=2, latent_dim=1, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_div = 0

        for _, a_t_batch, x_t_batch in data_loader:
            # Move data to device if using GPU
            a_t_batch = a_t_batch.to(device)
            x_t_batch = x_t_batch.to(device)

            # Flatten batch and time dimensions
            batch_size, seq_length = x_t_batch.shape
            x_t_flat = x_t_batch.view(-1)
            a_t_flat = a_t_batch.view(-1)

            # Forward pass
            mu_x, mu_z, logvar_z, _ = model(
                x_t_flat, a_t_flat
            )

            # Compute loss
            loss, recon_loss, kl_div = model.compute_loss(
                x_t_flat, mu_x, mu_z, logvar_z
            )

            # Backpropagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()

        avg_loss = total_loss / len(dataset)
        avg_recon_loss = total_recon_loss / len(dataset)
        avg_kl_div = total_kl_div / len(dataset)

        print(
            f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, '
            f'Recon Loss: {avg_recon_loss:.4f}, KL Div: {avg_kl_div:.4f}'
        )
    
    return dataset, model