import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from task import compute_visual_obs
from observation_models import generate_proboscis_observations_1d
import math

def compute_entropy(logvar_z):
    return 0.5 * (1 + math.log(2 * math.pi)) + 0.5 * logvar_z

class RNNModel(nn.Module):

    def __init__(self, obs_dim=1, action_dim=1, latent_dim=1, hidden_dim=32, beta=0.1, enum_action:int=None):
        """
        Args:
            input_dim (int): Dimension of the input [x_t, a_{t-1}]
            latent_dim (int): Dimension of the latent variable s_t
            hidden_dim (int): Dimension of hidden layers
        """
        super(RNNModel, self).__init__()

        input_dim = obs_dim + action_dim

        self.beta = beta
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.enum_action = enum_action
        self.latent_dim = latent_dim

        # Posterior network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 64),
            nn.ReLU()   
        )
        self.enc_mu = nn.Linear(64, latent_dim)
        self.enc_logvar = nn.Linear(64, latent_dim)

        # RNN
        self.s2h = nn.Linear(latent_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)

        # Prior network
        self.prior = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, 64),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(64, latent_dim)
        self.prior_logvar = nn.Linear(64, latent_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),  # s_t and a_{t-1}
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

        # Action network
        self.actor_mu = nn.Linear(
            2*latent_dim + hidden_dim, action_dim
        )
        self.actor_logvar = nn.Linear(
            2*latent_dim + hidden_dim, action_dim
        )
    
    @property
    def perception_params(self):
        params = list(self.encoder.parameters()) + list(self.enc_mu.parameters()) + list(self.enc_logvar.parameters())
        params += list(self.s2h.parameters()) + list(self.h2h.parameters())
        params += list(self.prior.parameters()) + list(self.prior_mu.parameters()) + list(self.prior_logvar.parameters())
        params += list(self.decoder.parameters())
        return params

    @property
    def action_params(self):
        return list(self.actor_logvar.parameters()) + list(self.actor_mu.parameters()) + list(self.s2h.parameters()) + list(self.h2h.parameters())
    
    def encode_t(self, x_t, a_m_1, h_m_1):
        inputs = torch.cat([x_t, a_m_1, h_m_1], dim=-1)
        enc_h = self.encoder(inputs)
        mu = self.enc_mu(enc_h)
        logvar = self.enc_logvar(enc_h)
        return mu, logvar
    
    def prior_t(self, h_m_1, a_m_1):
        inputs = torch.cat([a_m_1, h_m_1], dim=-1)
        pr_h = self.prior(inputs)
        pr_mu = self.prior_mu(pr_h)
        pr_logvar = self.prior_logvar(pr_h)
        return pr_mu, pr_logvar

    def decode_t(self, s_t, a_m_1):
        inputs = torch.cat([s_t, a_m_1], dim=-1)
        return self.decoder(inputs)
    
    def act_t(self, post_mu, post_logvar, h_t):

        # concatenate
        actor_input = torch.concatenate([post_mu, post_logvar, h_t], dim=-1)

        # get action distribution
        action_mu = self.actor_mu(actor_input)
        action_logvar = self.actor_logvar(actor_input)

        # sample action
        action = action_mu + torch.randn_like(action_mu) * torch.exp(0.5 * action_logvar)

        return action
    
    @torch.no_grad
    def enumerate_actions(self, h, s, obs_fxn=None, z=None):
        
        candidate_array = torch.linspace(0, 1, self.enum_action)
        candidate_actions = candidate_array.reshape(1, -1, 1).repeat_interleave(dim=0, repeats=h.shape[0])
        h_t_rep = h.unsqueeze(1).repeat_interleave(dim=1, repeats=self.enum_action)
        s_t_rep = s.unsqueeze(1).repeat_interleave(dim=1, repeats=self.enum_action)

        # prior
        _, cand_prior_logvar = self.prior_t(h_t_rep.flatten(0, 1), candidate_actions.flatten(0, 1))

        # expected observation
        if obs_fxn is None:
            x_hat_cand = self.decode_t(s_t_rep.flatten(0, 1), candidate_actions.flatten(0, 1))
        else:
            z_t_rep = z.unsqueeze(1).repeat_interleave(dim=1, repeats=self.enum_action)
            x_hat_cand = obs_fxn(z_t_rep.flatten(0, 1), candidate_actions.flatten(0, 1)).reshape(-1, self.obs_dim)

        # expected posterior
        _, cand_posterior_logvar = self.encode_t(x_hat_cand, candidate_actions.flatten(0, 1), h_t_rep.flatten(0, 1))

        ent_diff = (compute_entropy(cand_prior_logvar) - compute_entropy(cand_posterior_logvar)).squeeze().detach()
        if self.latent_dim > 1: # TODO: do better
            ent_diff = ent_diff.sum(-1)
        ent_diff = ent_diff.reshape(h.shape[0], self.enum_action)
        
        # rescale
        #ent_diff = (ent_diff - ent_diff.min(-1, keepdim=True)[0]) / (ent_diff.max(-1, keepdim=True)[0] - ent_diff.min(-1, keepdim=True)[0])

        # select action with largest entropy reduction
        _, action_indices = ent_diff.max(dim=-1)
        a_hat = candidate_array[action_indices]

        return ent_diff, a_hat

    def forward_t(self, x_t, a_m_1, h_m_1, obs_fxn=None, z_t=None):

        # get prior
        prior_mu, prior_logvar = self.prior_t(h_m_1, a_m_1)

        # encode posterior
        post_mu, post_logvar = self.encode_t(x_t, a_m_1, h_m_1)

        # reparametrize
        s_t = self.reparameterize(post_mu, post_logvar)

        # decode 
        x_t_hat = self.decode_t(s_t, a_m_1)

        # update recurrent hidden state
        h_t = torch.tanh(self.s2h(s_t) + self.h2h(h_m_1))

        # output action 
        a_t_hat = torch.randn_like(s_t).clamp(0, 1)
        #a_t_hat = torch.sigmoid(self.act_t(post_mu.detach(), post_logvar.detach(), h_t))    # this is in the case of using a neural network for action selection

        ur_map = None
        if self.enum_action is not None:
            ur_map, a_t_hat = self.enumerate_actions(h_t.detach(), s_t.detach(), obs_fxn, z_t)

        # compute perception losses for this time step
        total_loss, recon_loss, kl_div = self.compute_loss(x_t, x_t_hat, post_mu, post_logvar, prior_mu, prior_logvar)

        # return responses and losses
        responses_t = {
            "x_hat": x_t_hat,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "s": s_t,
            "h": h_t,
            "a_hat": a_t_hat,
            "UR_map": ur_map
        }

        losses_t = {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_div
        }

        return responses_t, losses_t

    def forward(self, X, A, h_0=None, obs_fxn=None):
        
        batch_size, seq_len = X.shape[:2]

        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_dim).to(X)
        
        seq_responses = []

        mean_total, mean_mse, mean_kl = 0., 0., 0.

        for t in range(seq_len):

            x_t = X[:, t].reshape(-1, self.obs_dim)
            a_m_1 = A[:, t].reshape(-1, self.action_dim)
            h_t_minus_1 = h_0 if t == 0 else seq_responses[t-1]['h']
            
            responses_t, losses_t = self.forward_t(x_t, a_m_1, h_t_minus_1, obs_fxn)

            mean_total += losses_t['total'] / seq_len
            mean_mse += losses_t['recon'] / seq_len
            mean_kl += losses_t['kl'] / seq_len

            seq_responses.append(responses_t)
        
        responses = {
            k: torch.stack([d[k] for d in seq_responses], 1) for k in seq_responses[0].keys() if seq_responses[0][k] is not None
        }

        losses = {
            "total": mean_total,
            "mse": mean_mse,
            "kl": mean_kl
        }

        return responses, losses

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, x_t, x_t_hat, mu_z, logvar_z, prior_mu, prior_logvar):
        # Reconstruction loss (negative log-likelihood)
        recon_loss = F.mse_loss(x_t_hat.squeeze(), x_t.squeeze(), reduction='sum')

        # KL divergence between q(s_t | x_t, a_{t-1}) and p(s_t| h_{t-1}, a_{t-1})
        sigma_q, sigma_p = logvar_z.exp() ** 2, prior_logvar.exp() ** 2
        kl_divergence = 0.5 * (torch.log(sigma_p / sigma_q) + ((sigma_q + (mu_z - prior_mu) ** 2) / sigma_p) - 1)
        kl_divergence = kl_divergence.sum()

        # Total loss
        total_loss = recon_loss + self.beta * kl_divergence

        return total_loss, recon_loss, kl_divergence

    def compute_expected_UR(self, a, h):

        # prior
        prior_mu, prior_logvar = self.prior_t(h, a)

        # predicted observation
        s_hat = self.reparameterize(prior_mu, prior_logvar)    # TODO: we are only sampling once for now but we should avg across multiple samples

        # decode
        x_hat = self.decode_t(s_hat, a)

        # posterior
        _, post_logvar = self.encode_t(x_hat, a, h)

        # difference in entropy
        ur = compute_entropy(prior_logvar) - compute_entropy(post_logvar)

        return ur.mean()    

class ActiveModelEnv(nn.Module):

    def __init__(self, dataset, perception_model):
        super().__init__()

        self.dataset = dataset
        self.perception_model = perception_model
    
    def compute_observation(self, s, a):

        visual_x = compute_visual_obs(s, a, sigma=self.dataset.sigma, nu=self.dataset.nu, A=self.dataset.A) 
        if not self.dataset.mechanosensory:
            x = visual_x
        else:
            mech_x = generate_proboscis_observations_1d(s.cpu().numpy(), a.cpu().numpy(),
                                                        dt=1e-2, **self.dataset.mech_obs_params)
            mech_x = torch.tensor(mech_x).to(s)
            x = torch.stack([visual_x, mech_x], dim=-1)
        
        return x
    
    def forward(self, S, h_0=None):

        batch_size, seq_len = S.shape[:2]

        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.perception_model.hidden_dim).to(S)
        
        seq_responses = []

        a_0 = 0.5 * torch.ones_like(S[:, 0])

        for t in range(seq_len):

            a_m_1 = a_0 if t == 0 else seq_responses[-1]['a_hat']
            a_m_1 = a_m_1.reshape(-1, self.perception_model.action_dim)
            h_t_minus_1 = h_0 if t == 0 else seq_responses[t-1]['h']
            x_t = self.compute_observation(S[:, [t]], a_m_1).reshape(S.shape[0], self.perception_model.obs_dim)

            responses_t, _ = self.perception_model.forward_t(x_t, a_m_1, h_t_minus_1, obs_fxn=self.compute_observation, z_t=S[:, [t]])
            responses_t['x'] = x_t

            seq_responses.append(responses_t)
        
        responses = {
            k: torch.stack([d[k] for d in seq_responses], 1) for k in seq_responses[0].keys() if seq_responses[0][k] is not None
        }

        return responses

###################################################################################################################
###################################################################################################################
###################################################################################################################

class AtemporalModel(nn.Module):
    def __init__(self, input_dim=2, latent_dim=1, hidden_dim=32, beta=0.1):
        """
        Args:
            input_dim (int): Dimension of the input [x_t, a_{t-1}]
            latent_dim (int): Dimension of the latent variable s_t
            hidden_dim (int): Dimension of hidden layers
        """
        super(AtemporalModel, self).__init__()

        self.beta = beta

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # s_t and a_{t-1}
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU()
        )
        self.output_mu = nn.Linear(hidden_dim, 1)  # Output mean of x_t

    def encode(self, x_t, a_t_minus_1):
        # Concatenate x_t and a_{t-1}
        inputs = torch.cat([x_t.unsqueeze(1), a_t_minus_1.unsqueeze(1)], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, s_t, a_t_minus_1):
        # Concatenate s_t and a_{t-1}
        inputs = torch.cat([s_t, a_t_minus_1.unsqueeze(1)], dim=1)
        h = self.decoder(inputs)
        mu_x = self.output_mu(h)
        return mu_x

    def forward(self, x_t, a_t_minus_1):
        # Encode
        mu_z, logvar_z = self.encode(x_t, a_t_minus_1)
        # Reparameterize
        s_t = self.reparameterize(mu_z, logvar_z)
        # Decode
        mu_x = self.decode(s_t, a_t_minus_1)
        return mu_x, mu_z, logvar_z, s_t

    def compute_loss(self, x_t, mu_x, mu_z, logvar_z):
        # Reconstruction loss (negative log-likelihood)
        recon_loss = F.mse_loss(mu_x.squeeze(), x_t, reduction='sum')

        # KL divergence between q(s_t | x_t, a_{t-1}) and p(s_t)
        kl_divergence = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_divergence

        return total_loss, recon_loss, kl_divergence
