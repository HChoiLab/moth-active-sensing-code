import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import ContinuousBernoulli

class AllocentricNectarySearchModel(nn.Module):
    """
    This corresponds to the allocentric nectary search environment where moth locations
    are in absolute/world coordinates and are observable by the moth
    """

    MOE_POSTERIOR_FUSIONS = {"uniform_moe", "weighted_moe"}
    SUPPORTED_POSTERIOR_FUSIONS = {"early", "poe", "uniform_moe", "weighted_moe"}
    SUPPORTED_RNN_TYPES = {"rnn", "lstm"}
    SUPPORTED_MODALITY_RNN_MODES = {"shared", "separate"}

    @staticmethod
    def _group_norm_groups(num_channels, max_groups=4):
        for num_groups in range(min(max_groups, num_channels), 0, -1):
            if num_channels % num_groups == 0:
                return num_groups
        return 1

    @classmethod
    def _normalize_posterior_fusion(cls, posterior_fusion):
        normalized = str(posterior_fusion or "early").strip().lower().replace("-", "_")
        aliases = {
            "early_fusion": "early",
            "product_of_experts": "poe",
            "product": "poe",
            "moe": "weighted_moe",
            "mixture_of_experts": "weighted_moe",
            "mixture": "weighted_moe",
            "weighted_mixture_of_experts": "weighted_moe",
            "weighted_mixture": "weighted_moe",
            "moe_weighted": "weighted_moe",
            "uniform_mixture_of_experts": "uniform_moe",
            "uniform_mixture": "uniform_moe",
            "moe_uniform": "uniform_moe",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.SUPPORTED_POSTERIOR_FUSIONS:
            raise ValueError(
                f"Unsupported posterior_fusion '{posterior_fusion}'. "
                f"Expected one of {sorted(cls.SUPPORTED_POSTERIOR_FUSIONS)}."
            )
        return normalized

    @classmethod
    def _normalize_rnn_type(cls, rnn_type):
        normalized = str(rnn_type or "rnn").strip().lower().replace("-", "_")
        aliases = {
            "simple": "rnn",
            "vanilla": "rnn",
            "vanilla_rnn": "rnn",
            "lstm_cell": "lstm",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.SUPPORTED_RNN_TYPES:
            raise ValueError(
                f"Unsupported rnn_type '{rnn_type}'. "
                f"Expected one of {sorted(cls.SUPPORTED_RNN_TYPES)}."
            )
        return normalized

    @classmethod
    def _normalize_modality_rnn_mode(cls, modality_rnn_mode):
        normalized = str(modality_rnn_mode or "shared").strip().lower().replace("-", "_")
        aliases = {
            "joint": "shared",
            "fused": "shared",
            "shared_fusion": "shared",
            "separate_modalities": "separate",
            "modality_separate": "separate",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.SUPPORTED_MODALITY_RNN_MODES:
            raise ValueError(
                f"Unsupported modality_rnn_mode '{modality_rnn_mode}'. "
                f"Expected one of {sorted(cls.SUPPORTED_MODALITY_RNN_MODES)}."
            )
        return normalized

    @staticmethod
    def _build_gaussian_encoder(input_dim, hidden_dim=64):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _build_moe_weight_net(input_dim, num_experts=2, hidden_dim=32):
        hidden_dim = min(hidden_dim, input_dim)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def __init__(self, mech_obs_dim,
                 vis_obs_shape,
                 action_dim,
                 mech_embedding_dim=None,
                 vis_embedding_dim=64,
                 fusion_dim=None,
                 latent_dim=1,
                 hidden_dim=64,
                 beta=0.1,
                 visual_likelihood="gaussian",
                 learn_decoder_variance=True,
                 posterior_fusion="early",
                 moment_matching=True,
                 rnn_type="rnn",
                 modality_rnn_mode="shared"):

        super().__init__()

        visual_likelihood = str(visual_likelihood).lower()
        supported_visual_likelihoods = {"gaussian", "continuous_bernoulli"}
        if visual_likelihood not in supported_visual_likelihoods:
            raise ValueError(
                f"Unsupported visual_likelihood '{visual_likelihood}'. "
                f"Expected one of {sorted(supported_visual_likelihoods)}."
            )
        posterior_fusion = self._normalize_posterior_fusion(posterior_fusion)
        rnn_type = self._normalize_rnn_type(rnn_type)
        modality_rnn_mode = self._normalize_modality_rnn_mode(modality_rnn_mode)
        if modality_rnn_mode == "separate" and posterior_fusion == "early":
            raise ValueError(
                "modality_rnn_mode='separate' is only supported for "
                "posterior_fusion='poe' or a MoE fusion."
            )
        if modality_rnn_mode == "separate" and rnn_type != "rnn":
            raise ValueError(
                "modality_rnn_mode='separate' is only supported for rnn_type='rnn'."
            )

        self.beta = beta
        self.mech_obs_dim = mech_obs_dim
        self.vis_obs_shape = vis_obs_shape
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.visual_likelihood = visual_likelihood
        self.learn_decoder_variance = bool(learn_decoder_variance)
        self.posterior_fusion = posterior_fusion
        self.moment_matching = bool(moment_matching)
        if not self.moment_matching and self.posterior_fusion not in self.MOE_POSTERIOR_FUSIONS:
            raise ValueError(
                "moment_matching=False is only supported for MoE posterior fusions."
            )
        self.rnn_type = rnn_type
        self.modality_rnn_mode = modality_rnn_mode
        self.use_separate_modality_rnns = self.modality_rnn_mode == "separate"
        self.recurrent_feature_dim = (
            2 * self.hidden_dim if self.use_separate_modality_rnns else self.hidden_dim
        )
        self.mech_cont_dim = max(0, self.mech_obs_dim - 1)
        self.vis_numel = vis_obs_shape[0] * vis_obs_shape[1] * vis_obs_shape[2]
        self.vis_channels, self.vis_height, self.vis_width = vis_obs_shape
        self.visual_hidden_channels = 16
        self.visual_pool_shape = (4, 4)
        self.visual_downsample_shape = (
            math.ceil(self.vis_height / 2),
            math.ceil(self.vis_width / 2),
        )
        self.visual_output_padding = (
            1 - (self.vis_height % 2),
            1 - (self.vis_width % 2),
        )
        self.visual_uses_conv = bool(vis_embedding_dim)
        self.decoder_variance_multiplier = 2 if self.learn_decoder_variance else 1
        self.visual_param_multiplier = (
            self.decoder_variance_multiplier
            if self.visual_likelihood == "gaussian"
            else 1
        )

        #############################
        #  Modality embedding layers
        ##############################

        # Mechanosensory 
        if mech_embedding_dim:
            mech_encoder = nn.Linear(mech_obs_dim, mech_embedding_dim)
        else:
            mech_encoder = nn.Identity()
            mech_embedding_dim = mech_obs_dim
        
        # Visual
        if self.visual_uses_conv:
            visual_feature_dim = vis_embedding_dim * self.visual_pool_shape[0] * self.visual_pool_shape[1]
            visual_encoder = nn.Sequential(
                nn.Conv2d(self.vis_channels, self.visual_hidden_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(self._group_norm_groups(self.visual_hidden_channels), self.visual_hidden_channels),
                nn.SiLU(),
                nn.Conv2d(self.visual_hidden_channels, vis_embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(self._group_norm_groups(vis_embedding_dim), vis_embedding_dim),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(self.visual_pool_shape),
                nn.Flatten(start_dim=-3),
                nn.Linear(visual_feature_dim, vis_embedding_dim),
            )
        else:
            visual_encoder = nn.Flatten(start_dim=-3)
            vis_embedding_dim = self.vis_numel
        
        # Fusing
        if fusion_dim:
            fusion_layer = nn.Sequential(nn.Linear(vis_embedding_dim + mech_embedding_dim, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, fusion_dim))
        else:
            fusion_layer = nn.Identity()
            fusion_dim = vis_embedding_dim + mech_embedding_dim
        
        self.mech_encoder = mech_encoder
        self.visual_encoder = visual_encoder
        self.fusion_layer = fusion_layer
        self.obs_dim = fusion_dim     # the fused features are the final observations to the model
        
        #####################
        # VAE layers 
        #####################

        # RNN
        recurrent_input_dim = self.obs_dim + self.action_dim
        if self.rnn_type == "rnn" and not self.use_separate_modality_rnns:
            self.y2h = nn.Linear(recurrent_input_dim, self.hidden_dim)
            self.h2h = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.lstm_cell = None
            self.mech_y2h = None
            self.mech_h2h = None
            self.vis_y2h = None
            self.vis_h2h = None
        elif self.use_separate_modality_rnns:
            self.y2h = None
            self.h2h = None
            self.lstm_cell = None
            self.mech_y2h = nn.Linear(mech_embedding_dim + self.action_dim, self.hidden_dim)
            self.mech_h2h = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.vis_y2h = nn.Linear(vis_embedding_dim + self.action_dim, self.hidden_dim)
            self.vis_h2h = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.y2h = None
            self.h2h = None
            self.lstm_cell = nn.LSTMCell(recurrent_input_dim, self.hidden_dim)
            self.mech_y2h = None
            self.mech_h2h = None
            self.vis_y2h = None
            self.vis_h2h = None

        # Posterior network. The default keeps the original early-fusion path.
        if self.posterior_fusion == "early":
            self.encoder = self._build_gaussian_encoder(self.hidden_dim)
            self.enc_mu = nn.Linear(64, self.latent_dim)
            self.enc_logvar = nn.Linear(64, self.latent_dim)
            self.mech_posterior_encoder = None
            self.mech_post_mu = None
            self.mech_post_logvar = None
            self.vis_posterior_encoder = None
            self.vis_post_mu = None
            self.vis_post_logvar = None
            self.moe_weight_net = None
        else:
            self.encoder = None
            self.enc_mu = None
            self.enc_logvar = None
            if self.use_separate_modality_rnns:
                mech_posterior_input_dim = self.hidden_dim
                vis_posterior_input_dim = self.hidden_dim
            else:
                modality_context_dim = self.action_dim + self.hidden_dim
                mech_posterior_input_dim = mech_embedding_dim + modality_context_dim
                vis_posterior_input_dim = vis_embedding_dim + modality_context_dim
            self.mech_posterior_encoder = self._build_gaussian_encoder(
                mech_posterior_input_dim
            )
            self.mech_post_mu = nn.Linear(64, self.latent_dim)
            self.mech_post_logvar = nn.Linear(64, self.latent_dim)
            self.vis_posterior_encoder = self._build_gaussian_encoder(
                vis_posterior_input_dim
            )
            self.vis_post_mu = nn.Linear(64, self.latent_dim)
            self.vis_post_logvar = nn.Linear(64, self.latent_dim)
            self.moe_weight_net = (
                self._build_moe_weight_net(self.recurrent_feature_dim)
                if self.posterior_fusion == "weighted_moe"
                else None
            )

        # Decoder network
        self.mech_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1 + (self.decoder_variance_multiplier * self.mech_cont_dim))
        )

        if self.visual_uses_conv:
            self.vis_decoder = nn.Sequential(
                nn.Linear(self.latent_dim + self.action_dim, visual_feature_dim),
                nn.SiLU(),
                nn.Unflatten(-1, (vis_embedding_dim, *self.visual_pool_shape)),
                nn.Upsample(size=self.visual_downsample_shape, mode="bilinear", align_corners=False),
                nn.ConvTranspose2d(
                    vis_embedding_dim,
                    self.visual_hidden_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=self.visual_output_padding,
                ),
                nn.GroupNorm(self._group_norm_groups(self.visual_hidden_channels), self.visual_hidden_channels),
                nn.SiLU(),
                nn.ConvTranspose2d(
                    self.visual_hidden_channels,
                    self.visual_param_multiplier * self.vis_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        else:
            self.vis_decoder = nn.Sequential(
                nn.Linear(self.latent_dim + self.action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.visual_param_multiplier * self.vis_numel),
            )
    
    def embed_modalities(self, mech, vis):
        mech_embedding = self.mech_encoder(mech)
        vis_embedding = self.visual_encoder(vis)
        return mech_embedding, vis_embedding

    def fuse_embeddings(self, mech_embedding, vis_embedding):
        return self.fusion_layer(torch.cat([mech_embedding, vis_embedding], dim=-1))

    def integrate_multisensory(self, mech, vis):
        mech_embedding, vis_embedding = self.embed_modalities(mech, vis)
        return self.fuse_embeddings(mech_embedding, vis_embedding)

    def _split_recurrent_state(self, recurrent_state):
        if self.rnn_type == "lstm" and isinstance(recurrent_state, tuple):
            return recurrent_state
        return recurrent_state, None

    def _hidden_from_recurrent_state(self, recurrent_state):
        h_t, _ = self._split_recurrent_state(recurrent_state)
        return h_t

    def _split_modality_hidden(self, recurrent_state):
        hidden = self._hidden_from_recurrent_state(recurrent_state)
        expected_dim = 2 * self.hidden_dim
        if hidden.shape[-1] != expected_dim:
            raise ValueError(
                "Separate modality RNN state must have last dimension "
                f"{expected_dim}; got {hidden.shape[-1]}."
            )
        return torch.split(hidden, self.hidden_dim, dim=-1)

    def initial_hidden(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        hidden_dim = self.recurrent_feature_dim
        h0 = torch.zeros(batch_size, hidden_dim, device=device, dtype=dtype)
        if self.rnn_type != "lstm":
            return h0
        return h0, torch.zeros_like(h0)

    def update_hidden(self, x_t, a_m_1, h_m_1):
        y_t = torch.cat([x_t, a_m_1], dim=-1)
        h_prev, c_prev = self._split_recurrent_state(h_m_1)
        if self.rnn_type == "rnn":
            if self.use_separate_modality_rnns:
                raise RuntimeError(
                    "update_hidden is not used when modality_rnn_mode='separate'."
                )
            return torch.tanh(self.y2h(y_t) + self.h2h(h_prev))
        if c_prev is None:
            c_prev = torch.zeros_like(h_prev)
        return self.lstm_cell(y_t, (h_prev, c_prev))

    @staticmethod
    def update_modality_hidden(embedding, a_m_1, h_prev, y2h, h2h):
        y_t = torch.cat([embedding, a_m_1], dim=-1)
        return torch.tanh(y2h(y_t) + h2h(h_prev))

    def encode_t(self, x_t, a_m_1, h_m_1):
        recurrent_state_t = self.update_hidden(x_t, a_m_1, h_m_1)
        h_t = self._hidden_from_recurrent_state(recurrent_state_t)
        enc_h = self.encoder(h_t)
        mu = self.enc_mu(enc_h)
        logvar = self.enc_logvar(enc_h)
        
        return mu, logvar, recurrent_state_t

    def encode_modality_t(self, embedding, a_m_1, h_m_1, encoder, mu_head, logvar_head):
        posterior_input = torch.cat([embedding, a_m_1, h_m_1], dim=-1)
        enc_h = encoder(posterior_input)
        return mu_head(enc_h), logvar_head(enc_h)

    def encode_modality_hidden_t(self, h_t, encoder, mu_head, logvar_head):
        enc_h = encoder(h_t)
        return mu_head(enc_h), logvar_head(enc_h)

    @staticmethod
    def _aggregate_poe(mus, logvars, prior_mu=None, prior_logvar=None, min_var=1e-8):
        expert_mus = list(mus)
        expert_logvars = list(logvars)
        if prior_mu is not None and prior_logvar is not None:
            expert_mus.append(prior_mu)
            expert_logvars.append(prior_logvar)

        precisions = [torch.exp(-logvar) for logvar in expert_logvars]
        precision_sum = torch.stack(precisions, dim=0).sum(dim=0).clamp_min(min_var)
        weighted_mu_sum = torch.stack(
            [mu * precision for mu, precision in zip(expert_mus, precisions)],
            dim=0,
        ).sum(dim=0)
        mu = weighted_mu_sum / precision_sum
        var = precision_sum.reciprocal().clamp_min(min_var)
        return mu, var.log()

    @staticmethod
    def _aggregate_moe(mus, logvars, mixture_weights=None, min_var=1e-8):
        stacked_mu = torch.stack(list(mus), dim=-2)
        stacked_var = torch.stack([logvar.exp() for logvar in logvars], dim=-2)
        if mixture_weights is None:
            mixture_weights = stacked_mu.new_full(
                stacked_mu.shape[:-1],
                1.0 / stacked_mu.shape[-2],
            )
        mixture_weights = mixture_weights.unsqueeze(-1)
        mu = (mixture_weights * stacked_mu).sum(dim=-2)
        second_moment = (
            mixture_weights * (stacked_var + stacked_mu.pow(2))
        ).sum(dim=-2)
        var = (second_moment - mu.pow(2)).clamp_min(min_var)
        return mu, var.log()

    def aggregate_modality_posteriors(
        self,
        mech_mu,
        mech_logvar,
        vis_mu,
        vis_logvar,
        prior_mu,
        prior_logvar,
        mixture_weights=None,
    ):
        mus = (mech_mu, vis_mu)
        logvars = (mech_logvar, vis_logvar)
        if self.posterior_fusion == "poe":
            return self._aggregate_poe(
                mus,
                logvars,
                prior_mu=prior_mu,
                prior_logvar=prior_logvar,
            )
        if self.posterior_fusion in self.MOE_POSTERIOR_FUSIONS:
            return self._aggregate_moe(
                mus,
                logvars,
                mixture_weights=mixture_weights,
            )
        raise RuntimeError(
            "aggregate_modality_posteriors is only used when "
            "posterior_fusion is 'poe' or a MoE fusion."
        )

    def encode_observation_t(self, x_mech_t, x_vis_t, a_m_1, h_m_1):
        mech_embedding, vis_embedding = self.embed_modalities(x_mech_t, x_vis_t)
        x_t = self.fuse_embeddings(mech_embedding, vis_embedding)

        modality_posteriors = {}
        if self.posterior_fusion == "early":
            post_mu, post_logvar, recurrent_state_t = self.encode_t(x_t, a_m_1, h_m_1)
            h_t, c_t = self._split_recurrent_state(recurrent_state_t)
            prior_mu = torch.zeros_like(post_mu)
            prior_logvar = torch.zeros_like(post_logvar)
        else:
            if self.use_separate_modality_rnns:
                h_mech_prev, h_vis_prev = self._split_modality_hidden(h_m_1)
                h_mech_t = self.update_modality_hidden(
                    mech_embedding,
                    a_m_1,
                    h_mech_prev,
                    self.mech_y2h,
                    self.mech_h2h,
                )
                h_vis_t = self.update_modality_hidden(
                    vis_embedding,
                    a_m_1,
                    h_vis_prev,
                    self.vis_y2h,
                    self.vis_h2h,
                )
                h_t = torch.cat([h_mech_t, h_vis_t], dim=-1)
                c_t = None
                mech_mu, mech_logvar = self.encode_modality_hidden_t(
                    h_mech_t,
                    self.mech_posterior_encoder,
                    self.mech_post_mu,
                    self.mech_post_logvar,
                )
                vis_mu, vis_logvar = self.encode_modality_hidden_t(
                    h_vis_t,
                    self.vis_posterior_encoder,
                    self.vis_post_mu,
                    self.vis_post_logvar,
                )
            else:
                recurrent_state_t = self.update_hidden(x_t, a_m_1, h_m_1)
                h_t, c_t = self._split_recurrent_state(recurrent_state_t)
                shared_h_prev = self._hidden_from_recurrent_state(h_m_1)
                mech_mu, mech_logvar = self.encode_modality_t(
                    mech_embedding,
                    a_m_1,
                    shared_h_prev,
                    self.mech_posterior_encoder,
                    self.mech_post_mu,
                    self.mech_post_logvar,
                )
                vis_mu, vis_logvar = self.encode_modality_t(
                    vis_embedding,
                    a_m_1,
                    shared_h_prev,
                    self.vis_posterior_encoder,
                    self.vis_post_mu,
                    self.vis_post_logvar,
                )
            prior_mu = torch.zeros_like(mech_mu)
            prior_logvar = torch.zeros_like(mech_logvar)
            if self.posterior_fusion == "weighted_moe":
                moe_weights = F.softmax(self.moe_weight_net(h_t), dim=-1)
            elif self.posterior_fusion == "uniform_moe":
                moe_weights = h_t.new_full((*h_t.shape[:-1], 2), 0.5)
            else:
                moe_weights = None
            post_mu, post_logvar = self.aggregate_modality_posteriors(
                mech_mu,
                mech_logvar,
                vis_mu,
                vis_logvar,
                prior_mu,
                prior_logvar,
                mixture_weights=moe_weights,
            )
            modality_posteriors = {
                "mech_post_mu": mech_mu,
                "mech_post_logvar": mech_logvar,
                "vis_post_mu": vis_mu,
                "vis_post_logvar": vis_logvar,
            }
            if self.use_separate_modality_rnns:
                modality_posteriors["h_mech"] = h_mech_t
                modality_posteriors["h_vis"] = h_vis_t
            if moe_weights is not None:
                modality_posteriors["moe_weights"] = moe_weights
        if c_t is not None:
            modality_posteriors["c"] = c_t

        return post_mu, post_logvar, prior_mu, prior_logvar, h_t, modality_posteriors

    @staticmethod
    def _gaussian_nll_from_logvar(
        target: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        reduction: str = "mean",
        min_logvar: float = -8.0,
        max_logvar: float = 8.0,
    ) -> torch.Tensor:
        logvar = logvar.clamp(min=min_logvar, max=max_logvar)
        loss = 0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (target - mean).pow(2) * torch.exp(-logvar)
        )
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction '{reduction}'.")

    @staticmethod
    def _continuous_bernoulli_probs(
        probs: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return probs.clamp(min=eps, max=1.0 - eps)

    @classmethod
    def _continuous_bernoulli_nll(
        cls,
        target: torch.Tensor,
        probs: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        dist = ContinuousBernoulli(probs=cls._continuous_bernoulli_probs(probs))
        loss = -dist.log_prob(target)
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction '{reduction}'.")

    def decode_t(self, s, a):
        decoder_input = torch.cat([s, a], dim=-1)
        mech_params = self.mech_decoder(decoder_input)
        mech_binary_logit = mech_params[..., :1]
        if self.mech_cont_dim > 0:
            if self.learn_decoder_variance:
                mech_cont_mean, mech_cont_logvar = torch.split(
                    mech_params[..., 1:],
                    self.mech_cont_dim,
                    dim=-1,
                )
            else:
                mech_cont_mean = mech_params[..., 1:]
                mech_cont_logvar = torch.zeros_like(mech_cont_mean)
            mech_output = torch.cat([mech_binary_logit, mech_cont_mean], dim=-1)
        else:
            mech_cont_logvar = mech_binary_logit.new_zeros(*mech_binary_logit.shape[:-1], 0)
            mech_output = mech_binary_logit

        vis_params = self.vis_decoder(decoder_input)
        expected_vis_params = self.visual_param_multiplier * self.vis_numel
        if vis_params.shape[-1] != expected_vis_params:
            vis_params = torch.flatten(vis_params, start_dim=-3)
        if self.visual_likelihood == "gaussian":
            if self.learn_decoder_variance:
                vis_mean, vis_logvar = torch.split(vis_params, self.vis_numel, dim=-1)
            else:
                vis_mean = vis_params
                vis_logvar = torch.zeros_like(vis_mean)
            vis_output = torch.sigmoid(vis_mean)
        else:
            vis_output = torch.sigmoid(vis_params)
            vis_logvar = None
        return mech_output, mech_cont_logvar, vis_output, vis_logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _align_target_to_prediction(target, prediction):
        while target.ndim < prediction.ndim:
            target = target.unsqueeze(-2)
        return target.expand_as(prediction)

    @staticmethod
    def _gaussian_latent_log_prob(value, mu, logvar):
        return -0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (value - mu).pow(2) * torch.exp(-logvar)
        ).sum(dim=-1)

    @staticmethod
    def _expand_action_for_latents(action, latents):
        while action.ndim < latents.ndim:
            action = action.unsqueeze(-2)
        return action.expand(*latents.shape[:-1], action.shape[-1])

    def _reconstruction_nll_terms(
        self,
        x_mech_t,
        x_vis_t,
        x_mech_t_hat,
        x_mech_t_logvar,
        x_vis_t_hat,
        x_vis_t_logvar,
    ):
        x_mech_target = self._align_target_to_prediction(x_mech_t, x_mech_t_hat)
        binary_nll = F.binary_cross_entropy_with_logits(
            x_mech_t_hat[..., 0],
            x_mech_target[..., 0],
            reduction='none',
        )
        if self.mech_cont_dim > 0:
            mech_logvar = self._align_target_to_prediction(
                x_mech_t_logvar,
                x_mech_t_hat[..., 1:],
            )
            mech_cont_nll = self._gaussian_nll_from_logvar(
                x_mech_target[..., 1:],
                x_mech_t_hat[..., 1:],
                mech_logvar,
                reduction='none',
            ).sum(-1)
        else:
            mech_cont_nll = torch.zeros_like(binary_nll)
        mech_nll = binary_nll + mech_cont_nll

        flat_x_vis_t = torch.flatten(x_vis_t, start_dim=-3)
        flat_x_vis_t = self._align_target_to_prediction(flat_x_vis_t, x_vis_t_hat)
        if self.visual_likelihood == "gaussian":
            vis_logvar = self._align_target_to_prediction(x_vis_t_logvar, x_vis_t_hat)
            vis_nll = self._gaussian_nll_from_logvar(
                flat_x_vis_t,
                x_vis_t_hat,
                vis_logvar,
                reduction='none',
            ).sum(-1)
        else:
            vis_nll = self._continuous_bernoulli_nll(
                flat_x_vis_t,
                x_vis_t_hat,
                reduction='none',
            ).sum(-1)

        recon_nll = 50. * mech_nll + vis_nll
        return mech_nll, vis_nll, recon_nll

    def compute_loss(self, x_mech_t, x_vis_t, x_mech_t_hat, x_mech_t_logvar, x_vis_t_hat, x_vis_t_logvar,
                     post_mu, post_logvar, prior_mu, prior_logvar, return_per_sample=False):

        ##################################################
        #  Reconstruction loss (negative log-likelihood)
        ##################################################
        _, _, recon_nll = self._reconstruction_nll_terms(
            x_mech_t,
            x_vis_t,
            x_mech_t_hat,
            x_mech_t_logvar,
            x_vis_t_hat,
            x_vis_t_logvar,
        )
        recon_loss = recon_nll.mean()

        ##################################################
        # KL divergence
        ##################################################
        sigma_q, sigma_p = post_logvar.exp(), prior_logvar.exp()
        kl_per_sample = 0.5 * (torch.log(sigma_p / sigma_q) + ((sigma_q + (post_mu - prior_mu) ** 2) / sigma_p) - 1)
        kl_per_sample = kl_per_sample.sum(-1)
        kl_divergence = kl_per_sample.mean()

        # Total loss
        total_loss = recon_loss + self.beta * kl_divergence

        if return_per_sample:
            total_per_sample = recon_nll + self.beta * kl_per_sample
            return (
                total_loss,
                recon_loss,
                kl_divergence,
                total_per_sample,
                recon_nll,
                kl_per_sample,
            )

        return total_loss, recon_loss, kl_divergence

    def compute_moe_mixture_loss(
        self,
        x_mech_t,
        x_vis_t,
        a_m_1,
        prior_mu,
        prior_logvar,
        modality_posteriors,
        return_per_sample=False,
    ):
        mech_mu = modality_posteriors["mech_post_mu"]
        mech_logvar = modality_posteriors["mech_post_logvar"]
        vis_mu = modality_posteriors["vis_post_mu"]
        vis_logvar = modality_posteriors["vis_post_logvar"]
        expert_mu = torch.stack([mech_mu, vis_mu], dim=-2)
        expert_logvar = torch.stack([mech_logvar, vis_logvar], dim=-2)
        mixture_weights = modality_posteriors.get("moe_weights")
        if mixture_weights is None:
            mixture_weights = expert_mu.new_full(expert_mu.shape[:-1], 0.5)
        mixture_weights = mixture_weights.clamp_min(1e-8)
        mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True)

        z_samples = self.reparameterize(expert_mu, expert_logvar)
        expanded_action = self._expand_action_for_latents(a_m_1, z_samples)
        sample_shape = z_samples.shape[:-1]
        flat_z_samples = z_samples.reshape(-1, z_samples.shape[-1])
        flat_action = expanded_action.reshape(-1, expanded_action.shape[-1])
        x_mech_hat, x_mech_logvar, x_vis_hat, x_vis_logvar = self.decode_t(
            flat_z_samples,
            flat_action,
        )
        x_mech_hat = x_mech_hat.reshape(*sample_shape, x_mech_hat.shape[-1])
        x_mech_logvar = x_mech_logvar.reshape(*sample_shape, x_mech_logvar.shape[-1])
        x_vis_hat = x_vis_hat.reshape(*sample_shape, x_vis_hat.shape[-1])
        if x_vis_logvar is not None:
            x_vis_logvar = x_vis_logvar.reshape(*sample_shape, x_vis_logvar.shape[-1])
        _, _, recon_nll = self._reconstruction_nll_terms(
            x_mech_t,
            x_vis_t,
            x_mech_hat,
            x_mech_logvar,
            x_vis_hat,
            x_vis_logvar,
        )

        component_log_probs = self._gaussian_latent_log_prob(
            z_samples.unsqueeze(-2),
            expert_mu.unsqueeze(-3),
            expert_logvar.unsqueeze(-3),
        )
        log_q_mix = torch.logsumexp(
            mixture_weights.log().unsqueeze(-2) + component_log_probs,
            dim=-1,
        )
        log_prior = self._gaussian_latent_log_prob(
            z_samples,
            prior_mu.unsqueeze(-2),
            prior_logvar.unsqueeze(-2),
        )
        kl_estimate = log_q_mix - log_prior

        weighted_recon_per_sample = (mixture_weights * recon_nll).sum(dim=-1)
        weighted_kl_per_sample = (mixture_weights * kl_estimate).sum(dim=-1)
        weighted_recon = weighted_recon_per_sample.mean()
        weighted_kl = weighted_kl_per_sample.mean()
        total_loss = weighted_recon + self.beta * weighted_kl
        if return_per_sample:
            total_per_sample = weighted_recon_per_sample + self.beta * weighted_kl_per_sample
            return (
                total_loss,
                weighted_recon,
                weighted_kl,
                total_per_sample,
                weighted_recon_per_sample,
                weighted_kl_per_sample,
            )
        return total_loss, weighted_recon, weighted_kl

    def forward_t(self, x_mech_t, x_vis_t, a_m_1, h_m_1):

        post_mu, post_logvar, prior_mu, prior_logvar, h_t, modality_posteriors = (
            self.encode_observation_t(x_mech_t, x_vis_t, a_m_1, h_m_1)
        )

        # reparametrize
        s_t = self.reparameterize(post_mu, post_logvar)

        # decode 
        x_mech_t_hat, x_mech_t_logvar, x_vis_t_hat, x_vis_t_logvar = self.decode_t(s_t, a_m_1)

        # compute loss
        if (not self.moment_matching) and self.posterior_fusion in self.MOE_POSTERIOR_FUSIONS:
            (
                total_loss,
                recon_loss,
                kl_div,
                total_per_sample,
                recon_per_sample,
                kl_per_sample,
            ) = self.compute_moe_mixture_loss(
                x_mech_t,
                x_vis_t,
                a_m_1,
                prior_mu,
                prior_logvar,
                modality_posteriors,
                return_per_sample=True,
            )
        else:
            (
                total_loss,
                recon_loss,
                kl_div,
                total_per_sample,
                recon_per_sample,
                kl_per_sample,
            ) = self.compute_loss(
                x_mech_t, x_vis_t, x_mech_t_hat, x_mech_t_logvar, x_vis_t_hat, x_vis_t_logvar,
                post_mu, post_logvar, prior_mu, prior_logvar, return_per_sample=True
            )

        responses_t = {
            "x_mech_hat": x_mech_t_hat,
            "x_mech_logvar": x_mech_t_logvar,
            "x_vis_hat": x_vis_t_hat,
            "x_vis_logvar": x_vis_t_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "s": s_t,
            "h": h_t
        }
        responses_t.update(modality_posteriors)

        losses_t = {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_div,
            "total_per_sample": total_per_sample,
            "recon_per_sample": recon_per_sample,
            "kl_per_sample": kl_per_sample,
        }

        return responses_t, losses_t
    
    def forward(self, X_mech, X_vis, A, h0=None):

        batch_size, seq_len = X_mech.shape[:2]

        if h0 is None:
            h0 = self.initial_hidden(batch_size, device=X_mech.device, dtype=X_mech.dtype)

        seq_responses = []
        mean_total, mean_mse, mean_kl = 0., 0., 0.

        for t in range(seq_len):

            x_mech_t = X_mech[:, t].reshape(-1, self.mech_obs_dim)
            x_vis_t = X_vis[:, t].reshape(-1, *self.vis_obs_shape)
            a_m_1 = A[:, t].reshape(-1, self.action_dim)
            if t == 0:
                h_m_1 = h0
            elif self.rnn_type == "lstm":
                h_m_1 = (seq_responses[t-1]['h'], seq_responses[t-1]['c'])
            else:
                h_m_1 = seq_responses[t-1]['h']
            
            responses_t, losses_t = self.forward_t(x_mech_t, x_vis_t, a_m_1, h_m_1)

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

class NectarySearchModel(nn.Module):

    def __init__(self, obs_dim=1, action_dim=1, latent_dim=1, hidden_dim=64, beta=0.1):

        super(NectarySearchModel, self).__init__()

        self.beta = beta
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Initial prior
        self._p0_mu = nn.Parameter(
            torch.zeros((self.latent_dim,), requires_grad=True)
        )
        self._p0_logvar = nn.Parameter(
            torch.zeros((self.latent_dim,), requires_grad=True)
        )

        # Initial posterior
        self.q0_mu = nn.Linear(self.obs_dim, self.latent_dim)
        self.q0_logvar = nn.Linear(self.obs_dim, self.latent_dim)

        # Posterior network
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim + self.obs_dim, 64),
            nn.ReLU()
        )
        self.enc_mu = nn.Linear(64, latent_dim)
        self.enc_logvar = nn.Linear(64, latent_dim)

        # RNN
        self.s2h = nn.Linear(latent_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)

        # Prior network
        self.prior = nn.Sequential(
            nn.Linear(self.action_dim + self.hidden_dim, 64),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(64, self.latent_dim)
        self.prior_logvar = nn.Linear(64, self.latent_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.obs_dim)
        )

    def encode_t(self, x_t, a_m_1=None, h_m_1=None):
        if (a_m_1 is not None) and (h_m_1 is not None):
            inputs = torch.cat([x_t, a_m_1, h_m_1], dim=-1)
            enc_h = self.encoder(inputs)
            mu = self.enc_mu(enc_h)
            logvar = self.enc_logvar(enc_h)
        else:
            mu = self.q0_mu(x_t)
            logvar = self.q0_logvar(x_t)
        
        return mu, logvar
    
    def prior_t(self, h_m_1, a_m_1):
        inputs = torch.cat([h_m_1, a_m_1], dim=-1)
        pr_h = self.prior(inputs)
        pr_mu = self.prior_mu(pr_h)
        pr_logvar = self.prior_logvar(pr_h)
        return pr_mu, pr_logvar

    def decode_t(self, s_t):
        return self.decoder(s_t)
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, x_t, x_t_hat, post_mu, post_logvar, prior_mu, prior_logvar):
        
        ## Reconstruction loss (negative log-likelihood)
        # For the first observation, this is Bernoulli log likelihood
        binary_loglik = F.binary_cross_entropy_with_logits(
            x_t_hat[..., 0], x_t[..., 0], reduction='sum'
        )
        gauss_loglik = F.mse_loss(
            x_t_hat[..., 1:], x_t[..., 1:], reduction='sum'
        )
        recon_loss = binary_loglik + gauss_loglik

        # KL divergence between q(s_t | x_t, a_{t-1}) and p(s_t| h_{t-1}, a_{t-1})
        sigma_q, sigma_p = post_logvar.exp(), prior_logvar.exp()
        kl_divergence = 0.5 * (torch.log(sigma_p / sigma_q) + ((sigma_q + (post_mu - prior_mu) ** 2) / sigma_p) - 1)
        kl_divergence = kl_divergence.sum()

        # Total loss
        total_loss = recon_loss + self.beta * kl_divergence

        return total_loss, recon_loss, kl_divergence

    
    def forward_t(self, x_t, a_m_1, h_m_1, t=1):

        # get prior
        if t > 0:
            prior_mu, prior_logvar = self.prior_t(h_m_1, a_m_1)
        else:
            prior_mu = self._p0_mu.expand(x_t.shape[0], -1)
            prior_logvar = self._p0_logvar.expand(x_t.shape[0], -1)

        # encode posterior
        if t > 0:
            post_mu, post_logvar = self.encode_t(x_t, a_m_1, h_m_1)
        else:
            post_mu, post_logvar = self.encode_t(x_t)

        # reparametrize
        s_t = self.reparameterize(post_mu, post_logvar)

        # decode 
        x_t_hat = self.decode_t(s_t)

        # update the recurrent hidden state
        h_t = torch.tanh(self.s2h(s_t) + self.h2h(h_m_1))

        # compute loss
        total_loss, recon_loss, kl_div = self.compute_loss(
            x_t, x_t_hat, post_mu, post_logvar, prior_mu, prior_logvar
        )

        responses_t = {
            "x_hat": x_t_hat,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "s": s_t,
            "h": h_t
        }

        losses_t = {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_div
        }

        return responses_t, losses_t

    def forward(self, X, A, h0=None):

        batch_size, seq_len = X.shape[:2]

        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_dim, device=X.device, dtype=X.dtype)

        seq_responses = []
        mean_total, mean_mse, mean_kl = 0., 0., 0.

        for t in range(seq_len):

            x_t = X[:, t].reshape(-1, self.obs_dim)
            a_m_1 = None if t == 0 else A[:, t-1].reshape(-1, self.action_dim)
            h_m_1 = h0 if t == 0 else seq_responses[t-1]['h']
            
            responses_t, losses_t = self.forward_t(x_t, a_m_1, h_m_1, t)

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

        

    
