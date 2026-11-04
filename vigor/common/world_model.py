import copy

from tensordict import TensorDict
import torch
from tensordict.nn import TensorDictParams
import torch.nn as nn
import torch.nn.functional as F

from vigor.common import layers, math, init, augmentation

class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for single-task experiments only.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
        self._termination = layers.mlp(cfg.latent_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
        self._pi = layers.mlp(cfg.latent_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
        self.init()

        self.aug = augmentation.Augmentation
        self.aug.dataset_dir = cfg.dataset_dir
        self.aug.num_workers = cfg.places_num_workers
        self.aug.num_bg = cfg.places_num_bg
        self.aug.bank_seed = cfg.places_seed

    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        # Create modules
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = copy.deepcopy(self._Qs)
            self._target_Qs = copy.deepcopy(self._Qs)

        # Assign params to modules
        # We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        repr = "TD-MPC2 World Model\n"
        modules = ["Encoder", "Dynamics", "Reward", "Termination", "Policy prior", "Q-functions"]
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
            if m == self._termination and not self.cfg.episodic:
                continue
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def _batched(self, fn, x):
        """
        Applies `fn` to a (B, C, H, W) or (T, B, C, H, W) batch in a single call.
        Time steps are flattened into the batch, so every sample is processed independently.
        """
        if x.ndim == 5:
            return fn(x.flatten(0, 1)).unflatten(0, x.shape[:2])
        elif x.ndim == 4:
            return fn(x)
        raise ValueError("Given observation has a wrong dimension.")

    def weak_augment(self, observation):
        """Random shift with an independent shift per sample (and per time step)."""
        return self._batched(lambda x: self.aug.random_shift(x, pad_size=4), observation)

    def strong_augment(self, weak_aug_obs):
        """Random overlay on top of weak augmentation, with an independent image per sample (and per time step)."""
        return self._batched(lambda x: self.aug.random_overlay(x, device=x.device), weak_aug_obs)

    def encode(self, observation):
        """
        Encodes a (weakly augmented) observation into its latent representation.
        Returns the latent state and the weakly augmented observation.
        """
        weak_aug_obs = self.weak_augment(observation)
        return self._batched(self._encoder, weak_aug_obs), weak_aug_obs

    def encode_mixed(self, weak_aug_obs, strong_aug_obs):
        """
        Encodes a batch where a random subset of B//(1+zeta) samples is strongly augmented
        and the remaining samples are weakly augmented.
        """
        B, zeta = weak_aug_obs.shape[0], self.cfg.zeta
        assert zeta >= 0.0, "Zeta should be greater than or equal to zero"
        strong_aug_size = int(B//(1 + zeta))
        _, strong_aug_idx = torch.split(torch.randperm(B, device=weak_aug_obs.device), [B - strong_aug_size, strong_aug_size])
        mixed_aug_obs = weak_aug_obs.clone()
        mixed_aug_obs[strong_aug_idx] = strong_aug_obs[strong_aug_idx]
        return self._encoder(mixed_aug_obs)

    def compute_reg_loss(self, weak_aug_z, strong_aug_obs):
        """
        Constrain the latent model to predict consistent representations regardless of given perturbations.

        `weak_aug_z`: (T, B, latent_dim) latents of the weakly augmented observations, used as fixed targets.
        `strong_aug_obs`: (T, B, C, H, W) strongly augmented counterparts of the same observations.
        Returns the sum over time steps of the per-step mean squared error.
        """
        strong_aug_z = self._batched(self._encoder, strong_aug_obs)
        weak_aug_z = F.normalize(weak_aug_z.detach(), p=2, dim=-1)
        strong_aug_z = F.normalize(strong_aug_z, p=2, dim=-1)
        return F.mse_loss(strong_aug_z, weak_aug_z, reduction="none").mean(dim=(1, 2)).sum()

    def next(self, z, a):
        """
        Predicts the next latent state given the current latent state and action.
        """
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a):
        """
        Predicts instantaneous (single-step) reward.
        """
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def termination(self, z, unnormalized=False):
        """
        Predicts termination signal.
        """
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))

    def pi(self, z):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        log_prob = math.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1]
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict({
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        })
        return action, info

    def Q(self, z, a, return_type="min", target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == "all":
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2