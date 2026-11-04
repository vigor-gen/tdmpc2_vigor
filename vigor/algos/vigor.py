from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from vigor.common import math
from vigor.algos.tdmpc2 import TDMPC2

class VIGOR(TDMPC2):
	"""VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning"""
	def _update(self, observation, action, reward, terminated):
		# Compute targets
		# Latents of the weakly augmented observations serve both as consistency targets (t>=1)
		# and as regularization targets (all t), so they are encoded once for the whole horizon.
		with torch.no_grad():
			weak_aug_z, weak_aug_obs = self.model.encode(observation)
			next_z = weak_aug_z[1:]
			td_targets = self._td_target(next_z, reward, terminated)
		strong_aug_obs = self.model.strong_augment(weak_aug_obs)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode_mixed(weak_aug_obs[0], strong_aug_obs[0])
		zs[0] = z
		# Compute regularization loss for weak and strong augmentation over the whole horizon
		regularization_loss = self.model.compute_reg_loss(weak_aug_z, strong_aug_obs)
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, return_type="all")
		reward_preds = self.model.reward(_zs, action)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		regularization_loss = regularization_loss / self.cfg.horizon
		total_loss = (
			(self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss) +
			self.cfg.alpha * regularization_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach())

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"regularization_loss": regularization_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()