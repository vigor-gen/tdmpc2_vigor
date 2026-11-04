import time

import numpy as np
from tensordict.tensordict import TensorDict
import torch

from vigor.trainer.base import Trainer

class OnlineTrainer(Trainer):
	"""Trainer class for single-task online training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._episode_index = 0
		self._start_time = time.time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time.time() - self._start_time
		return dict(
			step=self._step,
			episode=self._episode_index,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate an agent."""
		episode_returns, episode_successes, episode_lengths = [], [], []
		for episode in range(self.cfg.eval_episodes):
			observation, done, episode_return, timestep = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(episode==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(observation, t0=timestep==0, eval_mode=True)
				observation, reward, done, info = self.env.step(action)
				episode_return += reward
				timestep += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			episode_returns.append(episode_return.item())
			episode_successes.append(info["success"])
			episode_lengths.append(timestep)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		eval_metric = dict(episode_return=episode_returns,
						   episode_success=episode_successes,
						   episode_index=[episode_index for episode_index in range(self.cfg.eval_episodes)],
						   episode_length= np.nanmean(episode_lengths),)
		return eval_metric

	def to_td(self, observation, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(observation, dict):
			observation = TensorDict(observation, batch_size=(), device="cpu")
		else:
			observation = observation.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float("nan"))
		if reward is None:
			reward = torch.tensor(float("nan"))
		if terminated is None:
			terminated = torch.tensor(float("nan"))
		td = TensorDict(
			observation=observation,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train an agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, "eval")
					eval_next = False

				if self._step > 0:
					if info["terminated"] and not self.cfg.episodic:
						raise ValueError("Termination detected but you are not in episodic mode. " \
						"Set `episodic=true` to enable support for terminations.")
					train_metrics.update(
						episode_return=torch.tensor([td["reward"] for td in self._tds[1:]]).sum(),
						episode_success=info["success"],
						episode_length=len(self._tds),
						episode_terminated=info["terminated"])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, "train")
					self._episode_index = self.buffer.add(torch.cat(self._tds))

				observation = self.env.reset()
				self._tds = [self.to_td(observation)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(observation, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			observation, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(observation, action, reward, info["terminated"]))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print("Pretraining agent on seed data...")
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)