from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch

class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
	
	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, observation):
		if isinstance(observation, dict):
			for k in observation.keys():
				observation[k] = self._try_f32_tensor(observation[k])
		else:
			observation = self._try_f32_tensor(observation)
		return observation

	def reset(self):
		return self._obs_to_tensor(self.env.reset())

	def step(self, action):
		observation, reward, done, info = self.env.step(action.numpy())
		info = defaultdict(float, info)
		info["success"] = float(info["success"])
		info["terminated"] = torch.tensor(float(info["terminated"]))
		return self._obs_to_tensor(observation), torch.tensor(reward, dtype=torch.float32), done, info

	def render(self, height=None, width=None, camera_id=0, **kwargs):
		return self.env.render(height, width, camera_id, **kwargs)