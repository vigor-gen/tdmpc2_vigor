from collections import defaultdict, deque

from dm_control import suite
from dm_control.suite.wrappers import action_scale
import gymnasium as gym
import numpy as np
import torch

from vigor.envs.wrappers.dmcvgb import DMCVGBWrapper
from vigor.envs.wrappers.timeout import Timeout

def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)


class DMControlWrapper:
	def __init__(self, env, action_repeat):
		self.env = env
		self._action_repeat = action_repeat
		self._camera_id = 0
		obs_shape = get_obs_shape(env)
		action_shape = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(obs_shape, -np.inf, dtype=np.float32),
			high=np.full(obs_shape, np.inf, dtype=np.float32),
			dtype=np.float32)
		self.action_space = gym.spaces.Box(
			low=np.full(action_shape, env.action_spec().minimum),
			high=np.full(action_shape, env.action_spec().maximum),
			dtype=np.float32)
		self.action_spec_dtype = np.float32

	@property
	def unwrapped(self):
		return self.env
	
	def _obs_to_array(self, observation):
		return torch.from_numpy(
			np.concatenate([v.flatten() for v in observation.values()], dtype=np.float32))
	
	def reset(self):
		return self._obs_to_array(self.env.reset().observation)

	def step(self, action):
		reward = 0
		action = np.nan_to_num(action.astype(self.action_spec_dtype))
		for _ in range(self._action_repeat):
			step = self.env.step(action)
			reward += step.reward
			if step.last(): break
		return self._obs_to_array(step.observation), reward, False, defaultdict(float)
	
	def render(self, height=None, width=None, camera_id=None):
		return self.env.physics.render(height=height, width=width, camera_id=camera_id or self._camera_id)
	
	def __getattr__(self, name):
	    return getattr(self.env, name)

class PixelWrapper(gym.Wrapper):
	def __init__(self, env, image_size, frame_stack):
		super().__init__(env)
		self._image_size = image_size
		self._frame_stack = frame_stack
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(self._frame_stack*3, self._image_size, self._image_size), dtype=np.uint8)
		self._frames = deque([], maxlen=self._frame_stack)

	def _get_obs(self, is_reset=False):
		frame = self.env.render(height=self._image_size, width=self._image_size).transpose(2, 0, 1)
		num_frames = self._frames.maxlen if is_reset else 1
		for _ in range(num_frames):
			self._frames.append(frame)
		return torch.from_numpy(np.concatenate(self._frames))

	def reset(self):
		self.env.reset()
		return self._get_obs(is_reset=True)

	def step(self, action):
		_, reward, done, info = self.env.step(action)
		info["success"] = 0.0 # Compatibility for Robosuite
		return self._get_obs(), reward, done, info

	def render(self, height=None, width=None, camera_id=0, **kwargs):
		return self.env.render(height, width, camera_id, **kwargs)

def make_env(cfg):
	"""
	Make DMControl environment.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	domain, task = cfg.task.replace("-", "_").split("_", 1)
	if (domain, task) not in suite.ALL_TASKS:
		raise ValueError("Unknown task:", task)
	env = suite.load(domain,
					 task,
					 task_kwargs={"random": cfg.seed},
					 visualize_reward=False)
	env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
	env = DMControlWrapper(env, cfg.action_repeat)
	env = Timeout(env, max_episode_steps=1000//cfg.action_repeat)
	if cfg.exp_type == "eval":
		env = DMCVGBWrapper(env, domain, cfg.seed, cfg.frame_stack, cfg.image_size, cfg.gen_type, cfg.gen_diff)
	else:
		env = PixelWrapper(env, cfg.image_size, cfg.frame_stack)
	cfg.domain = "dmc"
	return env