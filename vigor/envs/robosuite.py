from collections import deque

import gymnasium as gym
import numpy as np
import robosuite
from robosuite.controllers import load_controller_config

from vigor.envs.wrappers.robovgb import RoboVGBWrapper
from vigor.envs.wrappers.timeout import Timeout

class RobosuiteWrapper(gym.Wrapper):
	def __init__(self, env, action_repeat, frame_stack):
		super().__init__(env)
		self._action_repeat = action_repeat
		self._frame_stack = frame_stack
		obs_shape = env.observation_spec()["agentview_image"].transpose(2,0,1).shape
		obs_shape = (obs_shape[0] * self._frame_stack, *obs_shape[1:])
		self._frames = deque([], maxlen=self._frame_stack)
		self.observation_space = gym.spaces.Box(
			low=np.full(obs_shape, 0, dtype=np.uint8),
			high=np.full(obs_shape, 255, dtype=np.uint8),
			dtype=np.uint8)
		self.action_space = gym.spaces.Box(
			low=env.action_spec[0],
			high=env.action_spec[1],
			dtype=np.float32) # env.action_spec().dtype)
		self.action_spec_dtype = np.float32 # env.action_spec().dtype

	def reset(self, **kwargs):
		observation = self.env.reset(**kwargs)
		for _ in range(self._frames.maxlen):
			self._frames.append(observation)
		return np.concatenate(self._frames)

	def step(self, action):
		cum_reward = 0
		action = action.astype(self.action_spec_dtype)
		for _ in range(self._action_repeat):
			observation, reward, done, info = self.env.step(action)
			cum_reward += reward
			self._frames.append(observation)
		return np.concatenate(self._frames), cum_reward, False, info
	
	def render(self, width=None, height=None, camera_id=None):
		return self.env.render(width, height, camera_id)

def make_env(cfg):
	"""
	Make robosuite environment.
	Adapted from https://github.com/gemcollector/RL-ViGen
	"""
	task = cfg.task
	if task not in robosuite.ALL_ENVIRONMENTS:
		raise ValueError("Unknown task:", task)
	horizon = 500
	# single arm tasks: Door, Lift, NutAssembly, NutAssemblyRound, NutAssemblySingle, NutAssemblySquare, PickPlace,
	#                   PickPlaceBread, PickPlaceCan, PickPlaceCereal, PickPlaceMilk, PickPlaceSingle, Stack, Wipe
	# two arms tasks: TwoArmHandover, TwoArmLift, TwoArmPegInHole
	# 
	# robots: Panda, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
	# For two arms tasks, this should be a list. And if {env_configuration} is bimanual, {robots} will be set to the default Baxter
	task_cfg = {"env_name": task, "robots": "Panda" if task not in ["TwoArmHandover", "TwoArmLift", "TwoArmPegInHole"] else ["Panda", "Panda"],
				"camera_names": "agentview",
				"camera_heights": cfg.image_size,
				"camera_widths": cfg.image_size,
				"camera_depths": False,
				"reward_shaping": True,
				"has_renderer": False,
				"render_gpu_device_id": 0 if cfg.device.split(":")[-1] == "cpu" else int(cfg.device.split(":")[-1]),
				"ignore_done": False,
				"control_freq": 20,
				"hard_reset": False,
				"horizon": horizon}
	task_cfg["controller_configs"] = load_controller_config(default_controller="OSC_POSE") # JOINT_VELOCITY, JOINT_TORQUE, JOINT_POSITION, OSC_POSITION, OSC_POSE, IK_POSE
	env = robosuite.make(**task_cfg,
						use_camera_obs=True,
						has_offscreen_renderer=True)
	env = RoboVGBWrapper(env, task, cfg.seed, cfg.image_size, cfg.get("gen_type", "train"), cfg.get("scene_id", 0))
	env = RobosuiteWrapper(env, cfg.action_repeat, cfg.frame_stack)
	env = Timeout(env, max_episode_steps=horizon//cfg.action_repeat)
	cfg.domain = "robosuite"
	return env