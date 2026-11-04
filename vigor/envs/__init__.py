import warnings

import gymnasium as gym

from vigor.envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f"Missing dependencies for task {task}; install dependencies to use this environment.")

try:
	from vigor.envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies
try:
	from vigor.envs.robosuite import make_env as make_robosuite_env
except:
	make_robosuite_env = missing_dependencies

warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_env(cfg):
	"""
	Make an environment for VIGOR experiments.
	"""
	gym.logger.set_level(40)

	env = None
	for fn in [make_dm_control_env, make_robosuite_env]:
		try:
			env = fn(cfg)
		except ValueError:
			pass
	if env is None:
		raise ValueError(f"Failed to make environment '{cfg.task}': please verify that dependencies are installed and that the task exists.")
	env = TensorWrapper(env)
	cfg.obs_shape = env.observation_space.shape
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env