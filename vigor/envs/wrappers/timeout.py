import gymnasium as gym

class Timeout(gym.Wrapper):
	"""
	Wrapper for enforcing a time limit on the environment.
	"""

	def __init__(self, env, max_episode_steps):
		super().__init__(env)
		self._max_episode_steps = max_episode_steps
	
	@property
	def max_episode_steps(self):
		return self._max_episode_steps

	def reset(self, **kwargs):
		self._timestep = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		self._timestep += 1
		done = done or self._timestep >= self.max_episode_steps
		return observation, reward, done, info

	def render(self, height=None, width=None, camera_id=0, **kwargs):
		return self.env.render(height, width, camera_id, **kwargs)