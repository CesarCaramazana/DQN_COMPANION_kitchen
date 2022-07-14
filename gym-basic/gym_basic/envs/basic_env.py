import gym
from gym import error, spaces, utils
from gym.utils import seeding


class BasicEnv(gym.Env):
	message = "Custom environment"
	
	def __init__(self):
		pass
	
	def step(self, action):
		pass
		
	def reset(self):
		pass
		
	def render(self, mode='human'):
		pass
		
	def close(self):
		pass				
