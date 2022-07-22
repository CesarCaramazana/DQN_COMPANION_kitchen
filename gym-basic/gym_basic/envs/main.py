import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import time
from aux import *
import config as cfg

#CONFIGURATION GLOBAL ENVIRONMENT VARIABLES
ACTION_SPACE = cfg.ACTION_SPACE
N_ATOMIC_ACTIONS = cfg.N_ATOMIC_ACTIONS
N_OBJECTS = cfg.N_OBJECTS

ATOMIC_ACTIONS_MEANINGS = cfg.ATOMIC_ACTIONS_MEANINGS
OBJECTS_MEANINGS = cfg.OBJECTS_MEANINGS
ROBOT_ACTIONS_MEANINGS = cfg.ROBOT_ACTIONS_MEANINGS

VERSION = cfg.VERSION


class BasicEnv(gym.Env):
	message = "Custom environment for recipe preparation scenario."
	
	def __init__(self, display=False):
		self.action_space = gym.spaces.Discrete(ACTION_SPACE) #[0, ACTION_SPACE-1]
		
		if VERSION == 1:
			self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS) #State as Next Action
		elif VERSION == 2:
			self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS+N_OBJECTS) #State as Next Action + VWM	 

		
		self.state = [] #One hot encoded state
		
		self.total_reward = 0
		
		self.action_repertoire = ROBOT_ACTIONS_MEANINGS
		self.next_atomic_action_repertoire = ATOMIC_ACTIONS_MEANINGS
		
		self.display = display
		
	
	def get_action_meanings(self):
		return self.action_repertoire
	def get_state_meanings(self):
		return self.state_repertoire
			
	def step(self, action):
		"""
		Transition from the current state (self.state) to the next one given an action.
		
		Input:
			action: (int) action taken by the agent.
		Output:
			next_state: (numpy array) state transitioned to after taking action.
			reward: (int) reward received. 
			done: (bool) True if the episode is finished (the recipe has reached its end).
			info:	
		"""
		assert self.action_space.contains(action)
		done = False
		
		current_state = self.state #Current state
			
		reward = self._take_action(action) #Take action	
		#reward = self._take_action2(action) #REWARD GUI
		
		self.transition() #Transition to a new state
		next_state = self.state

		if undo_one_hot(self.state) == N_ATOMIC_ACTIONS-1: #If the next action is nothing (==terminal state), finish episode
			done = True
		
		#PRINT STATE-ACTION TRANSITION & REWARD
		if self.display: self.render(current_state, next_state, action, reward, self.total_reward)

		
		info = {}
		return next_state, reward, done, info		
		
		
	def get_total_reward(self):
		return self.total_reward
	
	def reset(self):
		"""
		Resets the environment to an initial state.
		"""
		super().reset()
		
		self.state = get_init_state(version=VERSION)

			
		self.total_reward = 0

		
		return self.state


	def _take_action(self, action): 
		"""
		Version of the take action function that considers a unique correct robot action for each state, related to the required object and its position (fridge or table). 
		
		
		Input:
			action: (int) from the action repertoire taken by the agent.
		Output:
			reward: (int) received from the environment.
		
		"""

		if VERSION == 1:
			state = undo_one_hot(self.state) #If the state is the Next Action vector, undo the O-H to obtain the integer value.
		elif VERSION == 2: #If the state is NA + VWM, first separate the two variables and then obtain the value of the state from the Next Action.
			na, ao = undo_concat_state(self.state)
			state = undo_one_hot(na) 	
		
		reward = 0
		
		if state == 1: #'pour milk'
			if action == 8: #'bring milk'
				reward = 1
			else: reward = -1	
		
		elif state == 2: #'pour water'
			if action == 17: #'bring water'
				reward = 1
			else: reward = -1
		
		elif state == 3: #'pour coffee'
			if action == 18: #'do nothing' -> *coffee is at arm's reach
				reward = 1
			else: reward = -1					
		
		elif state == 4: #'pour Nesquik'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 5: #'pour sugar'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 6: #'put microwave'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 7: #'stir spoon
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 8: #'extract milk fridge'
			if action == 8:
				reward = 1
			else: reward = -1					
		
		elif state == 9: #'extract water fridge'
			if action == 17:
				reward = 1
			else: reward = -1
		
		elif state == 10: #'extract sliced bread'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 11: #'put toaster'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 12: #'extract butter fridge'
			if action == 1: #'bring butter'
				reward = 1
			else: reward = -1
		
		elif state == 13: #'extract jam fridge'
			if action == 6: #'bring jam'
				reward = 1
			else: reward = -1					
		
		elif state == 14: #'extract tomato sauce fridge'
			if action == 16: #'bring tomato sauce'
				reward = 1
			else: reward = -1
		
		elif state == 15: #'extract nutella fridge'
			if action == 10: #'bring nutella'
				reward = 1
			else: reward = -1			
		
		elif state == 16: #'spread butter'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 17: #'spread jam'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 18: #'spread tomato sauce'
			if action == 18:
				reward = 1
			else: reward = -1					
		
		elif state == 19: #'spread nutella'
			if action == 18:
				reward = 1 
			else: reward = -1
		
		elif state == 20: #'pour olive oil'
			if action == 18:
				reward = 1
			else: reward = -1
			
		elif state == 21: #'put jam fridge'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 22: #'put butter fridge'
			if action == 20:
				reward = 1
			else: reward = -1
		
		elif state == 23: #'put tomato sauce fridge'
			if action == 21:
				reward = 1
			else: reward = -1					
		
		elif state == 24: #'put nutella fridge'
			if action == 22:
				reward = 1
			else: reward = -1
		
		elif state == 25: #'pour milk bowl'
			if action == 18:
				reward = 1
			else: reward = -1
			
		elif state == 26: #'pour cereals bowl'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 27: #'pour nesquik bowl'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 28: #'put bowl microwave'
			if action == 18:
				reward = 1
			else: reward = -1					
		
		elif state == 29: #'stir spoon bowl'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 30: #'put milk fridge'
			if action == 23:
				reward = 1
			else: reward = -1
			
		elif state == 31: #'put sliced bread plate'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		else:
			if action == 18: #'do nothing'
				reward = 1
			else: reward = -1					
		#------------------------------		
				
		self.total_reward += reward
			
			
	
		return reward
	
	def _take_action2(self, action): 
		"""
		Version of the take action function that considers the user's input as the reward signal. The input can be provided via different interfaces.
		
		Input:
			action: (int) from the action repertoire taken by the agent.
		Output:
			reward: (int) received from the user input.
		
		"""

		if VERSION == 1:
			state = undo_one_hot(self.state) #If the state is the Next Action vector, undo the O-H to obtain the integer value.
		elif VERSION == 2: #If the state is NA + VWM, first separate the two variables and then obtain the value of the state from the Next Action.
			na, ao = undo_concat_state(self.state)
			state = undo_one_hot(na) 	
		elif VERSION == 3:
			na, vwm, ao = undo_concat_state_extended(self.state)
			state = undo_one_hot(na)
			
							
		print("| STATE: {0:>29s}".format(self.next_atomic_action_repertoire[state]), " | ACTION: {0:>20s}".format(self.action_repertoire[action]))
		
		#reward = get_reward_GUI() #REWARD VIA GRAPHICAL INTERFACE
		#reward = get_sentiment_keyboard() #REWARD VIA TEXT (SENTIMENT ANALYSIS OF A SENTENCE)
		reward = reward_confirmation_perform(action)	

		
		self.total_reward += reward
			
		
		return reward
	
	
	def transition(self):
		"""
		Gets a new observation of the environment and updates the state.
		"""
	
		self.state = get_state(version=VERSION)

	
	def render(self, state, next_state, action, reward, total_reward):
		"""
		Prints the environment.		
		Input:
			state: (numpy array) current state of the environment.
			next_state: (numpy array) state transitioned to after taking action.
			action: (int) action taken in current state.
			reward: (int) reward received in current state by taking action.
			total_reward: (int) cumulative reward of the episode.
		"""
		if VERSION == 1:
			state = undo_one_hot(state)
			next_state = undo_one_hot(next_state)		
		elif VERSION == 2:
			na, vwm = undo_concat_state(state)
			next_na, next_vwm = undo_concat_state(next_state)
			state = undo_one_hot(na)
			next_state = undo_one_hot(next_na)
		elif VERSION == 3:
			na, vwm, ao = undo_concat_state_extended(state)
			next_na, next_vwm, next_ao = undo_concat_state_extended(next_state)
			state = undo_one_hot(na)
			next_state = undo_one_hot(next_na)
		
		#Numerical version
		#print('| State: {:>3g}'.format(state), ' | Action: {:>3g}'.format(action), ' | New state: {:>3g}'.format(next_state), ' | Reward {:>3g}'.format(reward), ' | Total reward {:>3g}'.format(total_reward), ' |')
		#print('='*82)
		
		#Version with dictionary meanings (provided that the state/action spaces are shorter than or equal to the dictionary.
		print('| STATE: {0:>29s}'.format(self.next_atomic_action_repertoire[state]), ' | ACTION: {0:>20s}'.format(self.action_repertoire[action]), ' | NEW STATE: {0:>29s}'.format(self.next_atomic_action_repertoire[next_state]), ' | REWARD {:>3g}'.format(reward), ' | TOTAL REWARD {:>3g}'.format(total_reward), ' |')
		#print('='*151)


		
		
	def close(self):
		pass
