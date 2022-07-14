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
		elif VERSION == 3:
			self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS+3*N_OBJECTS+N_OBJECTS)
		
		self.state = [] #One hot encoded state
		self.steps = 0 #Number of actions

		self.end = N_ATOMIC_ACTIONS-1 #Next Act
		
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
		"""
		assert self.action_space.contains(action)
		done = False
		
		current_state = self.state #Current state
			
		reward = self._take_action(action) #Take action and transition from current state to state'
		
		#reward = self._take_action2(action) #REWARD GUI

		next_state = self.state
		
		#if undo_one_hot(self.state) == self.end or self.steps == 0: #Finish episode when reached end state or run out of moves
		"""
		if self.steps == 0: #Finish if run out of moves	
			done=True
		"""
		#print("Transitioned to: ", undo_one_hot(next_state))
		if undo_one_hot(next_state) == N_ATOMIC_ACTIONS-1: #If the next action is nothing (==terminal state), finish episode
			done = True

		
		#PRINT STATE-ACTION TRANSITION & REWARD
		if self.display: self.render(current_state, next_state, action, reward, self.total_reward)
		#self.render(undo_one_hot(current_state), undo_one_hot(next_state), action, reward, self.total_reward)
		
		#self.render(undo_one_hot(undo_concat_na(current_state, N_ATOMIC_ACTIONS)), undo_one_hot(undo_concat_na(next_state, N_ATOMIC_ACTIONS)), action, reward, self.total_reward) #NA + Ob
		

		
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
		self.steps = N_ATOMIC_ACTIONS
		
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
		
		if state == 0: #'pour milk'
			if action == 8: #'bring milk'
				reward = 1
			else: reward = -1	
		
		elif state == 1: #'pour water'
			if action == 17: #'bring water'
				reward = 1
			else: reward = -1
		
		elif state == 2: #'pour coffee'
			if action == 18: #'do nothing' -> *coffee is at arm's reach
				reward = 1
			else: reward = -1					
		
		elif state == 3: #'pour Nesquik'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 4: #'pour sugar'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 5: #'put microwave'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 6: #'stir spoon
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 7: #'extract milk fridge'
			if action == 8:
				reward = 1
			else: reward = -1					
		
		elif state == 8: #'extract water fridge'
			if action == 17:
				reward = 1
			else: reward = -1
		
		elif state == 9: #'extract sliced bread'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 10: #'put toaster'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 11: #'extract butter fridge'
			if action == 1: #'bring butter'
				reward = 1
			else: reward = -1
		
		elif state == 12: #'extract jam fridge'
			if action == 6: #'bring jam'
				reward = 1
			else: reward = -1					
		
		elif state == 13: #'extract tomato sauce fridge'
			if action == 16: #'bring tomato sauce'
				reward = 1
			else: reward = -1
		
		elif state == 14: #'extract nutella fridge'
			if action == 10: #'bring nutella'
				reward = 1
			else: reward = -1			
		
		elif state == 15: #'spread butter'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 16: #'spread jam'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 17: #'spread tomato sauce'
			if action == 18:
				reward = 1
			else: reward = -1					
		
		elif state == 18: #'spread nutella'
			if action == 18:
				reward = 1 
			else: reward = -1
		
		elif state == 19: #'pour olive oil'
			if action == 18:
				reward = 1
			else: reward = -1
			
		elif state == 20: #'put jam fridge'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 21: #'put butter fridge'
			if action == 20:
				reward = 1
			else: reward = -1
		
		elif state == 22: #'put tomato sauce fridge'
			if action == 21:
				reward = 1
			else: reward = -1					
		
		elif state == 23: #'put nutella fridge'
			if action == 22:
				reward = 1
			else: reward = -1
		
		elif state == 24: #'pour milk bowl'
			if action == 18:
				reward = 1
			else: reward = -1
			
		elif state == 25: #'pour cereals bowl'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		elif state == 26: #'pour nesquik bowl'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 27: #'put bowl microwave'
			if action == 18:
				reward = 1
			else: reward = -1					
		
		elif state == 28: #'stir spoon bowl'
			if action == 18:
				reward = 1
			else: reward = -1
		
		elif state == 29: #'put milk fridge'
			if action == 23:
				reward = 1
			else: reward = -1
			
		elif state == 30: #'put sliced bread plate'
			if action == 18:
				reward = 1
			else: reward = -1	
		
		else:
			if action == 18: #'do nothing'
				reward = 1
			else: reward = -1					
		#------------------------------		
				
		self.total_reward += reward
		self.steps += -1

			
		self.transition() #Get new observation -> state'
				
	
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
		
		reward = get_reward_GUI() #REWARD VIA GRAPHICAL INTERFACE
		#reward = get_sentiment_keyboard() #REWARD VIA TEXT (SENTIMENT ANALYSIS OF A SENTENCE)
		
		self.total_reward += reward
		self.steps += -1

			
		#time.sleep(1) #Frequency of observations
		self.transition()	
			
		
		return reward
	
	
	def transition(self):
		"""
		Gets a new observation of the environment and updates the state.
		"""
	
		self.state = get_state(version=VERSION)

	
	def render(self, state, next_state, action, reward, total_reward):
		"""
		Prints the environment.		
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
