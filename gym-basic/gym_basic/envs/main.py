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

#ANNOTATION-RELATED VARIABLES
root = "./video_annotations/train/*"
videos = glob.glob(root)
random.shuffle(videos)
total_videos = len(videos)
video_idx = 0 #Index of current video
action_idx = 0 #Index of next_action
frame = 0 #Current frame

annotations = np.load(videos[video_idx], allow_pickle=True)


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
		global frame, action_idx, annotations
		
		"""
		print("\nann: ", annotations)
		print("F: ", frame)
		print("Previous end frame. ", annotations['frame_end'][action_idx-1])
		
		print("ACTION: ", action)
		
		print("Execution frame: ", self.perform_action_time(action))
		print("idx: ", action_idx)
		print("ANNOTATION RIGHT NOW: ", annotations['label'][action_idx])
		"""
		
		assert self.action_space.contains(action)
		
		reward = 0
		
		done = False
		optim = False
		
		current_state = self.state #Current state
		
		if frame % 50 == 0:
			#print("\nWe do an action here")
			reward = self._take_action(action)
			optim = True

		
		self.transition() #Transition to a new state
		next_state = self.state


		if undo_one_hot(self.state) == N_ATOMIC_ACTIONS-1: #If the next action is nothing (==terminal state), finish episode
			#print("TERMINAL STATE.")
			done = True
		
		#PRINT STATE-ACTION TRANSITION & REWARD
		if self.display: self.render(current_state, next_state, action, reward, self.total_reward)
		
		
		return next_state, reward, done, optim, frame		
		
		
	def get_total_reward(self):
		return self.total_reward
	
	
	def perform_action_time(self, action):

		global annotations, action_idx, frame
		
		length = len(annotations['label'])
		last_frame = int(annotations['frame_end'][length-1])
		
		exe_frame = cfg.ROBOT_ACTION_DURATIONS[action] + frame	
		
		return min(last_frame, exe_frame)
	
	def reset(self):
		"""
		Resets the environment to an initial state.
		"""
		super().reset()
		
		global video_idx, action_idx, annotations, frame
		
		annotations = np.load(videos[video_idx], allow_pickle=True)
		
		if video_idx+1 < total_videos:
			video_idx += 1
		else:
			video_idx = 0
			print("EPOCH COMPLETED.")
		
		action_idx = 1   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! START AT 0 OR 1? 
		frame = 0
		self.total_reward = 0	
		
		#First Next_action and Active Object	
		na = one_hot(annotations['label'][action_idx], N_ATOMIC_ACTIONS)
		ao = np.zeros((N_OBJECTS))
		
		ao_idx = annotations['object_label'][action_idx]	
		if type(ao_idx) == int:
			pass
		else:
			for idx in ao_idx:
				ao = 0.6 * ao
				ao[idx] = 1
		
		if VERSION == 1:
			self.state = na
		elif VERSION == 2:
			self.state = concat_vectors(na, ao)

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
			state = undo_one_hot(na) #Only consider the Next Action as the state.	
		
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

	
	def transition(self):
		"""
		Gets a new observation of the environment based on the current frame and updates the state.
		
		Global variables:
			frame: current time step.
			action_idx: index of the NEXT ACTION (state as the predicted action). *The action_idx points towards the next atomic action at the current frame.
			annotations: pickle with the annotations, in the form of a table. 
		
		"""
		
		global action_idx, frame, annotations
		
		frame += 1 #Update frame
		length = len(annotations['label']) - 1 #Length from 0 to L-1 (same as action_idx). "Duration of the recipe, in number of actions"
		
		#print("Frmae. ", frame, end='\r')
		
		
		# 1)
		#GET TIME STEP () (Updates the action_idx)
		#We transition to a new action index when we surpass the init frame of an action (so that we point towards the next one).	
		if frame > annotations['frame_init'][action_idx]:
			action_idx += 1
		
		
		
		# 2) GET NA & AO FROM ANNOTATIONS		
		# Check if the action_idx is pointing towards nothing == the person is performing the last action of the recipe and there is no NEXT ACTION.
		if action_idx >= length+1: #If we finish, code TERMINAL STATE
			na = one_hot(-1, N_ATOMIC_ACTIONS) #Code a TERMINAL STATE as [0, 0, ..., 1]
			ao = np.zeros((N_OBJECTS)) #Code Active Object as zeros.
		
		# If we haven't finished the recipe, then we update the STATE by getting the NEXT ACTION and the ACTIVE OBJECT.
		else:
			#Generate a random number between 0 and 1. 
			p = random.uniform(0, 1)
			
			# 5% chance of erroneously coding another action that does not correspond to the annotations.
			if p>0.95:
				na = random.randint(0, N_ATOMIC_ACTIONS-2) #Random label with 5% prob (from action 0 to N_AA-2, to avoid coding the TERMINAL STATE)
			
			# 95 % chance of coding the proper action
			else:
				na = annotations['label'][action_idx] #Correct label
			
			na = one_hot(na, N_ATOMIC_ACTIONS) #From int to one-hot vector.
			
			# Generate gaussian noise with 0 mean and 1 variance.
			noise = np.random.normal(0, 1, N_ATOMIC_ACTIONS)
			na_noisy = na + 0.1*noise #Add the noise to the NEXT ACTION vector. The multiplication factor modulates the amplitude of the noise. Right now, 0.1 is not very aggresive but it is noticeable.
			na_norm = (na_noisy + abs(np.min(na_noisy))) / (np.max(na_noisy) - np.min(na_noisy)) #Normalize so that the vector represents the probability of each action. 
			na = na_norm / np.sum(na_norm)
			
			# This is an invented variable that codes the active objects of the NEXT ACTION.
			ao = np.zeros((N_OBJECTS))
			ao_idx = annotations['object_label'][action_idx] #Indices of the objects, for example [1, 4, 5] 
			if type(ao_idx) == int:
				pass
			else:
				for idx in ao_idx: # This generates different activation values, so that it is not coded with 0s and 1s.
					ao = 0.6 * ao
					ao[idx]=1 # In the example with [1, 4, 5], the AO would look like [0, 0.36, 0, 0, 0.6, 1, 0, ..., 0]
	
		#Either take the NEXT ACTION as the STATE, or also concatenate the ACTIVE OBJECT
		if VERSION == 1:
			state = na
		elif VERSION == 2:
			state = concat_vectors(na, ao)	
		
		
		self.state = state

	
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
		
		#Numerical version
		#print('| State: {:>3g}'.format(state), ' | Action: {:>3g}'.format(action), ' | New state: {:>3g}'.format(next_state), ' | Reward {:>3g}'.format(reward), ' | Total reward {:>3g}'.format(total_reward), ' |')
		#print('='*82)
		
		#Version with dictionary meanings (provided that the state/action spaces are shorter than or equal to the dictionary.
		print('| STATE: {0:>29s}'.format(self.next_atomic_action_repertoire[state]), ' | ACTION: {0:>20s}'.format(self.action_repertoire[action]), ' | NEW STATE: {0:>29s}'.format(self.next_atomic_action_repertoire[next_state]), ' | REWARD {:>3g}'.format(reward), ' | TOTAL REWARD {:>3g}'.format(total_reward), ' |')
		#print('='*151)


		
		
	def close(self):
		pass
