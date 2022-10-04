import numpy as np
import torch
import random
import PySimpleGUI as sg #Graphic Interface

import glob
import time
import os


import config as cfg


N_ATOMIC_ACTIONS = cfg.N_ATOMIC_ACTIONS
N_OBJECTS = cfg.N_OBJECTS

"""
In this script, some auxiliary functions that are used in the environment setup (./gym-basic/gym_basic/envs/main.py) are implemented.
There are three types of functions:
	1) General purpose: regarding the management of array variables.
	2) Get state: as interface functions between the input systems and the environment. Right now using video_annotations. In the future, these functions will be used to retrieve the outputs of the Action Prediction system (among others).
	3) Reward: user interfaces to get the reward value. 

"""


# 1) GENERAL PURPOSE FUNCTIONS
#Management of vectors: one hot encoding, softmax, concatenations and deconcatenations.
#---------------------------------------------------------------------------

def one_hot(x, n):
	"""
	One hot encoding of an integer variable x with n number of possible values.
	Input:
		x: (int) integer value.	
		n: (int) max number of states.
	Output:
		vector: (numpy array) one-hot encoded vector with all 0s and 1 in position x. 	
	"""
	
	vector = np.zeros((n))
	vector[x] = 1	
	
	return vector

def undo_one_hot(vector):
	"""
	Reverts a one-hot encoded vector and returns an integer as the argmax of the vector.
	Input:
		vector: (numpy array) one-hot encoded vector with all zeros except for a 1 at any position.
	Output:
		x: (int) argmax of the vector.	
	
	"""
	x = np.argmax(vector)

	return int(x)
	
def softmax(x):
	"""
	Performs the softmax operation in a vector.
	Input:
		x: (numpy array) vector or matrix.
	Output:
		Softmaxed vector/matrix.	
	"""
	e_x = np.exp(x - np.max(x))
	
	return e_x / e_x.sum(axis=0)	



def concat_vectors(a, b):
	"""
	Concatenates two vectors along axis 0.
	
	Input:
		a: vector (numpy array) of dimension N_a.
		b: vector (numpy array) of dimension N_b.
	Output:
		v: vector (numpy array) of dimension (N_a + N_b).	
	"""
	v = np.concatenate((a, b), axis=0)
	
	return v



#State: NA + AO
def undo_concat_state(state):
	"""
	Separates a full state vector into the Next Action vector and the Active Objects vector.
	Input:
		State: (numpy array) representation of the full state as a Next Action vector and the Active Object vector.
	Output:
		next_action: (numpy array) vector of dimensions N_ATOMIC_ACTIONS.
		ao: (numpy array) vector of dimensions N_OBJECTS.
	"""
	
	next_action = state[0:N_ATOMIC_ACTIONS]
	ao = state[N_ATOMIC_ACTIONS:]		
	
	assert next_action.shape[0] == N_ATOMIC_ACTIONS, f"Next action vector has expected number of elements"
	
	return next_action, ao


	
	
# 2) GET STATE 
#Get the state of the environment from video annotations.
#---------------------------------------------------------------------------

root = "./video_annotations/*"
videos = glob.glob(root)

random.shuffle(videos)

total_videos = len(videos)
video_idx = 0
action_idx = 0 
frame = 0 #Current frame
freq_obs = 30
t_na_i = 0 #Difference to frame init

annotations = np.load(videos[video_idx], allow_pickle=True)
print()
print(annotations)


def get_init_state(version=1):	
	"""
	Initial state as next action one-hot encoded vector.
	Input:
		version (int): version of the state definition. If version is 1, the state is the next atomic action. If version is 2, it is also the active object.
	Output:
		state (numpy vector)
	"""
	global video_idx, action_idx, annotations, frame

	pck = np.load(videos[video_idx], allow_pickle=True)
	
	#print("We are at: ", video_idx)
	
	if video_idx+1 < total_videos:
		video_idx += 1
	else: 
		print("Ya hemos usado todos los vídeos.")
		video_idx = 0
		
	action_idx = 0
	frame = 0
	
	annotations = pck
	
	na = get_next_action_annotations(action_idx, annotations)
	ao = get_active_object_annotations(action_idx, annotations)

	if version == 1:
		state = na
	elif version == 2:
		state = concat_vectors(na, ao)
	
	return state

	
def get_next_action_annotations(action_idx, annotations):
	"""
	Returns the atomic action given the index of the annotations.
	Input:
		action_idx: (int) index of the action, understood as the time-step of the video.
		annotations: (numpy/pickle) dictionary of the annotations.
		
	Output:
		na: (numpy vector) one-hot encoded next-action vector.
	
	"""
	
	length = len(annotations['label'])
	
	p = random.uniform(0, 1)
	
	#Erroneous label 5% of the time
	if p<0.95:
		na = annotations['label'][action_idx] #Correct label
	else:
		na = random.randint(0, N_ATOMIC_ACTIONS-1) #Random erroneous label	
	
	if na == 99:
		na = one_hot(31, N_ATOMIC_ACTIONS)
	else:
		na = one_hot(na, N_ATOMIC_ACTIONS)
	
	
	noise = np.random.normal(0, 1, N_ATOMIC_ACTIONS)
	na_noise = na + 0.15*noise	
	
	na_norm = (na_noise + abs(np.min(na_noise))) / (np.max(na_noise) - np.min(na_noise))
	na_norm = na_norm / np.sum(na_norm)
	
	"""
	plt.figure(figsize=(12, 5))
	plt.subplot(131)
	plt.title("Original label")
	plt.plot(na)
	plt.subplot(132)
	plt.title("Noise")
	plt.plot(noise, 'x')
	plt.subplot(133)
	plt.title("Probability")
	plt.plot(na_norm)
	plt.show()
	"""

	return na



def get_active_object_annotations(action_idx, annotations):
	"""
	Gets the active objects of an atomic action given the index of the action.
	Input:
		action_idx: (int) index of the action, understood as the time-step of the video.
		annotations: (numpy/pickle) dictionary of the annotations.
	Output:
		ao (numpy vector): vector of active objects where the active elements have been multiplied by a decay factor between [0, N] times, where N is the number of objects for that atomic actions.
	"""
	
	ao_idx = annotations['object_label'][action_idx]	
	ao = np.zeros((N_OBJECTS))
	
	if type(ao_idx) == int:
		pass
	else:
		for idx in ao_idx:
			ao = 0.6 * ao
			ao[idx] = 1
	
	return ao	


def get_time_step():
	"""
	Gets action idx (from the annotations) with respect to the current frame of the video.
	
	Output:
		action_idx (int): index of the next atomic action for a given frame. 
	"""
	global action_idx, frame, annotations
	#time.sleep(1)
	frame += 15
	#time.sleep(1/30)


	#if frame % freq_obs == 0:
	#if frame > annotations['frame_init'][action_idx]:
	if frame > annotations['frame_end'][action_idx]:
		action_idx +=1
		
	return action_idx

def get_state_v2(version=1):
	global action_idx, frame, annotations
	length = len(annotations['label'])
	
	action_idx = get_time_step() #Current idx based on current frame
	
	if action_idx < length:

		if frame <= annotations['frame_end'][action_idx]:
			na = get_next_action_annotations(action_idx, annotations)
			ao = get_active_object_annotations(action_idx, annotations)
		else: 
			na = get_next_action_annotations(action_idx+1, annotations)
			ao = get_active_object_annotations(action_idx+1, annotations)
		"""	
		print("\nCurrent frame: ", frame)
		print("Current idx: ", action_idx)
		print("Length: ", length)
		print("Current action: ", annotations['label'][action_idx])
		print("Current frame init: ", annotations['frame_init'][action_idx])
		print("Current frame end: ", annotations['frame_end'][action_idx])
		print("DIFF init: ", annotations['frame_init'][action_idx] - frame)
		"""	
	else:

		na = one_hot(-1, N_ATOMIC_ACTIONS)
		ao = np.zeros((N_OBJECTS))
		#print("FINNNNNNNNNNNNNNNNNNNNNNN")
		

	
	
	#print("NA: ", na)
	if version == 1:
		state = na
	elif version == 2:
		state = concat_vectors(na, ao)	
	
	return state



def get_frame():
	global frame
	
	return frame

def get_state(version=1):
	"""
	Gets the state of the environment as the predicted next action (and the active objects of that action).
		
	Input:
		version (int): version of the state definition. If version is 1, the state is the next atomic action. If version is 2, it is also the active object.
	Output:
		state (numpy vector): concatenation (v2) of one-hot encoded next atomic action and active object vectors. 
	"""
	global action_idx, annotations
	
	
	length = len(annotations['label'])
	
	#-!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#action_idx += 1 #Transition to the next state (action index) inmediately
	action_idx = get_time_step() #Transition to next state based on time
	
	
	if action_idx+1 < length:
		na = get_next_action_annotations(action_idx+1, annotations)
		ao = get_active_object_annotations(action_idx, annotations)
	
		#print("NEXT ACTION: ", na)
		#print("ACTIVE OBJECT: ", ao)
	
	else: 
		#print("Terminal state") #TERMINAL STATE: NextAction is null because the current state is the last action of the recipe
		na = one_hot(-1, N_ATOMIC_ACTIONS)
		ao = np.zeros((N_OBJECTS))
		
	if version == 1:
		state = na
	elif version == 2:
		state = concat_vectors(na, ao)	

	return state	
	
	
	
# 3) GET REWARDS	
#User interfaces/Reward functions.
#---------------------------------------------------------------------------


def get_reward_keyboard():
	"""
	Returns an integer value read from keyboard input.
	Output:
		reward (int): Input value or 0 if the user did not provide a valid number.
	"""
	reward = input("Input reward value...\n")
	
	try:
		return int(reward)
	except: 
		print("ERROR: invalid reward value.")
		return 0	
	

def get_emotion():
	"""
	Returns an integer value based on the facial expression, as a reward signal.
	Output:
		reward (int): reward value provided by the facial expression recognition system. 
	
	"""
	reward = ... #whatever function calls the emotion recognition system
	try:
		return int(reward)
	except:
		print("Invalid value")
		return 0
	

def get_reward_GUI():
	"""
	Gets a reward signal from a Graphical User Interface with three buttons: Negative (-1), Neutral (0) and Positive (+1).
	
	Output:
		reward (int): reward value provided by the user. 
	
	"""
	button_size = (25, 15)	
	reward = 0
	
	#Button layout (as a matrix)
	interface = [[
	sg.Button('NEGATIVE', size=button_size, key='Negative', button_color='red'), 
	sg.Button('NEUTRAL', key='Neutral', size=button_size, button_color='gray'), 
	sg.Button('POSITIVE', key='Positive', size=button_size, button_color='blue')
	]]
	
	#Generate window with the button layout
	window = sg.Window('Interface', interface, background_color='black', return_keyboard_events=True).Finalize()	
	
	while True:
		event, values = window.read()
		
		if event == sg.WIN_CLOSED or event == 'Negative':
			reward = -1
			break
		
		elif event == sg.WIN_CLOSED or event == 'Neutral':
			reward = 0
			break
		
		elif event == sg.WIN_CLOSED or event == 'Positive':
			reward = +1
			break	
		
		elif event == '1:10': #Keyboard press 1
			window['Negative'].click()
		
		elif event == '2:11': #Keyboard press 2
			window['Neutral'].click()
		
		elif event == '3:12': #Keyboard press 3
			window['Positive'].click()
			
		elif event == 'q:24': #Keyboard press q
			break	
			
	window.close()
	
	return reward


def reward_confirmation_perform(action):
	"""
	Performs an action after receiving confirmation (POSITIVE reward) or cancels the operation if received NEGATIVE/NEUTRAL reward. The reward is offered through a Graphical User Interface with three buttons: Negative (-1), Neutral (0) and Positive (+1).
	
	Input:
		action (int): action-output of the DQN (according to an exploration-exploitation policy).
	
	Output:
		reward (int): reward value provided by the user. 
	
	"""
	button_size = (25, 15)	
	reward = 0
	
	#Button layout (as a matrix)
	interface = [[
	sg.Button('NEGATIVE', size=button_size, key='Negative', button_color='red'), 
	sg.Button('NEUTRAL', key='Neutral', size=button_size, button_color='gray'), 
	sg.Button('POSITIVE', key='Positive', size=button_size, button_color='blue')
	]]
	
	#Generate window with the button layout	
	
	if action != 18:
		window = sg.Window('Interface', interface, background_color='black', return_keyboard_events=True).Finalize()	
		print("\nROBOT: I'm going to", cfg.ROBOT_ACTIONS_MEANINGS[action])
	
		while True:
			event, values = window.read()
		
			if event == sg.WIN_CLOSED or event == 'Negative':
				reward = -1
				break
		
			elif event == sg.WIN_CLOSED or event == 'Neutral':
				reward = 0
				break
		
			elif event == sg.WIN_CLOSED or event == 'Positive':
				reward = +1
				break	
				
			elif event == '1:10': #Keyboard press 1
				window['Negative'].click()
				
			elif event == '2:11': #Keyboard press 2
				window['Neutral'].click()
			
			elif event == '3:12': #Keyboard press 3
				window['Positive'].click()	
		window.close()
	
		global frame
	
		#Confirmation
		if reward == 1:
			print("PERFORMING ACTION - ", cfg.ROBOT_ACTIONS_MEANINGS[action], "\n")
			for i in range(10):
				print("."*(i+1), end='\r')
				time.sleep(0.25)
			#time.sleep(1)
			frame += 300
		
		else: 
			#print("Or maybe not.")
			pass	
	
	
	return reward

def get_sentiment_keyboard():
	"""
	Returns an integer reward value extracted from the sentiment analysis of an input sentence.
	
	Output:
		reward: (int) value +1 if text was positive, -1 if text was negative, 0 if neutral.

	"""
	sentence = input("Type text\n")
	analyzer = SentimentIntensityAnalyzer()
	
	score = analyzer.polarity_scores(sentence)
	#print("Score : ", score['compound'])
		
	if score['compound'] > 0.1: reward = 1
	elif score['compound'] < -0.1: reward = -1
	else : reward = 0
	
	#print("Sentiment - ", score['compound'])
	#print("Reward - ", reward)
	

	return reward


"""
while True:
	s = get_state_v2()
	
"""

#---------------------------------
#Debug
"""
na = get_next_action()
na2 = get_init_state()
na3 = get_end_state()
na4 = get_random_state()

ao = get_active_object()


s = get_state()
s_ext = get_state_extended()

na_, vwm_ = undo_concat_state(s)
na__, vwm__, ao__ = undo_concat_state_extended(s_ext)

print("N ATOMIC ACTIONS: ", N_ATOMIC_ACTIONS)
print("N OBJECTS: ", N_OBJECTS)
print("\nNA prob of shape ", na.shape, "\n", na)
print("\nNA init of shape ", na2.shape, "\n", na2)
print("\nNA end of shape ", na3.shape, "\n", na3)
print("\nNA hard of shape ", na4.shape, "\n", na4)
print("\nActive Obj of shape ", ao.shape, "\n", ao)
print("\nVWM of shape ", vwm.shape , "\n", vwm)
print("\nState (NA + VWM) of shape ", s.shape, "\n", s)
print("\nExtended state (NA + VWM + AO) of shape ", s_ext.shape, "\n", s_ext)
print("\nUncat NA of shape ", na__.shape, "\n", na__)
print("\nUncat VWM of shape ", vwm__.shape, "\n", vwm__)
print("\nUncat AO of shape ", ao__.shape, "\n", ao__)

"""






	

