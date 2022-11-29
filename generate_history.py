import numpy as np
import wavedrom
import math
import os

resolution = 30 #Subsampling factor


def generate_action_duration(history):
	"""
	Generates a history of states in the format list of [action name, action duration], from a history that has a state associated to each timestep [action name, frame].
	
	Input:
		history: (list) history in the format [action name, frame].
	
	Output:
		new: (list) history in the format [action name, action duration].	
	
	"""
	
	global resolution	
	
	history.append(None)
		
	new = []
	prev = history[0]
	duration = 0
	
	for i in range(len(history)):
		if history[i] != prev:
			if prev is not None: new.append([prev, duration])
			prev = history[i]
			duration = resolution
		else:
			duration += resolution	
	return new




def generate_signal(history):
	"""
	Generates a string associated to the history of actions, in the format required to generate a plot with wavedrom.
	Particular actions are assigned a color code for the display of the bars.
	
	Input:
		history: (list) of tuples [action name (str), action duration (int)].
	
	Output:
		actions: (list) of action names.
		waveform: (str) code for graphical visualization of action durations.	
	
	"""
	
	actions = []
	waveform = "z"
	
	global resolution
	
	color_code = "z"
	
	for i, action in enumerate(history):
		duration = math.floor(action[1]/resolution)		
		
		if action[0] == 'do nothing':
			color_code = "x"
		
		elif action[0] == 'other manipulation':
			color_code = "5"
			actions.append('other')

		elif action[0] == 'Predicting...':
			color_code = "6"
			#actions.append(action[0])
			actions.append("...")
		
		elif action[0] == 'Waiting for evaluation...' or action[0] == 'Waiting for robot action...':
			color_code = "9"
			#actions.append(action[0])
			actions.append('wait')

		else: 
			color_code = "3"
			actions.append(action[0])	
		
		waveform = waveform + color_code + "."*(duration-1)
		
	waveform = waveform + "z"

	return actions, waveform



def create_graph(save_path, file_id):
	"""
	Reads a file name/file id, in the format .npz, containing the history of an interaction between robot and human, and generates a plot representing the states of both agents during the course of the interaction. 
	
	Input:
		file_id: (int) .npz file identifier. 
		
	Output:
		image: (.png) in the specified path. 	
	
	"""
	
	global resolution
	
	path = "./results/History_Arrays/"
	
	full_path = save_path + "/Visualization/"
	
	if not os.path.exists(full_path): os.mkdir(full_path)
	
	savepath = full_path + str(file_id) +".png"
	readpath = path + str(file_id) + ".npz" 
	history = np.load(readpath)
	
	human = history['h_history'] 
	robot = history['r_history']
	
	r_history=[robot[i].item() for i in range(0,len(robot),resolution)] #Subsampling: taking the first of every 'resolution' elements
	h_history=[human[i].item() for i in range(0,len(human),resolution)]
	
	human_history = generate_action_duration(h_history)
	robot_history = generate_action_duration(r_history)
	
	h_actions, h_waveform = generate_signal(human_history)
	r_actions, r_waveform = generate_signal(robot_history)
	
	
	clk = "P" + "."*(len(r_waveform)-1) #Clock signal
	
	clock = "" + str(resolution) + " frames"
	
	d = { "signal": [
		{ "name": clock, "wave": clk},
		{ "name": "Human", "wave": h_waveform, "data": h_actions},
		{ "name": "Robot",  "wave": r_waveform ,"data": r_actions }]}
	
	
	svg = wavedrom.render(str(d))	
		
	svg.saveas(savepath)	




"""
#########################################################################################

Main



path = "./results/History_Arrays/"

for i in range(10):
	try:
		create_graph(path)
	except:
		print("No file with ID ", i)	

"""
