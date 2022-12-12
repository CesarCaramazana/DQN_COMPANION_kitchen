import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random

import numpy as np
import matplotlib.pyplot as plt

from itertools import count
from statistics import mean
import os
import glob

import torch
import torch.nn as nn

from DQN import DQN, ReplayMemory, Transition, init_weights 
from config import print_setup
import config as cfg
from aux import *
from natsort import natsorted, ns

from generate_history import *
#from test_dqn import select_action, action_rate

#ARGUMENTS
import config as cfg
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model was saved during training. For example: my_first_DQN.")

parser.add_argument('--root', type=str, default=cfg.ROOT, help="(str) Name of the root folder for the saving of checkpoints. Parent folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/")
#parser.add_argument('--num_episodes', type=int, default=1000, help="(int) Number of episodes.")
parser.add_argument('--eps_test', type=float, default=0.0, help="(float) Exploration rate for the action-selection during test. For example: 0.05")
parser.add_argument('--display', action='store_true', default=False, help="Display environment info as [Current state, action taken, transitioned state, immediate reward, total reward].")
parser.add_argument('--cuda', action='store_true', default=True, help="Use GPU if available.")
args = parser.parse_args()






def select_action(state):
    sample = random.random()
    eps_threshold = EPS_TEST

    if sample > eps_threshold:
        with torch.no_grad():
            out = policy_net(state)  #Take optimal action

            return out.max(1)[1].view(1,1)

    else: 
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long) 
        
        
        
def action_rate(decision_cont,state):
   
    if decision_cont % cfg.DECISION_RATE == 0: 
        action_selected = select_action(state)
        flag_decision = True 
    else:
        action_selected = 18
        flag_decision = False
    
    return action_selected, flag_decision






#CONFIGURATION
ROOT = args.root
EXPERIMENT_NAME = args.experiment_name

#NUM_EPISODES = args.num_episodes
EPS_TEST = args.eps_test


device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

NUM_EPISODES = len(glob.glob("./video_annotations/test/*")) #Run the test only once for every video in the testset


#TEST 
#----------------
#Environment - Custom basic environment for kitchen recipes
env = gym.make("gym_basic:basic-v0", display=args.display, test=True, disable_env_checker=True)


n_actions = env.action_space.n
n_states = env.observation_space.n

policy_net = DQN(n_states, n_actions).to(device)

#LOAD MODEL from 'EXPERIMENT_NAME' folder
path = os.path.join(ROOT, EXPERIMENT_NAME)

print("PATH: ", path)


# Get all the .pt files in the folder 
pt_extension = path + "/*.pt"
pt_files = glob.glob(pt_extension)

pt_files = natsorted(pt_files, key=lambda y: y.lower()) #Sort in natural order


#print("PATH: ", path)
#print("Files: ", pt_files)



epoch_test = []
epoch_CA_intime = []
epoch_CA_late = []
epoch_IA_intime = []
epoch_IA_late = []
epoch_UA_intime= []
epoch_UA_late = []
epoch_CI = []
epoch_II = []


for f in pt_files:
	print(f)
	epoch = int(f.replace(path + "/model_", '').replace('.pt', ''))
	#print(epoch)
	
	epoch_test.append(epoch)
	
	checkpoint = torch.load(f)
	policy_net.load_state_dict(checkpoint['model_state_dict'])
	policy_net.eval()
	
		
	
	steps_done = 0
	
	total_reward = []
	total_reward_energy = []
	total_reward_time = []
	total_reward_energy_ep = []
	total_reward_time_ep = []
	total_times_execution = []
	total_CA_intime = []
	total_CA_late = []
	total_IA_intime = []
	total_IA_late = []
	total_UA_intime = []
	total_UA_late = []
	total_CI = []
	total_II = []
	
	
	
	decision_cont = 0

	flag_do_action = True 



	for i_episode in range(NUM_EPISODES):

	    print("| EPISODE #", i_episode , end='\r')

	    state = torch.tensor(env.reset(), dtype=torch.float, device=device).unsqueeze(0)

	    done = False
	    
	    steps_done += 1
	    num_optim = 0
	    
	    action = select_action(state) #1
	    
	    reward_energy_ep = 0
	    reward_time_ep = 0
	    
	    for t in count(): 
	    	decision_cont += 1
	    	action, flag_decision = action_rate(decision_cont, state)
	    	
	    	if flag_decision: 
	    		action_ = action
	    		action = action.item()
	    	
	    	array_action = [action,flag_decision]
	    	prev_state, next_state, reward, done, optim, flag_pdb, reward_time, reward_energy, execution_times = env.step(array_action)
	    	reward = torch.tensor([reward], device=device)
	    	reward_energy_ep += reward_energy
	    	reward_time_ep += reward_time
	    	
	    	prev_state = torch.tensor([prev_state], dtype=torch.float,device=device)
	    	next_state = torch.tensor([next_state], dtype=torch.float,device=device)
	    	
	    	if not done: 
	    		state = next_state
	    	else:
	    		next_state = None
	    	
	    	if done:
	    		total_times_execution.append(execution_times)
	    		total_reward_energy_ep.append(reward_energy_ep)
	    		total_reward_time_ep.append(reward_time_ep)
	    		total_reward.append(env.get_total_reward())
	    		total_CA_intime.append(env.CA_intime)
	    		total_CA_late.append(env.CA_late)
	    		total_IA_intime.append(env.IA_intime)
	    		total_IA_late.append(env.IA_late)
	    		total_UA_intime.append(env.UA_intime)
	    		total_UA_late.append(env.UA_late)
	    		total_CI.append(env.CI)
	    		total_II.append(env.II)
	    		
	    		break #Finish episode

	

	"""
	print("TOTAL CORRECT ACTIONS (in time): ", np.sum(total_CA_intime))
	print("TOTAL CORRECT ACTIONS (late): ", np.sum(total_CA_late))
	print("TOTAL INCORRECT ACTIONS (in time): ", np.sum(total_IA_intime))
	print("TOTAL INCORRECT ACTIONS (late): ", np.sum(total_IA_late))
	print("TOTAL UNNECESSARY ACTIONS (in time): ", np.sum(total_UA_intime))
	print("TOTAL UNNECESSARY ACTIONS (late): ", np.sum(total_UA_late))
	print("TOTAL CORRECT INACTIONS: ", np.sum(total_CI))
	print("TOTAL INCORRECT INACTIONS: ", np.sum(total_II))
	print()
	"""
	
	epoch_CA_intime.append(np.sum(total_CA_intime))
	epoch_CA_late.append(np.sum(total_CA_late))
	epoch_IA_intime.append(np.sum(total_IA_intime))
	epoch_IA_late.append(np.sum(total_IA_late))
	epoch_UA_intime.append(np.sum(total_UA_intime))
	epoch_UA_late.append(np.sum(total_UA_late))
	epoch_CI.append(np.sum(total_CI))
	epoch_II.append(np.sum(total_II))
 



# SORT LISTS in ascending order of epoch

epoch_CA_intime = [x for y, x in sorted(zip(epoch_test, epoch_CA_intime))]
epoch_CA_late = [x for y, x in sorted(zip(epoch_test, epoch_CA_late))]
epoch_IA_intime = [x for y, x in sorted(zip(epoch_test, epoch_IA_intime))]
epoch_IA_late = [x for y, x in sorted(zip(epoch_test, epoch_IA_late))]
epoch_UA_intime = [x for y, x in sorted(zip(epoch_test, epoch_UA_intime))]
epoch_UA_late = [x for y, x in sorted(zip(epoch_test, epoch_UA_late))]
epoch_CI = [x for y, x in sorted(zip(epoch_test, epoch_CI))]
epoch_II = [x for y, x in sorted(zip(epoch_test, epoch_II))]
epoch_test = sorted(epoch_test)



save_path = os.path.join(path, "Graphics") 
if not os.path.exists(save_path): os.makedirs(save_path)


fig = plt.figure(figsize=(20, 15))
plt.subplot(241)
plt.title("Correct actions (in time) [through epochs]")
plt.plot(epoch_test, epoch_CA_intime)
plt.xlabel("Epochs")

plt.subplot(242)
plt.title("Correct actions (late) [through epochs]")
plt.plot(epoch_test, epoch_CA_late)
plt.xlabel("Epochs")

plt.subplot(243)
plt.title("Incorrect actions (in time) [through epochs]")
plt.plot(epoch_test, epoch_IA_intime)
plt.xlabel("Epochs")

plt.subplot(244)
plt.title("Incorrect actions (late) [through epochs]")
plt.plot(epoch_test, epoch_IA_late)
plt.xlabel("Epochs")

plt.subplot(245)
plt.title("Unnecessary actions (in time) [through epochs]")
plt.plot(epoch_test, epoch_UA_intime)
plt.xlabel("Epochs")

plt.subplot(246)
plt.title("Unnecessary actions (late) [through epochs]")
plt.plot(epoch_test, epoch_UA_late)
plt.xlabel("Epochs")

plt.subplot(247)
plt.title("Correct inactions [through epochs]")
plt.plot(epoch_test, epoch_CI)
plt.xlabel("Epochs")

plt.subplot(248)
plt.title("Incorrect inactions [through epochs]")
plt.plot(epoch_test, epoch_II)
plt.xlabel("Epochs")

plt.show()

fig.savefig(save_path+'/00_testEPOCHS.jpg')




#-----------------------------




