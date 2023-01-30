
"""

This script runs the test for a model that was saved in different training epochs.
The only required parameter is the "experiment_name", which should be the name of the folder where the different checkpoints were saved.
The checkpoints are named as "model_#epoch.pt". 

"""




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
parser.add_argument('--train', action='store_true', default=False, help="Run test over the training set.")
args = parser.parse_args()


state_dic = cfg.ATOMIC_ACTIONS_MEANINGS   
action_dic = cfg.ROBOT_ACTIONS_MEANINGS 



def post_processed_possible_actions(out,index_posible_actions,posible_actions):
    
    action_pre_processed = out.max(1)[1].view(1,1)
     
    if action_pre_processed.item() in index_posible_actions:
        return action_pre_processed
    else:
        out = out.cpu().numpy()
        out = out[0]
        posible_actions = np.asarray(posible_actions)
        action = np.argmax(posible_actions * out)
        return torch.tensor([[action]], device=device, dtype=torch.long)
    

def select_action(state):
    
    #state_name = state_dic[undo_one_hot(state[0].detach().cpu().numpy())]
    
    posible_actions = env.possible_actions_taken_robot()
    index_posible_actions = [i for i, x in enumerate(posible_actions) if x==1]
    
    with torch.no_grad():
    	out = policy_net(state)
    
    best_action = post_processed_possible_actions(out, index_posible_actions, posible_actions)	
    #output = out.detach().cpu().numpy().squeeze()	

    #print("\nState: ", state_name)
    #print("Action taken: ", action_dic[best_action[0].detach().cpu().numpy()[0]])
    """
    if state_name == 'put toaster' or state_name != '':

        	print("STATE= ", state_name)
        	for i in range(output.shape[0]):
        		print("%2i | Q-value: %4.4f" %(i,output[i]))
        	print("\nAction returned: ", best_action)
    """
    return best_action

        
        
        
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






#TEST 
#----------------
#Environment - Custom basic environment for kitchen recipes
env = gym.make("gym_basic:basic-v0", display=args.display, test=not args.train, disable_env_checker=True)


if env.test: 
	NUM_EPISODES = len(glob.glob("./video_annotations/test/*")) #Run the test only once for every video in the testset
	print("Test set")
else:
	NUM_EPISODES = len(glob.glob("./video_annotations/train/*"))
	print("Train set")
	
	
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
epoch_UAC_intime= []
epoch_UAC_late = []
epoch_UAI_intime = []
epoch_UAI_late = []
epoch_CI = []
epoch_II = []

epoch_reward = []
epoch_total_times_execution = []
epoch_total_reward_energy_ep = []
epoch_total_reward_time_ep = []

epoch_total_time_video = []
epoch_total_time_interaction = []

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
	total_UAC_intime = []
	total_UAC_late = []
	total_UAI_intime = []
	total_UAI_late = []
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
	    	
	    	array_action = [action,flag_decision, 'val']
	    	prev_state, next_state, reward, done, optim, flag_pdb, reward_time, reward_energy, execution_times = env.step(array_action)
	    	
	    	
	    	reward = torch.tensor([reward], device=device)
	    	reward_energy_ep += reward_energy
	    	reward_time_ep += reward_time
	    	
	    	#print("Reward: ", reward)
	    	
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
	    		total_UAC_intime.append(env.UAC_intime)
	    		total_UAC_late.append(env.UAC_late)
	    		total_UAI_intime.append(env.UAI_intime)
	    		total_UAI_late.append(env.UAI_late)
	    		total_CI.append(env.CI)
	    		total_II.append(env.II)
	    		
	    		total_time_video = list(list(zip(*total_times_execution))[0])
	    		total_time_interaction = list(list(zip(*total_times_execution))[1])
	    		
	    		#print(total_time_video)
	    		#print(total_time_iteraction)
	    		
	    		break #Finish episode

	epoch_total_times_execution.append(np.sum(total_times_execution))
	epoch_total_reward_energy_ep.append(np.sum(total_reward_energy_ep))
	epoch_total_reward_time_ep.append(np.sum(total_reward_time_ep))

	epoch_total_time_video.append(np.sum(total_time_video))
	epoch_total_time_interaction.append(np.sum(total_time_interaction))

	epoch_CA_intime.append(np.sum(total_CA_intime))
	epoch_CA_late.append(np.sum(total_CA_late))
	epoch_IA_intime.append(np.sum(total_IA_intime))
	epoch_IA_late.append(np.sum(total_IA_late))
	epoch_UAC_intime.append(np.sum(total_UAC_intime))
	epoch_UAC_late.append(np.sum(total_UAC_late))
	epoch_UAI_intime.append(np.sum(total_UAI_intime))
	epoch_UAI_late.append(np.sum(total_UAI_late))
	epoch_CI.append(np.sum(total_CI))
	epoch_II.append(np.sum(total_II))
	
	epoch_reward.append(np.sum(total_reward))
 



# SORT LISTS in ascending order of epoch

"""
epoch_CA_intime = [x for y, x in sorted(zip(epoch_test, epoch_CA_intime))]
epoch_CA_late = [x for y, x in sorted(zip(epoch_test, epoch_CA_late))]
epoch_IA_intime = [x for y, x in sorted(zip(epoch_test, epoch_IA_intime))]
epoch_IA_late = [x for y, x in sorted(zip(epoch_test, epoch_IA_late))]
epoch_UA_intime = [x for y, x in sorted(zip(epoch_test, epoch_UA_intime))]
epoch_UA_late = [x for y, x in sorted(zip(epoch_test, epoch_UA_late))]
epoch_CI = [x for y, x in sorted(zip(epoch_test, epoch_CI))]
epoch_II = [x for y, x in sorted(zip(epoch_test, epoch_II))]
epoch_test = sorted(epoch_test)
"""


save_path = os.path.join(path, "Graphics") 
if not os.path.exists(save_path): os.makedirs(save_path)


fig = plt.figure(figsize=(34, 12))
plt.subplot(2,5,1)
plt.title("Correct actions (in time)")
plt.plot(epoch_test, epoch_CA_intime)
plt.xlabel("Epochs")

plt.subplot(2,5,2)
plt.title("Incorrect actions (in time)")
plt.plot(epoch_test, epoch_IA_intime)
plt.xlabel("Epochs")


plt.subplot(2,5,3)
plt.title("Unnecessary actions correct (in time)")
plt.plot(epoch_test, epoch_UAC_intime)
plt.xlabel("Epochs")


plt.subplot(2,5,4)
plt.title("Unnecessary actions incorrect (in time)")
plt.plot(epoch_test, epoch_UAI_intime)
plt.xlabel("Epochs")

plt.subplot(2,5,5)
plt.title("Correct inactions")
plt.plot(epoch_test, epoch_CI)
plt.xlabel("Epochs")



plt.subplot(2,5,6)
plt.title("Correct actions (late)")
plt.plot(epoch_test, epoch_CA_late)
plt.xlabel("Epochs")


plt.subplot(2,5,7)
plt.title("Incorrect actions (late)")
plt.plot(epoch_test, epoch_IA_late)
plt.xlabel("Epochs")


plt.subplot(2,5,8)
plt.title("Unnecessary actions correct (late)")
plt.plot(epoch_test, epoch_UAC_late)
plt.xlabel("Epochs")


plt.subplot(2,5,9)
plt.title("Unnecessary actions incorrect (late)")
plt.plot(epoch_test, epoch_UAI_late)
plt.xlabel("Epochs")



plt.subplot(2,5,10)
plt.title("Incorrect inactions")
plt.plot(epoch_test, epoch_II)
plt.xlabel("Epochs")

plt.show()

if env.test: fig.savefig(save_path+'/00_testEPOCHS_actions.jpg')
else: fig.savefig(save_path+'/00_trainEPOCHS_actions.jpg')

fig2 = plt.figure()

plt.title("Reward")
plt.plot(epoch_test, epoch_reward)
plt.xlabel("Epochs")

plt.show()

if env.test: fig2.savefig(save_path+'/00_testEPOCHS_reward.jpg')
else: fig2.savefig(save_path+'/00_trainEPOCHS_reward.jpg')
#-----------------------------




fig3 = plt.figure()
plt.title("HRI time vs annotations time")
plt.plot(epoch_test, epoch_total_time_video, label='Video')
plt.plot(epoch_test, epoch_total_time_interaction, label='Interaction')
plt.legend()
plt.ylabel("Frames")

plt.show()

if env.test: fig3.savefig(save_path+'/00_testEPOCHS_time.jpg')
else: fig3.savefig(save_path+'/00_trainEPOCHS_time.jpg')

plt.close()




