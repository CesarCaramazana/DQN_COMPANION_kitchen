
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
import config as cfg
from aux import *
from natsort import natsorted, ns

from generate_history import *

#ARGUMENTS
import config as cfg
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model was saved during training. For example: my_first_DQN.")

parser.add_argument('--root', type=str, default=cfg.ROOT, help="(str) Name of the root folder for the saving of checkpoints. Parent folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/")
#parser.add_argument('--num_episodes', type=int, default=1000, help="(int) Number of episodes.")
parser.add_argument('--eps_test', type=float, default=0.0, help="(float) Exploration rate for the action-selection during test. For example: 0.05")
parser.add_argument('--display', action='store_true', default=False, help="Display environment info as [Current state, action taken, transitioned state, immediate reward, total reward].")
parser.add_argument('--cuda', action='store_true', default=True, help="Use GPU if available.")
parser.add_argument('--train', action='store_true', default=False, help="Run test over the training set.")
parser.add_argument('--debug', action='store_true', default=False, help="Use debug test set")
args = parser.parse_args()


state_dic = cfg.ATOMIC_ACTIONS_MEANINGS   
action_dic = cfg.ROBOT_ACTIONS_MEANINGS 



def post_processed_possible_actions(out,index_posible_actions):
    """
    Function that performs a post-processing of the neural network output. 
    In case the output is an action that is not available, either because 
    of the object missing or left on the table, the most likely possible action will be selected 
    from the output of the neural network,

    Parameters
    ----------
    out : (tensor)
        DQN output.
    index_posible_actions : (list)
        Posible actions taken by the robot according to the objects available.

    Returns
    -------
    (tensor)
        Action to be performed by the robot.

    """
    action_pre_processed = out.max(1)[1].view(1,1)
     
    if action_pre_processed.item() in index_posible_actions:
        return action_pre_processed
    else:
        out = out.cpu().numpy()
        out = out[0]
   
        idx = np.argmax(out[index_posible_actions])
        action = index_posible_actions[idx]

        return torch.tensor([[action]], device=device, dtype=torch.long)
    


def select_action(state):
    
    #state_name = state_dic[undo_one_hot(state[0].detach().cpu().numpy())]
    
    posible_actions = env.possible_actions_taken_robot()
    index_posible_actions = [i for i, x in enumerate(posible_actions) if x==1]
    
    with torch.no_grad():
    	out = policy_net(state)
    
    best_action = action = post_processed_possible_actions(out,index_posible_actions)
    

    
    #best_action = torch.tensor([[5]], device=device, dtype=torch.long) #Para encontrar el tiempo de HRI de un robot pasivo

    return best_action

     

        
        
def action_rate(decision_cont,state):
   
    if decision_cont % cfg.DECISION_RATE == 0: 
        action_selected = select_action(state)
        flag_decision = True 
    else:
        action_selected = 5
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
env = gym.make("gym_basic:basic-v0", display=args.display, test=not args.train, debug=args.debug, disable_env_checker=True)


if env.test: 
    if args.debug:
        NUM_EPISODES = len(glob.glob("./video_annotations/5folds/"+cfg.TEST_FOLD+"/test_debug/*")) #Run the test only once for every video in the testset
        print("Debug set")
        root = './video_annotations/5folds/'+cfg.TEST_FOLD+'/test_debug/*'
    else:
        NUM_EPISODES = len(glob.glob("./video_annotations/5folds/"+cfg.TEST_FOLD+"/test/*")) #Run the test only once for every video in the testset
        print("Test set")
        root = './video_annotations/5folds/'+cfg.TEST_FOLD+'/test/*'
        
else:
    NUM_EPISODES = len(glob.glob("./video_annotations/5folds/"+cfg.TEST_FOLD+"/train/*"))
    print("Train set")
    root = './video_annotations/5folds/'+cfg.TEST_FOLD+'/train/*'
	
video_max_times = []
video_min_times = []


videos = glob.glob(root)  

#GET VIDEO TIME AND OPTIMAL TIME (MIN)
for video in videos:
	path = video + '/human_times'
	human_times = np.load(path, allow_pickle=True)  
	
	min_time = human_times['min']
	max_time = human_times['max']
	
	video_max_times.append(max_time)
	video_min_times.append(min_time)
	
minimum_time = sum(video_min_times)
maximum_time = sum(video_max_times)

	
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
	
	total_interaction_time_epoch = []
	#total_maximum_time_execution_epoch = []
	#total_minimum_time_execution_epoch = []
	
	
	
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
	    	#next_state_, reward, done, optim, flag_pdb, reward_time, reward_energy, execution_times, correct_action, _ = env.step(array_action)
	    	next_state_, reward, done, optim, flag_pdb, reward_time, reward_energy, hri_time, correct_action, type_threshold, error_pred, total_pred = env.step(array_action)
	    	
	    	
	    	reward = torch.tensor([reward], device=device)
	    	reward_energy_ep += reward_energy
	    	reward_time_ep += reward_time

	    	next_state = torch.tensor([next_state_], dtype=torch.float,device=device)
	    	
	    	if not done: 
	    		state = next_state
	    	else:
	    		next_state = None
	    	
	    	if done:
	    		#total_times_execution.append(execution_times)
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
	    		
	    		#total_time_video = list(list(zip(*total_times_execution))[0])
	    		#total_time_interaction = list(list(zip(*total_times_execution))[1])
	    		
	    		#HRI
	    		total_interaction_time_epoch.append(hri_time)
	    		
	    		#Human baseline
	    		#total_minimum_time_execution_epoch.append(min_time)
	    		#total_maximum_time_execution_epoch.append(max_time)	    		

	    		
	    		#print(total_time_video)
	    		#print(total_time_iteraction)
	    		
	    		break #Finish episode

	#epoch_total_times_execution.append(np.sum(total_times_execution))
	epoch_total_reward_energy_ep.append(np.sum(total_reward_energy_ep))
	epoch_total_reward_time_ep.append(np.sum(total_reward_time_ep))

	#epoch_total_time_video.append(np.sum(total_time_video))
	epoch_total_time_interaction.append(np.sum(total_interaction_time_epoch)) #HRI time

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
	
	
	#maximum_time = sum(total_maximum_time_execution_epoch) #Human times
	#minimum_time = sum(total_minimum_time_execution_epoch)
	
	


 


save_path = os.path.join(path, "Graphics") 
if not os.path.exists(save_path): os.makedirs(save_path)





#--------------------ACTIONS ----------------

fig = plt.figure(figsize=(25, 8))
plt.subplot(2,5,1)
plt.title("Short-term correct actions (in time)")
plt.plot(epoch_test, epoch_CA_intime)


plt.subplot(2,5,5)
plt.title("Incorrect (in time)")
plt.plot(epoch_test, epoch_IA_intime)



plt.subplot(2,5,2)
plt.title("Long-term correct actions (in time)")
plt.plot(epoch_test, epoch_UAC_intime)



plt.subplot(2,5,4)
plt.title("Unnecessary incorrect (in time)")
plt.plot(epoch_test, epoch_UAI_intime)


plt.subplot(2,5,3)
plt.title("Correct inactions")
plt.plot(epoch_test, epoch_CI)




plt.subplot(2,5,6)
plt.title("Short-term correct actions (late)")
plt.plot(epoch_test, epoch_CA_late)
plt.xlabel("Epochs")


plt.subplot(2,5,10)
plt.title("Incorrect (late)")
plt.plot(epoch_test, epoch_IA_late)
plt.xlabel("Epochs")


plt.subplot(2,5,7)
plt.title("Long-term correct actions (late)")
plt.plot(epoch_test, epoch_UAC_late)
plt.xlabel("Epochs")


plt.subplot(2,5,9)
plt.title("Unnecessary incorrect (late)")
plt.plot(epoch_test, epoch_UAI_late)
plt.xlabel("Epochs")



plt.subplot(2,5,8)
plt.title("Incorrect inactions")
plt.plot(epoch_test, epoch_II)
plt.xlabel("Epochs")

# plt.show()

if env.test: fig.savefig(save_path+'/00_TEST_ACTIONS.jpg')
else: fig.savefig(save_path+'/00_TRAIN_ACTIONS.jpg')






 # ---------------------------------------------------------------------------------------------

# ------------------- ACTIONS ------ PIE CHART
# (ONLY IN THE LAST EPOCH)

stci = np.sum(total_CA_intime) +1 #short term correct intime
stcl = np.sum(total_CA_late) +1 #short term correct late
ltci = np.sum(total_UAC_intime) +1 #long term correct intime
ltcl = np.sum(total_UAC_late) +1 #long term correct late
ci = np.sum(total_CI) +1 #correct inactions
ii = np.sum(total_II) +1 #incorrect inactions
ui = np.sum(total_UAI_intime) +1 #unnec intime
ul = np.sum(total_UAI_late) +1 #unnec late
iai = np.sum(total_IA_intime) +1 #incorrect intime
ial = np.sum(total_IA_late) +1 #incorrect late

labels1 = 'Short-term in time', 'Short-term late', 'Long-term in time', 'Long-term late', 'Unnecessary in time', 'Unnecessary late', 'Incorrect in time', 'Incorrect late'
sizes1 = [stci, stcl, ltci, ltcl, ui, ul, iai, ial]

labels2 = 'Short-term', 'Long-term'
sizes2 = [stci + stcl, 
	ltci + ltcl]


labels3 = 'In time', 'Late'
sizes3 = [stci + ltci + ui + iai,
	stcl + ltcl + ul + ial]

labels4 = 'Useful actions', 'Useless actions' #Kinda like correct vs Incorrect+Unnecs
sizes4 = [stci + stcl + ltci + ltcl,
	ui + ul + iai + ial]	

fig1 = plt.figure(figsize=(20,10))

plt.subplot(1,4,1)
plt.title("Action decisions")
plt.pie(sizes1, labels=labels1, autopct='%1.1f%%')

plt.subplot(1,4,2)
plt.title("Short-term vs. Long-term")
plt.pie(sizes2, labels=labels2, autopct='%1.1f%%')


plt.subplot(1,4,3)
plt.title("In time vs. Late")
plt.pie(sizes3, labels=labels3, autopct='%1.1f%%', colors=['mediumseagreen', 'crimson'])


plt.subplot(1,4,4)
plt.title("Useful vs. Useless")
plt.pie(sizes4, labels=labels4, autopct='%1.1f%%', colors=['mediumseagreen', 'crimson'])

if env.test: fig1.savefig(save_path+'/00_TEST_ACTIONS_PIE.jpg')
else: fig1.savefig(save_path+'/00_TRAIN_ACTIONS_PIE.jpg')









# -------------__REWARDS -------------------------
fig2 = plt.figure(figsize=(20,6))


plt.subplot2grid((1,3),(0,0))

plt.plot(epoch_total_reward_energy_ep, 'c:')
plt.title("Energy reward")
plt.legend(["Energy reward"])
plt.xlabel("Epoch")


plt.subplot2grid((1,3),(0,1))

plt.plot(epoch_total_reward_time_ep, 'c:')
plt.title("Time reward")
plt.legend(["Time reward"])
plt.xlabel("Epoch")


plt.subplot2grid((1,3),(0,2))

plt.title("Total reward")
plt.plot(epoch_reward, 'c-.')
plt.legend(["Total reward"])
plt.xlabel("Epoch")

# plt.show()

if env.test: fig2.savefig(save_path+'/00_TEST_REWARD.jpg')
else: fig2.savefig(save_path+'/00_TRAIN_REWARD.jpg')
#-----------------------------




#--------------- INTERACTION ---------------------
fig3 = plt.figure(figsize=(15,6))
plt.title("Interaction time")
#plt.plot(epoch_total_time_video, 'k',label='Video')
plt.plot(epoch_test, epoch_total_time_interaction, 'c--',label='Interaction')
plt.axhline(y=maximum_time, color='k', label='Video')
plt.axhline(y=minimum_time, color='r', label='Minimum')
plt.legend()
plt.ylabel("Frames")

# plt.show()

if env.test: fig3.savefig(save_path+'/00_TEST_INTERACTION_TIME.jpg')
else: fig3.savefig(save_path+'/00_TRAIN_INTERACTION_TIME.jpg')

plt.close()


if env.test: 
	for i in range(NUM_EPISODES):
		create_graph(save_path, i)

