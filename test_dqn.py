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

from generate_history import *

#ARGUMENTS
import config as cfg
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model was saved during training. For example: my_first_DQN.")
parser.add_argument('--load_episode', type=int, default=cfg.LOAD_EPISODE, help="(int) Number of episode to load from the EXPERIMENT_NAME folder, as the sufix added to the checkpoints when the save files were created. For example: 500, which will load 'model_500.pt'.")
parser.add_argument('--root', type=str, default=cfg.ROOT, help="(str) Name of the root folder for the saving of checkpoints. Parent folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/")
#parser.add_argument('--num_episodes', type=int, default=1000, help="(int) Number of episodes.")
parser.add_argument('--eps_test', type=float, default=0.0, help="(float) Exploration rate for the action-selection during test. For example: 0.05")
parser.add_argument('--display', action='store_true', default=False, help="Display environment info as [Current state, action taken, transitioned state, immediate reward, total reward].")
parser.add_argument('--cuda', action='store_true', default=False, help="Use GPU if available.")
args = parser.parse_args()



#CONFIGURATION
ROOT = args.root
EXPERIMENT_NAME = args.experiment_name
LOAD_EPISODE = args.load_episode
#NUM_EPISODES = args.num_episodes
EPS_TEST = args.eps_test


device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

NUM_EPISODES = len(glob.glob("./video_annotations/test/*")) #Run the test only once for every video in the testset


total_CA_intime = []
total_CA_late = []
total_IA_intime = []
total_IA_late = []
total_UA_intime = []
total_UA_late = []
total_CI = []
total_II = []



#TEST 
#----------------
#Environment - Custom basic environment for kitchen recipes
env = gym.make("gym_basic:basic-v0", display=args.display, test=True, disable_env_checker=True)


n_actions = env.action_space.n
n_states = env.observation_space.n

policy_net = DQN(n_states, n_actions).to(device)

#LOAD MODEL from 'EXPERIMENT_NAME' folder
path = os.path.join(ROOT, EXPERIMENT_NAME)
if LOAD_EPISODE:
    model_name = 'model_' + str(LOAD_EPISODE) + '.pt'
    full_path = os.path.join(path, model_name)
else:
        try: 
            best = open(ROOT + EXPERIMENT_NAME + "/best_episode.txt", mode='r')
            best_episode = best.read().splitlines()[0]
            LOAD_EPISODE = int(best_episode)
            model_name = 'model_' + best_episode + '.pt'
            full_path = os.path.join(path, model_name)
            
        except:
            list_of_files = glob.glob(path+ '/*') 
            full_path = max(list_of_files, key=os.path.getctime) #Get the latest file in directory

print("\nLoading model from ", full_path)
checkpoint = torch.load(full_path)
policy_net.load_state_dict(checkpoint['model_state_dict'])


#TEST Select action function, with greedy policy, epsilon = EPS_TEST*.
#*Maybe consider to set epsilon to 0 so that every action is optimal.
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

        

#TEST EPISODE LOOP
print("\nTESTING...")
print("=========================================")

policy_net.eval()
total_reward = []
steps_done = 0

total_reward_energy = []
total_reward_time = []
total_reward_energy_ep = []
total_reward_time_ep = []
total_times_execution = []
decision_cont = 0


flag_do_action = True 


for i_episode in range(NUM_EPISODES):
    if(args.display): print("| EPISODE #", i_episode , end='\n')
    else: print("| EPISODE #", i_episode , end='\r')

    state = torch.tensor(env.reset(), dtype=torch.float, device=device).unsqueeze(0)

    done = False
    
    steps_done += 1
    num_optim = 0
    
    action = select_action(state) #1
    
    reward_energy_ep = 0
    reward_time_ep = 0
    
    for t in count(): 
        # action = select_action(state).item()
        decision_cont += 1
        action, flag_decision = action_rate(decision_cont, state)
        
        if flag_decision: 
            action_ = action
            action = action.item()
            
        array_action = [action,flag_decision]
        prev_state, next_state, reward, done, optim, flag_pdb, reward_time, reward_energy, execution_times = env.step(array_action)
        #print("Frame: ", frame)
        reward = torch.tensor([reward], device=device)
                
        reward_energy_ep += reward_energy
        reward_time_ep += reward_time
        # print("reward time: ", reward_time)
        # print("reward energy: ", reward_energy)
        # print("reward time + reward energy: ", reward_time + reward_energy)
        # print("reward: ", reward)
        
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


    print("CORRECT ACTIONS (in time): ", env.CA_intime)
    print("CORRECT ACTIONS (late): ", env.CA_late)
    print("INCORRECT ACTIONS (in time): ", env.IA_intime)
    print("INCORRECT ACTIONS (late): ", env.IA_late)
    print("UNNECESSARY ACTIONS (in time): ", env.UA_intime)
    print("UNNECESSARY ACTIONS (late): ", env.UA_late)
    print("CORRECT INACTIONS: ", env.CI)
    print("INCORRECT INACTIONS: ", env.II)
    print("")

print("\n\n TEST COMPLETED.\n")
print(" RESULTS ")
print("="*33)
print("| Average reward       | {:5.2f}".format(mean(total_reward)), " |")
print("| Best episode reward  | {:5.2f}".format(max(total_reward)), " |")
print("| Worst episode reward | {:5.2f}".format(min(total_reward)), " |")
print("="*33)


save_path = os.path.join(path, "Graphics") 
if not os.path.exists(save_path): os.makedirs(save_path)


#Plots
fig = plt.figure(figsize=(15, 10))
plt.title("Total reward")
plt.xlabel("Video")
plt.ylabel("Episode reward")
plt.stem(total_reward)
plt.show()
fig.savefig(save_path+'/TEST_reward_results.jpg')


fig = plt.figure(figsize=(20, 10))
plt.subplot(241)
plt.title("Correct actions (in time)")
plt.stem(total_CA_intime)
plt.subplot(242)
plt.title("Correct actions (late)")
plt.stem(total_CA_late)
plt.subplot(243)
plt.title("Incorrect actions (in time)")
plt.stem(total_IA_intime)
plt.subplot(244)
plt.title("Incorrect actions (late)")
plt.stem(total_IA_late)
plt.subplot(245)
plt.title("Unnecessary actions (in time)")
plt.stem(total_UA_intime)
plt.subplot(246)
plt.title("Unnecessary actions (late)")
plt.stem(total_UA_late)
plt.subplot(247)
plt.title("Correct inactions")
plt.stem(total_CI)
plt.subplot(248)
plt.title("Incorrect inactions")
plt.stem(total_II)
plt.show()

fig.savefig(save_path+'/test_detailed_results.jpg')

# total_results = [total_CA_intime,total_CA_late,total_IA_intime,total_IA_late,total_UA_intime,total_UA_late,total_CI,total_II]

# n = 10
# plot_detailed_results(n, total_results, save_path, 'TEST')

# n = 100
# plot_detailed_results(n, total_results, save_path, 'TEST')

total_time_video = list(list(zip(*total_times_execution))[0])
total_time_iteraction = list(list(zip(*total_times_execution))[1])

lower_iteraction_time = [False]*len(total_time_video)

for idx, val in enumerate(total_time_video):
    
    if total_time_iteraction[idx] < val: 
        lower_iteraction_time[idx] = True
        








for i in range(10):
	create_graph(save_path, i)








        
fig3 = plt.figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')

# colors = ['red','green']
plt.yticks([1.0, 0.0], ["Iteraction time < Video",
                        "Video > Iteraction time"])

colors = np.where(lower_iteraction_time, 'green', 'red')
plt.scatter(range(0,len(lower_iteraction_time)), lower_iteraction_time, c=colors)


# plt.scatter(x=range(0,len(lower_iteraction_time)),y = lower_iteraction_time, c= list(map(colors.get, lower_iteraction_time)), marker='d')#plt.cm.get_cmap('RdBu'))
plt.title("Time execution")
plt.xlabel("Video")
plt.show()

fig3.savefig(save_path+'/TEST_time_execution_boolean.jpg')

fig1 = plt.figure(figsize=(15, 6))

plt.plot([x - y for x, y in zip(total_time_video, total_time_iteraction)])
# plt.plot(total_time_iteraction)
plt.legend(["Video - Iteraction"])
plt.xlabel("Video")
plt.ylabel("Frames")
plt.title("Time execution")
plt.show()

fig1.savefig(save_path+'/TEST_execution_time_v2.jpg')

fig1 = plt.figure(figsize=(15, 6))


plt.plot(total_reward_energy_ep)

plt.legend(["Energy reward"])
plt.xlabel("Video")
plt.ylabel("Reward")
plt.title("Reward")
plt.show()

fig1.savefig(save_path+'/TEST_energy_reward.jpg')

fig1 = plt.figure(figsize=(15, 6))


plt.plot(total_reward_time_ep)

plt.legend(["Time reward"])
plt.xlabel("Video")
plt.ylabel("Reward")
plt.title("Reward")
plt.show()

fig1.savefig(save_path+'/TEST_time_reward.jpg')


	

