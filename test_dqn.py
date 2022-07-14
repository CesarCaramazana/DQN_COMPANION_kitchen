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

from DQN import DQN
import torch
import torch.nn as nn

#ARGUMENTS
import config as cfg
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model was saved during training. For example: my_first_DQN.")
parser.add_argument('--load_episode', type=int, default=cfg.LOAD_EPISODE, help="(int) Number of episode to load from the EXPERIMENT_NAME folder, as the sufix added to the checkpoints when the save files were created. For example: 500, which will load 'model_500.pt'.")
parser.add_argument('--root', type=str, default=cfg.ROOT, help="(str) Name of the root folder for the saving of checkpoints. Parent folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/")
parser.add_argument('--num_episodes', type=int, default=1000, help="(int) Number of episodes.")
parser.add_argument('--eps_test', type=float, default=0.0, help="(float) Exploration rate for the action-selection during test. For example: 0.05")
parser.add_argument('--display', action='store_true', default=False, help="Display environment info as [Current state, action taken, transitioned state, immediate reward, total reward].")
parser.add_argument('--cuda', action='store_true', default=False, help="Use GPU if available.")
args = parser.parse_args()



#CONFIGURATION
ROOT = args.root
EXPERIMENT_NAME = args.experiment_name
LOAD_EPISODE = args.load_episode
NUM_EPISODES = args.num_episodes
EPS_TEST = args.eps_test

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


#TEST 
#----------------
#Environment - Custom basic environment for kitchen recipes
env = gym.make("gym_basic:basic-v0", display=args.display, disable_env_checker=True)
env.reset()


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
		return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long) #Take random action


#TEST EPISODE LOOP
print("\nTESTING...")
print("=========================================")

policy_net.eval()
total_reward = []


for i_episode in range(NUM_EPISODES):
	print("| EPISODE #", i_episode , end='\r')

	state = torch.tensor(env.reset(), dtype=torch.float, device=device).unsqueeze(0) #Get initial state
	
	done = False

	for t in count():
		action = select_action(state) #Select action
		_, reward, done, _ = env.step(action.item()) #Take action and receive reward
		reward = torch.tensor([reward], device=device)
		next_state = torch.tensor(env.state, dtype=torch.float, device=device).unsqueeze(0) #Transition to next state

		if not done: 
			state = next_state
		else: 
			next_state = None
		
		if done: #When the episode is finished, we save the cumulative reward in the list 'total_reward'.
			total_reward.append(env.get_total_reward())
			break


print("\n\n TEST COMPLETED.\n")
print(" RESULTS ")
print("="*33)
print("| Average reward       | {:.2f}".format(mean(total_reward)), " |")
print("| Best episode reward  | {:.2f}".format(max(total_reward)), " |")
print("| Worst episode reward | {:.2f}".format(min(total_reward)), " |")
print("="*33)


#Plots
plt.title("Total reward")
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.plot(total_reward, 'b')
plt.show()

