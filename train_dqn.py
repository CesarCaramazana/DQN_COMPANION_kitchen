import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from statistics import mean

import os
import glob
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from DQN import DQN, ReplayMemory, Transition, init_weights 
from config import print_setup
import config as cfg
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model is saved. For example: my_first_DQN.")
parser.add_argument('--save_model', action='store_true', default=False, help="Save a checkpoint in the EXPERIMENT_NAME folder.")
parser.add_argument('--load_model', action='store_true', help="Load a checkpoint from the EXPERIMENT_NAME folder. If no episode is specified (LOAD_EPISODE), it loads the latest created file.")
parser.add_argument('--load_episode', type=int, default=0, help="(int) Number of episode to load from the EXPERIMENT_NAME folder, as the sufix added to the checkpoints when the save files are created. For example: 500, which will load 'model_500.pt'.")
parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help="(int) Batch size for the training of the network. For example: 64.")
parser.add_argument('--num_episodes', type=int, default=cfg.NUM_EPISODES, help="(int) Number of episodes or training epochs. For example: 2000.")
parser.add_argument('--lr', type=float, default=cfg.LR, help="(float) Learning rate. For example: 1e-3.")
parser.add_argument('--replay_memory', type=int, default=cfg.REPLAY_MEMORY, help="(int) Size of the Experience Replay memory. For example: 1000.")
parser.add_argument('--gamma', type=float, default=cfg.GAMMA, help="(float) Discount rate of future rewards. For example: 0.99.")
parser.add_argument('--eps_start', type=float, default=cfg.EPS_START, help="(float) Initial exploration rate. For example: 0.99.")
parser.add_argument('--eps_end', type=float, default=cfg.EPS_END, help="(float) Terminal exploration rate. For example: 0.05.")
parser.add_argument('--eps_decay', type=int, default=cfg.EPS_DECAY, help="(int) Decay factor of the exploration rate. Episode where the epsilon has decay to 0.367 of the initial value. For example: num_episodes/2.")
parser.add_argument('--target_update', type=int, default=cfg.TARGET_UPDATE, help="(int) Frequency of the update of the Target Network, in number of episodes. For example: 10.")
parser.add_argument('--root', type=str, default=cfg.ROOT, help="(str) Name of the root folder for the saving of checkpoints. Parent folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/")
parser.add_argument('--display', action='store_true', default=False, help="Display environment info as [Current state, action taken, transitioned state, immediate reward, total reward].")
parser.add_argument('--cuda', action='store_true', default=False, help="Use GPU if available.")
args = parser.parse_args()


SAVE_MODEL = args.save_model
SAVE_EPISODE = cfg.SAVE_EPISODE
EXPERIMENT_NAME = args.experiment_name
LOAD_MODEL = args.load_model
LOAD_EPISODE = args.load_episode

REPLAY_MEMORY = args.replay_memory
NUM_EPISODES = args.num_episodes
BATCH_SIZE = args.batch_size
GAMMA = args.gamma
EPS_START = args.eps_start
EPS_END = args.eps_end
EPS_DECAY = args.eps_decay
TARGET_UPDATE = args.target_update
LR = args.lr

ROOT = args.root

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

	

# --------------------------------
#Lists to debug training
total_loss = [] #List to save the mean values of the episode losses.
episode_loss = [] #List to save every loss value during a single episode.
best_loss = [0, 9999]

total_reward = [] #List to save the total reward gathered each episode.
ex_rate = [] #List to save the epsilon value after each episode.

#Environment - Custom basic environment for kitchen recipes
env = gym.make("gym_basic:basic-v0", display=args.display, disable_env_checker=True)
env.reset() #Set initial state

n_states = env.observation_space.n #Dimensionality of the input of the DQN
n_actions = env.action_space.n #Dimensionality of the output of the DQN 

#Networks and optimizer
policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)
policy_net.apply(init_weights)

target_net.eval()
policy_net.train()

#optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 800, gamma= 0.99)

memory = ReplayMemory(REPLAY_MEMORY)



print_setup(args)

steps_done = 0 

# ----------------------------------

if LOAD_MODEL:	
	path = os.path.join(ROOT, EXPERIMENT_NAME)
	if LOAD_EPISODE: 
		model_name = 'model_' + str(LOAD_EPISODE) + '.pt' #If an episode is specified
		full_path = os.path.join(path, model_name)

	else:
		list_of_files = glob.glob(path+ '/*') 
		full_path = max(list_of_files, key=os.path.getctime) #Get the latest file in directory

	print("-"*30)
	print("\nLoading model from ", full_path)
	checkpoint = torch.load(full_path)
	policy_net.load_state_dict(checkpoint['model_state_dict'])
	target_net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	LOAD_EPISODE = checkpoint['episode']
	total_loss = checkpoint['loss']
	steps_done = checkpoint['steps']
	print("-"*30)

target_net.load_state_dict(policy_net.state_dict())

	


#Action taking
def select_action(state):
	"""
	Function that chooses which action to take in a given state based on the exploration-exploitation paradigm.
	Input:
		state: (tensor) current state of the environment.
	Output:
		action: (tensor) either the greedy action (argmax Q value) from the policy network output, or a random action. 
	"""
	global steps_done
	sample = random.random() #Generate random number [0, 1]
	eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) #Get current exploration rate

	if sample > eps_threshold: #If the random number is higher than the current exploration rate, the policy network determines the best action.
		with torch.no_grad():
			out = policy_net(state)

			return out.max(1)[1].view(1,1)

	else: #If the random number is lower than the current exploration rate, return a random action.
		return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# TRAINING OPTIMIZATION FUNCTION
# ----------------------------------


def optimize_model():
	if len(memory) < BATCH_SIZE:
		#print("Memory capacity is lower than the batch size")
		return
		
	transitions = memory.sample(BATCH_SIZE)	
	batch = Transition(*zip(*transitions))
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
	
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)	

	out = policy_net(state_batch)
	
	state_action_values = policy_net(state_batch).gather(1, action_batch) #Get Q value for current state with the Policy Network. Q(s)
	
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach() #Get Q value for next state with the Target Network. Q(s')
	
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch #Get Q value for current state as R + Q(s')
	
	criterion = nn.SmoothL1Loss() #MSE

	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	#print("LOSS: ", loss)
	episode_loss.append(loss.detach().item())
	
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()
	scheduler.step()	


# ----------------------------------

# TRAINING LOOP
# ----------------------------------

print("\nTraining...")
print("_"*30)
t1 = time.time() #Tik

for i_episode in range(LOAD_EPISODE, NUM_EPISODES+1):
	if(args.display): print("| EPISODE #", i_episode , end='\n')
	else: print("| EPISODE #", i_episode , end='\r')

	state = torch.tensor(env.reset(), dtype=torch.float, device=device).unsqueeze(0)

	episode_loss = []
	done = False
	
	steps_done += 1

	for t in count(): 
		action = select_action(state)
		_, reward, done, _ = env.step(action.item())
		reward = torch.tensor([reward], device=device)

		next_state = torch.tensor(env.state, dtype=torch.float, device=device).unsqueeze(0)

		memory.push(state, action, next_state, reward)

		if not done: 
			state = next_state
		else: 
			next_state = None
			
		optimize_model()
		
		if done: 
			if episode_loss: 
				#print("Count t: ", t)
				total_reward.append(env.get_total_reward()/(t+1))
				total_loss.append(mean(episode_loss))
				ex_rate.append(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))
				
				

			break #Finish episode
	#print(scheduler.optimizer.param_groups[0]['lr']) #Print LR (to check scheduler)
	
	if i_episode % TARGET_UPDATE == 0: #Copy the Policy Network parameters into Target Network
		target_net.load_state_dict(policy_net.state_dict())
	
	if SAVE_MODEL:
		if i_episode % SAVE_EPISODE == 0 and i_episode != 0: 
			path = os.path.join(ROOT, EXPERIMENT_NAME)
			model_name = 'model_' + str(i_episode) + '.pt'
			if not os.path.exists(path): os.makedirs(path)
			print("Saving model at ", os.path.join(path, model_name))
			torch.save({
			'model_state_dict': policy_net.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'episode': i_episode,
			'loss': total_loss,
			'steps': steps_done
				
			}, os.path.join(path, model_name))

			
			if episode_loss:
				if mean(episode_loss) < best_loss[1]:
					best_loss[1] = mean(episode_loss)
					best_loss[0] = i_episode
					with open(ROOT + EXPERIMENT_NAME + '/best_episode.txt', 'w') as f: f.write(str(best_loss[0]))
			
"""				
#Save best episode in a text file
if SAVE_MODEL:
	with open(ROOT + EXPERIMENT_NAME + '/best_episode.txt', 'w') as f:
		f.write(str(best_loss[0]))
"""
t2 = time.time() - t1 #Tak

print("\nTraining completed in {:.1f}".format(t2), "seconds.\n")

plt.subplot(131)
plt.title("Loss")
plt.xlabel("Episode")
plt.ylabel("Average MSE")
plt.plot(total_loss, 'r')

plt.subplot(132)
plt.title("Reward")
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.plot(total_reward, 'b')

plt.subplot(133)
plt.title("Exploration rate")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.plot(ex_rate)

plt.show()

#print("GLOBAL: ", steps_done)
#print("EXPLORATION RATE: ", (EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)))

#print("Memory again? ", memory.show_batch(20))

# ----------------------------------
