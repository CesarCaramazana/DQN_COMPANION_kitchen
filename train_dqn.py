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
import pandas as pd
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
from aux import *
import argparse
import pdb
from datetime import datetime
# import sched, time





parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model is saved. For example: my_first_DQN.")
parser.add_argument('--save_model', action='store_true', default=False, help="Save a checkpoint in the EXPERIMENT_NAME folder.")
parser.add_argument('--load_model', action='store_true', help="Load a checkpoint from the EXPERIMENT_NAME folder. If no episode is specified (LOAD_EPISODE), it loads the latest created file.")
parser.add_argument('--load_episode', type=int, default=0, help="(int) Number of episode to load from the EXPERIMENT_NAME folder, as the sufix added to the checkpoints when the save files are created. For example: 500, which will load 'model_500.pt'.")
parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help="(int) Batch size for the training of the network. For example: 64.")
parser.add_argument('--num_episodes', type=int, default=cfg.NUM_EPISODES, help="(int) Number of episodes per epoch.")
parser.add_argument('--num_epochs', type=int, default=cfg.NUM_EPOCH, help="(int) Number of training epochs.")
parser.add_argument('--lr', type=float, default=cfg.LR, help="(float) Learning rate. For example: 1e-3.")
parser.add_argument('--replay_memory', type=int, default=cfg.REPLAY_MEMORY, help="(int) Size of the Experience Replay memory. For example: 1000.")
parser.add_argument('--gamma', type=float, default=cfg.GAMMA, help="(float) Discount rate of future rewards. For example: 0.99.")
parser.add_argument('--eps_start', type=float, default=cfg.EPS_START, help="(float) Initial exploration rate. For example: 0.99.")
parser.add_argument('--eps_end', type=float, default=cfg.EPS_END, help="(float) Terminal exploration rate. For example: 0.05.")
parser.add_argument('--eps_decay', type=int, default=cfg.EPS_DECAY, help="(int) Decay factor of the exploration rate. Episode where the epsilon has decay to 0.367 of the initial value. For example: num_episodes/2.")
parser.add_argument('--target_update', type=int, default=cfg.TARGET_UPDATE, help="(int) Frequency of the update of the Target Network, in number of episodes. For example: 10.")
parser.add_argument('--root', type=str, default=cfg.ROOT, help="(str) Name of the root folder for the saving of checkpoints. Parent folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/")
parser.add_argument('--display', action='store_true', default=False, help="Display environment info as [Current state, action taken, transitioned state, immediate reward, total reward].")
parser.add_argument('--cuda', action='store_true', default=True, help="Use GPU if available.")
args = parser.parse_args()


SAVE_MODEL = args.save_model
SAVE_EPISODE = cfg.SAVE_EPISODE
EXPERIMENT_NAME = args.experiment_name
LOAD_MODEL = args.load_model
LOAD_EPISODE = args.load_episode

NUM_EPOCH = args.num_epochs
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


# s = sched.scheduler(time.time, time.sleep)
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
    LOAD_EPISODE = checkpoint['epoch']
    # LOAD_EPISODE = checkpoint['episode']
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


def action_rate(decision_cont,state):
   
    if decision_cont % cfg.DECISION_RATE == 0: 
        action_selected = select_action(state)
        flag_decision = True 
    else:
        action_selected = 18
        flag_decision = False
    
    return action_selected, flag_decision

    

# TRAINING OPTIMIZATION FUNCTION
# ----------------------------------


def optimize_model():
    # print(len(memory))
    
        #print("Memory capacity is lower than the batch size")
    t_batch_size = min(len(memory),BATCH_SIZE)

        
    transitions = memory.sample(t_batch_size)    
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)    
    
 
    
    out = policy_net(state_batch)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch) #Forward pass on the policy network -> Q values for every action -> Keep only Qvalue for the action that we took when exploring (action_batch), for which we have the reward (reward_batch) and the transition (non_final_next_states).
    
    next_state_values = torch.zeros(t_batch_size, device=device)
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
    # scheduler.step()    


# ----------------------------------

# TRAINING LOOP
# ----------------------------------

print("\nTraining...")
print("_"*30)




now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")


t1 = time.time() #Tik

decision_cont = 0

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
total_times_execution = []
total_reward_energy_ep = []
total_reward_time_ep = []

total_time_execution_epoch = []
total_reward_epoch = []
total_reward_energy_epoch = []
total_reward_time_epoch = []


epoch_loss = []

total_loss_epoch =[]
total_reward_vs_optim = []
total_reward_epoch_vs_optim = []


total_CA_intime_epoch = []
total_CA_late_epoch = []
total_IA_intime_epoch = []
total_IA_late_epoch = []
total_UA_intime_epoch = []
total_UA_late_epoch = []
total_CI_epoch = []
total_II_epoch = []

steps_done = 0 #esto antes no estaba

for i_epoch in range (0,NUM_EPOCH):
    total_loss = []
    total_reward_vs_optim = []
    total_reward = []
    total_reward_energy_ep = []
    total_reward_time_ep = []
    
    total_CA_intime = []
    total_CA_late = []
    total_IA_intime = []
    total_IA_late = []
    total_UA_intime = []
    total_UA_late = []
    total_CI = []
    total_II = []
    
    total_times_execution = []
    
    steps_done += 1
    
    print("| ----------- EPOCH " + str(i_epoch) + " ----------- ")
    for i_episode in range(LOAD_EPISODE, NUM_EPISODES):
        if(args.display): print("| EPISODE #", i_episode , end='\n')
        else: print("| EPISODE #", i_episode , end='\r')
    
        state = torch.tensor(env.reset(), dtype=torch.float, device=device).unsqueeze(0)
    
        episode_loss = []
        done = False
        
        
        num_optim = 0
            
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
            
            reward = torch.tensor([reward], device=device)
            prev_state = torch.tensor([prev_state], dtype=torch.float,device=device)
            next_state = torch.tensor([next_state], dtype=torch.float,device=device)
            # next_state = torch.tensor(env.state, dtype=torch.float, device=device).unsqueeze(0)
            reward_energy_ep += reward_energy
            reward_time_ep += reward_time
                            
            if optim: #Only train if we have taken an action (f==30)
                #print("OPTIMIZE NOW")
                # print("\n------------- TRAIN -------------") 
                # print("State: ", cfg.ATOMIC_ACTIONS_MEANINGS[undo_one_hot(prev_state[0][:33])])
                # print("Next State: ",cfg.ATOMIC_ACTIONS_MEANINGS[undo_one_hot(next_state[0][:33])])
                # print("Acción Robot: ",cfg.ROBOT_ACTIONS_MEANINGS[action])
                # print("Reward: ",reward)
                
                # if flag_pdb: 
                #     pdb.set_trace()
    
                memory.push(prev_state, action_, next_state, reward)
                optimize_model()
                num_optim += 1
    
    
            if not done: 
                state = next_state
    
            else: 
                next_state = None
                
            
            if done: 
                if episode_loss: 
                    #print("Count t: ", t)
                    #total_reward.append(env.get_total_reward()/(t+1))
    
                    # print("Memory:  ", memory.show_batch(20))
                    # print(env.get_total_reward())
                   
                    total_reward.append(env.get_total_reward())
                    # total_reward_vs_optim.append(env.get_total_reward()/num_optim)
                    
                    total_loss.append(mean(episode_loss))
                    # running_loss += mean(episode_loss)
                    # running_reward += env.get_total_reward()
                
                    total_reward_energy_ep.append(reward_energy_ep)
                    total_reward_time_ep.append(reward_time_ep)
                    
                total_times_execution.append(execution_times)
                total_CA_intime.append(env.CA_intime)
                total_CA_late.append(env.CA_late)
                total_IA_intime.append(env.IA_intime)
                total_IA_late.append(env.IA_late)
                total_UA_intime.append(env.UA_intime)
                total_UA_late.append(env.UA_late)
                total_CI.append(env.CI)
                total_II.append(env.II)
                # pdb.set_trace()

                break #Finish episode
    
        #print(scheduler.optimizer.param_groups[0]['lr']) #Print LR (to check scheduler)
        
        if i_episode % TARGET_UPDATE == 0: #Copy the Policy Network parameters into Target Network
            target_net.load_state_dict(policy_net.state_dict())
        
        # if SAVE_MODEL:
        #     if i_episode % SAVE_EPISODE == 0 and i_episode != 0: 
    if i_epoch % (cfg.NUM_EPOCH*0.01) == 0:

        path = os.path.join(ROOT, EXPERIMENT_NAME + '_' + dt_string + '_DECISION_RATE_' + str(cfg.DECISION_RATE))
        save_path = os.path.join(path, "Graphics") 
        model_name = 'model_' + str(i_epoch) + '.pt'
        if not os.path.exists(path): os.makedirs(path)
        if not os.path.exists(save_path): os.makedirs(save_path)

        print("Saving model at ", os.path.join(path, model_name))
        torch.save({
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': i_epoch,
        'loss': total_loss,
        'steps': steps_done            
        }, os.path.join(path, model_name))

        
        if episode_loss:
            if mean(episode_loss) < best_loss[1]:
                best_loss[1] = mean(episode_loss)
                best_loss[0] = i_epoch
                with open(ROOT + EXPERIMENT_NAME  + '_' + dt_string + '_DECISION_RATE_' + str(cfg.DECISION_RATE) + '/best_episode.txt', 'w') as f: f.write(str(best_loss[0]))

    # total_loss_epoch.append()
    # epoch_loss.append(running_loss/cfg.NUM_EPOCH)
    # total_reward_epoch_v2.append(running_reward/cfg.NUM_EPOCH)
    ex_rate.append(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))
    
    # total_reward_epoch_vs_optim.append(sum(total_reward_vs_optim))
    
    total_loss_epoch.append(sum(total_loss))
    total_reward_epoch.append(sum(total_reward))
    total_reward_energy_epoch.append(sum(total_reward_energy_ep))
    total_reward_time_epoch.append(sum(total_reward_time_ep))
    
    total_time_video = list(list(zip(*total_times_execution))[0])
    total_time_iteraction = list(list(zip(*total_times_execution))[1])
    
    # print(total_time_video)
    # print(len(total_time_video))
    total_time_execution_epoch.append(sum(total_time_iteraction))
    
    data = {'video': total_time_video,
	'iteraction': total_time_iteraction,
    'CA_intime': total_CA_intime,
    'CA_late':total_CA_late,
    'IA_intime': total_IA_intime,
    'IA_late':total_IA_late,
    'UA_intime': total_UA_intime,
    'UA_late': total_UA_late,
    'CI': total_CI,
    'II': total_II
    }
    
    if i_epoch == 0: 
        df = pd.DataFrame(data)
    else:
        df_new = pd.DataFrame(data)
        df = pd.concat([df,df_new])
        
    
    total_CA_intime_epoch.append(sum(total_CA_intime))
    total_CA_late_epoch.append(sum(total_CA_late))
    total_IA_intime_epoch.append(sum(total_IA_intime))
    total_IA_late_epoch.append(sum(total_IA_late))
    total_UA_intime_epoch.append(sum(total_UA_intime))
    total_UA_late_epoch.append(sum(total_UA_late))
    total_CI_epoch.append(sum(total_CI))
    total_II_epoch.append(sum(total_II))
    
    fig1 = plt.figure(figsize=(20, 6))
    plt.subplot(131)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE")
    plt.plot(total_loss_epoch, 'r')

    plt.subplot(132)
    plt.title("Reward")
    plt.xlabel("Epoch")
    plt.ylabel("Episode reward")
    plt.plot(total_reward_epoch)

    plt.subplot(133)
    plt.title("Exploration rate")
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.plot(ex_rate)
    fig1.savefig(save_path+'/train_results_epoch'+dt_string+'.jpg')
    
    
    fig3 = plt.figure(figsize=(20, 12))
    plt.subplot(241)
    plt.title("Correct actions (in time)")
    plt.plot(total_CA_intime_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Amount action")
    plt.subplot(242)
    plt.title("Correct actions (late)")
    plt.plot(total_CA_late_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Amount action")
    plt.subplot(243)
    plt.title("Incorrect actions (in time)")
    plt.plot(total_IA_intime_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Amount action")
    plt.subplot(244)
    plt.title("Incorrect actions (late)")
    plt.plot(total_IA_late_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Amount action")
    plt.subplot(245)
    plt.title("Unnecessary actions (in time)")
    plt.plot(total_UA_intime_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Amount action")
    plt.subplot(246)
    plt.title("Unnecessary actions (late)")
    plt.plot(total_UA_late_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Amount action")
    plt.subplot(247)
    plt.title("Correct inactions")
    plt.plot(total_CI_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Amount action")
    plt.subplot(248)
    plt.title("Incorrect inactions")
    plt.plot(total_II_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Amount action")
    # plt.show()
    fig3.savefig(save_path+'/train_detailed_results_epoch_'+dt_string+'.jpg')
    
    total_time_video_epoch = [sum(total_time_video)]*len(total_time_execution_epoch)
    fig1 = plt.figure(figsize=(15, 6))
    plt.plot(total_time_execution_epoch)
    plt.plot(total_time_video_epoch)
    plt.legend(["Interaction","Video"])
    plt.xlabel("Epoch")
    plt.ylabel("Frames")
    plt.title("Time")
    # plt.show()
    fig1.savefig(save_path+'/train_time_execution_'+dt_string+'.jpg')

    fig1 = plt.figure(figsize=(15, 6))
    plt.plot(total_reward_energy_epoch)
    plt.legend(["Energy reward"])
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Reward")
    # plt.show()
    fig1.savefig(save_path+'/train_energy_reward_'+dt_string+'.jpg')

    fig1 = plt.figure(figsize=(15, 6))
    plt.plot(total_reward_time_epoch)
    plt.legend(["Time reward"])
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Reward")
    # plt.show()
    fig1.savefig(save_path+'/train_time_reward_'+dt_string+'.jpg')

    # pdb.set_trace()
    
    
"""                
#Save best episode in a text file
if SAVE_MODEL:
    with open(ROOT + EXPERIMENT_NAME + '/best_episode.txt', 'w') as f:
        f.write(str(best_loss[0]))
"""
t2 = time.time() - t1 #Tak


print("\nTraining completed in {:.1f}".format(t2), "seconds.\n")

# pdb.set_trace()



fig1 = plt.figure(figsize=(20, 6))
plt.subplot(131)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Average MSE")
plt.plot(total_loss_epoch, 'r')

plt.subplot(132)
plt.title("Reward")
plt.xlabel("Epoch")
plt.ylabel("Episode reward")
plt.plot(total_reward_epoch)

plt.subplot(133)
plt.title("Exploration rate")
plt.xlabel("Epoch")
plt.ylabel("Epsilon")
plt.plot(ex_rate)
fig1.savefig(save_path+'/train_results_epoch'+dt_string+'.jpg')


fig3 = plt.figure(figsize=(20, 12))
plt.subplot(241)
plt.title("Correct actions (in time)")
plt.plot(total_CA_intime_epoch)
plt.xlabel("Epoch")
plt.ylabel("Amount action")
plt.subplot(242)
plt.title("Correct actions (late)")
plt.plot(total_CA_late_epoch)
plt.xlabel("Epoch")
plt.ylabel("Amount action")
plt.subplot(243)
plt.title("Incorrect actions (in time)")
plt.plot(total_IA_intime_epoch)
plt.xlabel("Epoch")
plt.ylabel("Amount action")
plt.subplot(244)
plt.title("Incorrect actions (late)")
plt.plot(total_IA_late_epoch)
plt.xlabel("Epoch")
plt.ylabel("Amount action")
plt.subplot(245)
plt.title("Unnecessary actions (in time)")
plt.plot(total_UA_intime_epoch)
plt.xlabel("Epoch")
plt.ylabel("Amount action")
plt.subplot(246)
plt.title("Unnecessary actions (late)")
plt.plot(total_UA_late_epoch)
plt.xlabel("Epoch")
plt.ylabel("Amount action")
plt.subplot(247)
plt.title("Correct inactions")
plt.plot(total_CI_epoch)
plt.xlabel("Epoch")
plt.ylabel("Amount action")
plt.subplot(248)
plt.title("Incorrect inactions")
plt.plot(total_II_epoch)
plt.xlabel("Epoch")
plt.ylabel("Amount action")
plt.show()

fig3.savefig(save_path+'/train_detailed_results_epoch'+dt_string+'.jpg')


total_results = [total_CA_intime_epoch,total_CA_late_epoch,total_IA_intime_epoch,total_IA_late_epoch,total_UA_intime_epoch,total_UA_late_epoch,total_CI_epoch,total_II_epoch]

n = int(cfg.NUM_EPOCH*0.1)
plot_detailed_results(n, total_results, save_path, 'TRAIN')

n = int(cfg.NUM_EPOCH*0.05)
plot_detailed_results(n, total_results, save_path, 'TRAIN')

n = int(cfg.NUM_EPOCH*0.2)
plot_detailed_results(n, total_results, save_path, 'TRAIN')



keys_video = df['video'][0:cfg.NUM_EPISODES]

iteraction_x = []
video_x = []

"""
for idx_key,val_key in enumerate(keys_video):
    iteraction_x.append(list(df[df['video']==val_key]['iteraction']))
    video_x.append([val_key]*cfg.NUM_EPOCH)
        
for idx_plt in range(0,cfg.NUM_EPISODES):
    if idx_plt == 0:
        cont = idx_plt + 1
        fig3 = plt.figure(figsize=(28, 12))
        plt.suptitle("Time execution", fontsize=20)
    else: 
        cont += 1
    
    plt.subplot(240 + cont)
    plt.title("Time Video "+str(idx_plt), fontsize=14)
    plt.plot(iteraction_x[idx_plt])
    plt.plot(video_x[idx_plt])
    plt.legend(["Iteraction", "Video"])   
    plt.xlabel("Epoch")
    plt.ylabel("Frames")
    
    if idx_plt == cfg.NUM_EPISODES - 1:
        fig3.savefig(save_path+'/train_time_results_per_video'+str(idx_plt)+'.jpg')

        plt.show()
    if cont==8:
        fig3.savefig(save_path+'/train_time_results_per_video'+str(idx_plt)+'.jpg')

        plt.show()
        fig3 = plt.figure(figsize=(28, 12))
        plt.suptitle("Time execution", fontsize=20)
        
        cont = 0
   

fig1 = plt.figure(figsize=(15, 6))

total_time_video_epoch = [sum(total_time_video)]*len(total_time_execution_epoch)
plt.plot(total_time_execution_epoch)
plt.plot(total_time_video_epoch)

plt.legend(["Iteraction","Video"])
plt.xlabel("Epoch")
plt.ylabel("Frames")
plt.title("Time")
plt.show()

"""
fig1 = plt.figure(figsize=(15, 6))


plt.plot(total_reward_energy_epoch)

plt.legend(["Energy reward"])
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.title("Reward")
plt.show()

fig1.savefig(save_path+'/train_energy_reward_'+dt_string+'.jpg')

fig1 = plt.figure(figsize=(15, 6))


plt.plot(total_reward_time_epoch)

plt.legend(["Time reward"])
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.title("Reward")
plt.show()

fig1.savefig(save_path+'/train_time_reward_'+dt_string+'.jpg')



