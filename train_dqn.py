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



parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true', default=False, help="(bool) Inizializate the model with a pretrained model.")
parser.add_argument('--freeze', type=str, default='False', help="(bool) Inizializate the model with a pretrained moddel freezing the layers but the last one.")
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model is saved. For example: my_first_DQN.")

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
parser.add_argument('--cuda', action='store_true', default=True, help="Use GPU if available.")
args = parser.parse_args()

PRETRAINED = args.pretrained
# print(PRETRAINED)
FREEZE = args.freeze

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
POSITIVE_REWARD = cfg.POSITIVE_REWARD
NO_ACTION_PROBABILITY = cfg.NO_ACTION_PROBABILITY

ROOT = args.root

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    

# --------------------------------
#Lists to debug training
total_loss = [] #List to save the mean values of the episode losses.
episode_loss = [] #List to save every loss value during a single episode.

total_reward = [] #List to save the total reward gathered each episode.
ex_rate = [] #List to save the epsilon value after each episode.


#Environment - Custom basic environment for kitchen recipes
env = gym.make("gym_basic:basic-v0", display=args.display, disable_env_checker=True)


if env.test:
    NUM_EPISODES = len(glob.glob("./video_annotations/Real_data/test/*"))
else:
    NUM_EPISODES = len(glob.glob("./video_annotations/Real_data/train/*"))

env.reset() #Set initial state

n_states = env.observation_space.n #Dimensionality of the input of the DQN
n_actions = env.action_space.n #Dimensionality of the output of the DQN 

#Networks and optimizer
policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)


if PRETRAINED:
    path_model = './Pretrained/model_real_data.pt' #Path al modelo pre-entrenado
    
    print("USING PRETRAINED MODEL---------------")
    
    policy_net.load_state_dict(torch.load(path_model))
    policy_net.to(device)
    
    EPS_START = 0.5
   

#else:
#    policy_net.apply(init_weights) # si no hay pretained


target_net.eval()


optimizer = optim.Adam(policy_net.parameters(), lr=LR) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 20, gamma= 0.99)

memory = ReplayMemory(REPLAY_MEMORY)


# s = sched.scheduler(time.time, time.sleep)
print_setup(args)

steps_done = 0 

# ----------------------------------

if LOAD_MODEL:  
    # pdb.set_trace()
    path = os.path.join(ROOT, EXPERIMENT_NAME)
    if LOAD_EPISODE: 
        model_name = 'model_' + str(LOAD_EPISODE) + '.pt' #If an episode is specified
        full_path = os.path.join(path, model_name)

    else:
        pdb.set_trace()
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
    

#Action taking
def select_action(state, phase):
    """
    Function that chooses which action to take in a given state based on the exploration-exploitation paradigm.
    This action will always be what is referred to as a possible action; the actions possible by the robot are 
    defined by the objects available in its environment.
    Input:
        state: (tensor) current state of the environment.
    Output:
        action: (tensor) either the greedy action (argmax Q value) from the policy network output, or a random action. 
    """
    global steps_done
    sample = random.random() #Generate random number [0, 1]
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) #Get current exploration rate
    
    posible_actions = env.possible_actions_taken_robot()
    index_posible_actions = [i for i, x in enumerate(posible_actions) if x == 1]
    
    if phase == 'val':
        with torch.no_grad():
            out = policy_net(state)
            action = post_processed_possible_actions(out,index_posible_actions)
            # pdb.set_trace()
            return action
    else:
         if sample > eps_threshold: #If the random number is higher than the current exploration rate, the policy network determines the best action.
             with torch.no_grad():
                 out = policy_net(state)
                 action = post_processed_possible_actions(out,index_posible_actions)
                 # pdb.set_trace()
                 return action
         else:
             
             if NO_ACTION_PROBABILITY != 0:
                 index_no_action = index_posible_actions.index(6)
                 
                 weights = [10]*len(index_posible_actions)
                 weights[index_no_action] = cfg.NO_ACTION_PROBABILITY
             
                 # print(index_action)
                 # pdb.set_trace()
                 action = random.choices(index_posible_actions, weights, k=1)[0]
             else:
                 index_action = random.randrange(len(index_posible_actions))
                 action = index_posible_actions[index_action]
    
                
                # pdb.set_trace()
             return torch.tensor([[action]], device=device, dtype=torch.long)


def action_rate(decision_cont,state,phase,prev_decision_rate):
    """
    Function that sets the rate at which the robot makes decisions.

    """
    if cfg.DECISION_RATE == "random":
        
        if phase == 'train':
            if decision_cont == 1:
                decision_rate = random.randrange(10,150)
                prev_decision_rate = decision_rate
            else:
                 decision_rate = prev_decision_rate
        else:
            decision_rate = 20
             
    else:
        decision_rate = cfg.DECISION_RATE
        prev_decision_rate = " "
        
    if decision_cont % decision_rate == 0:  
        action_selected = select_action(state,phase)
        flag_decision = True 
    else:
        action_selected = 6
        flag_decision = False
    # print("RANDOM NUMBER: ",decision_rate)
    # pdb.set_trace()
    return action_selected, flag_decision, prev_decision_rate
    

# TRAINING OPTIMIZATION FUNCTION
# ----------------------------------


def optimize_model(phase):
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
    
   
    state_action_values = policy_net(state_batch).gather(1, action_batch) #Forward pass on the policy network -> Q values for every action -> Keep only Qvalue for the action that we took when exploring (action_batch), for which we have the reward (reward_batch) and the transition (non_final_next_states).
    
    # pdb.set_trace()
    next_state_values = torch.zeros(t_batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach() #Get Q value for next state with the Target Network. Q(s')
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch #Get Q value for current state as R + Q(s')
    
    criterion = nn.SmoothL1Loss() #MSE

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    #print("LOSS: ", loss)
    episode_loss.append(loss.detach().item())
    
    if phase == 'train':
        optimizer.zero_grad()
        loss.backward()
        
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1,1)

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


NUM_EPOCH = cfg.NUM_EPOCH
epoch_loss = []

total_loss_epoch_train =[]

total_time_execution_epoch_train = []
total_reward_epoch_train = []
total_reward_energy_epoch_train = []
total_reward_time_epoch_train = []

total_CA_intime_epoch_train = []
total_CA_late_epoch_train = []
total_IA_intime_epoch_train = []
total_IA_late_epoch_train = []
total_UAC_intime_epoch_train = []
total_UAC_late_epoch_train = []
total_UAI_intime_epoch_train = []
total_UAI_late_epoch_train = []
total_CI_epoch_train = []
total_II_epoch_train = []

total_loss_epoch_val =[]

total_time_execution_epoch_val = []
total_reward_epoch_val = []
total_reward_energy_epoch_val = []
total_reward_time_epoch_val = []

total_CA_intime_epoch_val = []
total_CA_late_epoch_val = []
total_IA_intime_epoch_val = []
total_IA_late_epoch_val = []
total_UAC_intime_epoch_val = []
total_UAC_late_epoch_val = []
total_UAI_intime_epoch_val = []
total_UAI_late_epoch_val = []
total_CI_epoch_val = []
total_II_epoch_val = []

prev_decision_rate = 1
steps_done = 0 

for i_epoch in range (args.load_episode,NUM_EPOCH):

    
    steps_done += 1   
    decision_index_histogram_TRAIN = []
    decision_action_index_histogram_TRAIN = []

    good_reward_TRAIN = []
    good_reward_action_TRAIN = []
    good_reward_noaction_TRAIN = []
    bad_reward_TRAIN = []

    decision_index_histogram_VAL = []
    decision_action_index_histogram_VAL = []

    good_reward_VAL = []
    good_reward_action_VAL = []
    good_reward_noaction_VAL = []
    bad_reward_VAL = []
    # Each epoch has a training and validation phase
    print("| ----------- EPOCH " + str(i_epoch) + " ----------- ")
    #for phase in ['train', 'val']:
    for phase in ['train']:
        total_loss = []
        total_reward = []
        total_reward_energy_ep = []
        total_reward_time_ep = []
        total_reward_error_pred = []
       
        
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
        
        total_times_execution = []
        if phase == 'train':
            policy_net.train()  # Set model to training mode
        else:
            policy_net.eval()   # Set model to evaluate mode

        for i_episode in range(0, NUM_EPISODES):
            if(args.display): print("| EPISODE #", i_episode , end='\n')
            else: print("| EPISODE #", i_episode , end='\r')
        
            state = torch.tensor(env.reset(), dtype=torch.float, device=device).unsqueeze(0)
            # pdb.set_trace()
            episode_loss = []
            done = False
            to_optim = True
            
            decision_state = state
            
            num_optim = 0
            reward_energy_ep = 0
            reward_time_ep = 0
            error_pred_ep = 0
            total_pred_ep = 0
                 
            for t in count(): 

                decision_cont += 1
                action, flag_decision, prev_decision_rate = action_rate(decision_cont, state, phase, prev_decision_rate)
                
                if flag_decision: 
                    action_ = action
                    action = action.item()
                    frame_decision = env.get_frame()
                    action_idx = env.get_action_idx()
                    annotations = env.get_annotations()
                    decision_cont = 0
                    
                    if to_optim:
                        decision_state = state
                        to_optim = False
                        
                
                array_action = [action,flag_decision,phase]
                next_state_, reward, done, optim, flag_pdb, reward_time, reward_energy, execution_times, correct_action, type_threshold, error_pred, total_pred = env.step(array_action)
      
                    
                reward = torch.tensor([reward], device=device)

                next_state = torch.tensor(env.state, dtype=torch.float, device=device).unsqueeze(0)
                reward_energy_ep += reward_energy
                reward_time_ep += reward_time
                error_pred_ep += error_pred
                total_pred_ep += total_pred
                 
                if optim: #Only train if we have taken an action (f==30)                  
                    
                    reward = torch.tensor([reward], device=device)

                    to_optim = True                    

                    memory.push(decision_state, action_, next_state, reward)
                    
                    # Semi -supervised case where the correction is also taken into account to train the DQN
                    if (action != correct_action):
                        memory.push(decision_state, torch.tensor([[correct_action]], device=device), next_state, torch.tensor([0], device=device))
                    
                    optimize_model(phase)
                    num_optim += 1
                    
                    # DECISION FRAME HISTOGRAM
                    if type_threshold != "second":
                        fr_init_prev = annotations['frame_end'][action_idx-1]
                        if action_idx > len(annotations):
                            action_idx = len(annotations)-1
                            
                        if action_idx < 2:
                            fr_init_prev = 0
                        else:
                            fr_init_prev = annotations['frame_end'][action_idx-2]
                       
    
                        fr_init = annotations['frame_init'][action_idx]
                        if type_threshold == "first":
                            fr_init = annotations['frame_end'][action_idx-1]
                        
                        index_frame_decision = 1 - ((fr_init-frame_decision)/(fr_init-fr_init_prev))
                    if phase=='train':
                        decision_index_histogram_TRAIN.append(index_frame_decision)
                        if action != 6:
                            decision_action_index_histogram_TRAIN.append(index_frame_decision)
                        
                        if reward < 0:
                            bad_reward_TRAIN.append(index_frame_decision)
                        else:
                            if action != 6:
                                good_reward_action_TRAIN.append(index_frame_decision)
                            else:
                                good_reward_noaction_TRAIN.append(index_frame_decision)
                    else:
                        decision_index_histogram_VAL.append(index_frame_decision)
                        if action != 6:
                            decision_action_index_histogram_VAL.append(index_frame_decision)
                        
                        if reward < 0:
                            bad_reward_VAL.append(index_frame_decision)
                        else:
                            if action != 6:
                                good_reward_action_VAL.append(index_frame_decision)
                            else:
                                good_reward_noaction_VAL.append(index_frame_decision)

                if not done: 
                    state = next_state
        
                else: 
                    next_state = None 
                
                if done: 
                    if episode_loss: 
  
                        total_reward.append(env.get_total_reward())
                        total_loss.append(mean(episode_loss))
                        total_reward_energy_ep.append(reward_energy_ep)
                        total_reward_time_ep.append(reward_time_ep)
                        total_reward_error_pred.append(error_pred_ep/total_pred_ep)
                       
                        
                        
                    total_times_execution.append(execution_times)
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
                    
                    #memory.show_batch(10)
              

                    break #Finish episode
        
            #print(scheduler.optimizer.param_groups[0]['lr']) #Print LR (to check scheduler)
            
            if i_episode % TARGET_UPDATE == 0: #Copy the Policy Network parameters into Target Network
                target_net.load_state_dict(policy_net.state_dict())
                #scheduler.step()
                

                            
        total_time_video = list(list(zip(*total_times_execution))[0])
        total_time_iteraction = list(list(zip(*total_times_execution))[1])
            
        data = {'video': total_time_video,
        'iteraction': total_time_iteraction,
        'CA_intime': total_CA_intime,
        'CA_late':total_CA_late,
        'IA_intime': total_IA_intime,
        'IA_late':total_IA_late,

        'UAC_intime': total_UAC_intime,
        'UAC_late': total_UAC_late,
        'UAI_intime': total_UAI_intime,
        'UAI_late': total_UAI_late,
        'CI': total_CI,
        'II': total_II,
        
        'prediction error': total_reward_error_pred
        }
        
        if i_epoch == 0: 
            df = pd.DataFrame(data)
        else:
            df_new = pd.DataFrame(data)
            df = pd.concat([df,df_new])
            
        if phase == 'train':
            if i_epoch % 50 == 0:
                # print(PRETRAINED)
               
                if PRETRAINED == True:
                    pre = '_Using_pretained_model'
                else:
                    pre = ''
                    
                if FREEZE == True:
                    freeze = '_Freezing_layers'
                else: 
                    freeze = ''
                if NO_ACTION_PROBABILITY == 0:
                    weight_prob = ' '
                else:
                    weight_prob = '_NO_ACTION_PROBABILITY_EXPLORATION_' + str(cfg.NO_ACTION_PROBABILITY)
                    
                if cfg.DECISION_RATE == 'random':
                    decision_rate_name = '_DECISION_RATE_random_'
                else:
                    decision_rate_name = '_DECISION_RATE_'+str(cfg.DECISION_RATE)
                    

                path = os.path.join(ROOT, EXPERIMENT_NAME + '_' + dt_string + '_EPS_START_'+str(cfg.EPS_START) + decision_rate_name +weight_prob +'_LR_'+str(LR)+ pre + freeze + '_GAMMA_'+str(GAMMA))
                    
                
                # path = os.path.join(ROOT, EXPERIMENT_NAME)
                save_path = os.path.join(path, "Graphics") 
                save_path_hist = os.path.join(save_path, "Histograms") 
                model_name = 'model_' + str(i_epoch) + '.pt'
                if not os.path.exists(path): os.makedirs(path)
                if not os.path.exists(save_path): os.makedirs(save_path)
                if not os.path.exists(save_path_hist): os.makedirs(save_path_hist)
 
         
                print("Saving model at ", os.path.join(path, model_name))
                torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': i_epoch,
                'loss': total_loss,
                'steps': steps_done            
                }, os.path.join(path, model_name))


            ex_rate.append(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))

            total_loss_epoch_train.append(sum(total_loss))
            total_reward_epoch_train.append(sum(total_reward))
            total_reward_energy_epoch_train.append(sum(total_reward_energy_ep))
            total_reward_time_epoch_train.append(sum(total_reward_time_ep))

            total_time_execution_epoch_train.append(sum(total_time_iteraction))

            total_CA_intime_epoch_train.append(sum(total_CA_intime))
            total_CA_late_epoch_train.append(sum(total_CA_late))
            total_IA_intime_epoch_train.append(sum(total_IA_intime))
            total_IA_late_epoch_train.append(sum(total_IA_late))
            total_UAC_intime_epoch_train.append(sum(total_UAC_intime))
            total_UAC_late_epoch_train.append(sum(total_UAC_late))
            total_UAI_intime_epoch_train.append(sum(total_UAI_intime))
            total_UAI_late_epoch_train.append(sum(total_UAI_late))
            total_CI_epoch_train.append(sum(total_CI))
            total_II_epoch_train.append(sum(total_II))
            total_results_train = [total_CA_intime_epoch_train,total_CA_late_epoch_train,total_IA_intime_epoch_train,total_IA_late_epoch_train,total_UAC_intime_epoch_train,total_UAC_late_epoch_train,total_UAI_intime_epoch_train,total_UAI_late_epoch_train,total_CI_epoch_train,total_II_epoch_train]
           
            
            if i_epoch % 200 == 0: plot_each_epoch(i_epoch, phase,save_path, total_results_train,total_loss_epoch_train,total_reward_epoch_train,total_time_video,total_time_execution_epoch_train,total_reward_energy_epoch_train,total_reward_time_epoch_train,ex_rate)
            
            
            
            # if i_epoch == NUM_EPOCH-1:
            data_train = {
            'CA_intime': total_CA_intime_epoch_train,
            'CA_late':total_CA_late_epoch_train,
            'IA_intime': total_IA_intime_epoch_train,
            'IA_late':total_IA_late_epoch_train,
            # 'UA_intime': total_UA_intime_epoch_train,
            # 'UA_late': total_UA_late_epoch_train,
            'UAC_intime': total_UAC_intime_epoch_train,
            'UAC_late': total_UAC_late_epoch_train,
            'UAI_intime': total_UAI_intime_epoch_train,
            'UAI_late': total_UAI_late_epoch_train,
            'CI': total_CI_epoch_train,
            'II': total_II_epoch_train,
            'prediction error': np.mean(total_reward_error_pred)
            }
        
            # if i_epoch == 0: 
            #     df_train = pd.DataFrame(data_train)
            # else:
            #     df_new_train = pd.DataFrame(data_train)
            # df_train = pd.concat([df_train,df_new_train])
            df_train = pd.DataFrame(data_train)
            df_train.to_csv(save_path+'/data_train.csv')
            
            # HISTOGRAMS
            """
            fig1 = plt.figure(figsize=(12, 7))
            plt.hist(decision_index_histogram_TRAIN, bins = 100, edgecolor="black")
            plt.title("DECISION FRAME (ALL ACTIONS)")
            fig1.savefig(save_path_hist+'/train_hist_epoch_'+str(i_epoch)+'.jpg')
            # plt.show()
            plt.close()

            fig1 = plt.figure(figsize=(12, 7))
            plt.hist(good_reward_action_TRAIN, bins = 100, edgecolor="black")
            plt.title("DECISION FRAME (ONLY ACTIONS, GOOD REWARD)")
            fig1.savefig(save_path_hist+'/train_GOOD_action_hist_epoch_'+str(i_epoch)+'.jpg')
            plt.close()


            fig1 = plt.figure(figsize=(12, 7))
            plt.hist(bad_reward_TRAIN, bins = 100, edgecolor="black")
            plt.title("DECISION FRAME (BAD REWARD)")
            fig1.savefig(save_path_hist+'/train_BAD_hist_epoch_'+str(i_epoch)+'.jpg')
            plt.close()

            # pdb.set_trace()
            fig1 = plt.figure(figsize=(12, 7))
            plt.hist(decision_action_index_histogram_TRAIN, bins = 100, edgecolor="black")
            plt.title("DECISION FRAME (ALL ACTIONS BUT NO ACTION(18))")
            fig1.savefig(save_path_hist+'/train_hist_action_epoch_'+str(i_epoch)+'.jpg')
            plt.close()
            # plt.show()
            """
            
            #print("\n(train) PREDICTION ERROR: %.2f%%" %(np.mean(total_reward_error_pred)*100))
        elif phase=='val':
            # print(len(total_loss))
            # pdb.set_trace()
            total_loss_epoch_val.append(sum(total_loss))
            total_reward_epoch_val.append(sum(total_reward))
            total_reward_energy_epoch_val.append(sum(total_reward_energy_ep))
            total_reward_time_epoch_val.append(sum(total_reward_time_ep))

            total_time_execution_epoch_val.append(sum(total_time_iteraction))

            total_CA_intime_epoch_val.append(sum(total_CA_intime))
            total_CA_late_epoch_val.append(sum(total_CA_late))
            total_IA_intime_epoch_val.append(sum(total_IA_intime))
            total_IA_late_epoch_val.append(sum(total_IA_late))
            total_UAC_intime_epoch_val.append(sum(total_UAC_intime))
            total_UAC_late_epoch_val.append(sum(total_UAC_late))
            total_UAI_intime_epoch_val.append(sum(total_UAI_intime))
            total_UAI_late_epoch_val.append(sum(total_UAI_late))
            total_CI_epoch_val.append(sum(total_CI))
            total_II_epoch_val.append(sum(total_II))
            total_results = [total_CA_intime_epoch_val,total_CA_late_epoch_val,total_IA_intime_epoch_val,total_IA_late_epoch_val,total_UAC_intime_epoch_val,total_UAC_late_epoch_val,total_UAI_intime_epoch_val,total_UAI_late_epoch_val,total_CI_epoch_val,total_II_epoch_val]
            
            
            plot_each_epoch(i_epoch, phase,save_path, total_results,total_loss_epoch_val,total_reward_epoch_val,total_time_video,total_time_execution_epoch_val,total_reward_energy_epoch_val,total_reward_time_epoch_val)
            
            
            plot_each_epoch_together(i_epoch,save_path, total_results_train,total_loss_epoch_train,total_reward_epoch_train,total_time_video,total_time_execution_epoch_train,total_reward_energy_epoch_train,total_reward_time_epoch_train,ex_rate,total_results,total_loss_epoch_val,total_reward_epoch_val,total_time_execution_epoch_val,total_reward_energy_epoch_val,total_reward_time_epoch_val)
            
            # if i_epoch == NUM_EPOCH-1:
            data_val = {
            'CA_intime': total_CA_intime_epoch_val,
            'CA_late':total_CA_late_epoch_val,
            'IA_intime': total_IA_intime_epoch_val,
            'IA_late':total_IA_late_epoch_val,
            # 'UA_intime': total_UA_intime_epoch_val,
            # 'UA_late': total_UA_late_epoch_val,
            'UAC_intime': total_UAC_intime_epoch_val,
            'UAC_late': total_UAC_late_epoch_val,
            'UAI_intime': total_UAI_intime_epoch_val,
            'UAI_late': total_UAI_late_epoch_val,
            'CI': total_CI_epoch_val,
            'II': total_II_epoch_val,
            'prediction error': np.mean(total_reward_error_pred)
            }
            
    
            df_val = pd.DataFrame(data_val)
            df_val.to_csv(save_path+'/data_val.csv')
            
            fig1 = plt.figure(figsize=(12, 7))
            plt.hist(decision_index_histogram_VAL, bins = 100, edgecolor="black")
            plt.title("DECISION FRAME (ALL ACTIONS)")
            fig1.savefig(save_path_hist+'/val_hist_epoch_'+str(i_epoch)+'.jpg')
            # plt.show()
            plt.close()

            fig1 = plt.figure(figsize=(12, 7))
            plt.hist(decision_action_index_histogram_VAL, bins = 100, edgecolor="black")
            plt.title("DECISION FRAME (ALL ACTIONS BUT NO ACTION(6))")
            fig1.savefig(save_path_hist+'/val_hist_action_epoch_'+str(i_epoch)+'.jpg')
            plt.close()

            fig1 = plt.figure(figsize=(12, 7))
            plt.hist(good_reward_action_VAL, bins = 100, edgecolor="black")
            plt.title("DECISION FRAME (ONLY ACTIONS, GOOD REWARD)")
            fig1.savefig(save_path_hist+'/val_GOOD_action_hist_epoch_'+str(i_epoch)+'.jpg')
            plt.close()


            fig1 = plt.figure(figsize=(12, 7))
            plt.hist(bad_reward_VAL, bins = 100, edgecolor="black")
            plt.title("DECISION FRAME (BAD REWARD)")
            fig1.savefig(save_path_hist+'/val_BAD_hist_epoch_'+str(i_epoch)+'.jpg')
            plt.close()
            print("(val) PREDICTION ERROR: %.2f%%\n" %(np.mean(total_reward_error_pred)*100))
            

t2 = time.time() - t1 #Tak


print("\nTraining completed in {:.1f}".format(t2), "seconds.\n")


