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

from DQN import *
import config as cfg
from aux import *
import argparse
import pdb
from datetime import datetime


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true', default=False, help="(bool) Inizializate the model with a pretrained model.")
parser.add_argument('--freeze', type=str, default='False', help="(bool) Inizializate the model with a pretrained moddel freezing the layers but the last one.")
parser.add_argument('--experiment_name', type=str, default=cfg.EXPERIMENT_NAME, help="(str) Name of the experiment. Used to name the folder where the model is saved. For example: my_first_DQN.")

parser.add_argument('--load_model', action='store_true', help="Load a checkpoint from the EXPERIMENT_NAME folder. If no episode is specified (LOAD_EPISODE), it loads the latest created file.")
parser.add_argument('--load_episode', type=int, default=0, help="(int) Number of episode to load from the EXPERIMENT_NAME folder, as the sufix added to the checkpoints when the save files are created. For example: 500, which will load 'model_500.pt'.")
parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help="(int) Batch size for the training of the network. For example: 54.")

parser.add_argument('--lr', type=float, default=cfg.LR, help="(float) Learning rate. For example: 1e-3.")
parser.add_argument('--replay_memory', type=int, default=cfg.REPLAY_MEMORY, help="(int) Size of the Experience Replay memory. For example: 1000.")
parser.add_argument('--gamma', type=float, default=cfg.GAMMA, help="(float) Discount rate of future rewards. For example: 0.1.")
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
    NUM_EPISODES = len(glob.glob("./video_annotations/5folds/"+cfg.TEST_FOLD+"/test/*"))
else:
    NUM_EPISODES = len(glob.glob("./video_annotations/5folds/"+cfg.TEST_FOLD+"/train/*"))

env.reset() #Set initial state

n_states = env.observation_space.n #Dimensionality of the input of the DQN
n_actions = env.action_space.n #Dimensionality of the output of the DQN 

print("Dimensionality of observation space: ", n_states)

#Networks and optimizer
if cfg.Z_hidden_state:
    if cfg.LATE_FUSION:
        if cfg.TEMPORAL_CONTEXT:
            policy_net = DQN_LateFusion(n_states, n_actions).to(device)
            target_net = DQN_LateFusion(n_states, n_actions).to(device)
        else:
            policy_net = DQN_LateFusion_noTCtx(n_states, n_actions).to(device)
            target_net = DQN_LateFusion_noTCtx(n_states, n_actions).to(device)
    else:
        policy_net = DQN_Z(n_states, n_actions).to(device)
        target_net = DQN_Z(n_states, n_actions).to(device)
else:
    policy_net = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)

if PRETRAINED:
    # path_model = './Pretrained/model_real_data.pt' #Path al modelo pre-entrenado
    if cfg.Z_hidden_state:
        path_model = './Pretrained/model_with_Z.pt'
        print("With Z variable")
    else:
        path_model = './Pretrained/model_without_Z.pt'
        print("Without Z variable")
  
    print("\nUSING PRETRAINED MODEL---------------")
    
    policy_net.load_state_dict(torch.load(path_model))
    policy_net.to(device)
    
    EPS_START = 0.5
   
else:
    #Weight initialization
    policy_net.apply(init_weights) # si no hay pretained

#Regularization
L2 = 1e-4   
optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=L2) 

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 100, gamma= 0.5)

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
    total_loss = checkpoint['loss']
    steps_done = checkpoint['steps']
    print("-"*30)


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


#To count the number of trainable parameters
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print("Number of trainable parameters: ", count_parameters(policy_net))

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
    
    # print("train_dqn.py | Select action, state dim: ", state.shape, " | Phase: ", phase)
    
    policy_net.eval()
    
    if phase == 'val':
        
        with torch.no_grad():
            out = policy_net(state)
            action = post_processed_possible_actions(out,index_posible_actions)
            # pdb.set_trace()
            # return action
    else:
         if sample > eps_threshold: #If the random number is higher than the current exploration rate, the policy network determines the best action.
             with torch.no_grad():
                 out = policy_net(state)
                 action = post_processed_possible_actions(out,index_posible_actions)
                 # pdb.set_trace()
                 #print("Action: ", action)
                 # return action
         else:
             
             if NO_ACTION_PROBABILITY != 0:
                 index_no_action = index_posible_actions.index(5)
                 
                 weights = [10]*len(index_posible_actions)
                 weights[index_no_action] = cfg.NO_ACTION_PROBABILITY
             
                 # print(index_action)
                 # pdb.set_trace()
                 action = random.choices(index_posible_actions, weights, k=1)[0]
                 
                 #print("Action: ", action)
             else:
                 index_action = random.randrange(len(index_posible_actions))
                 action = index_posible_actions[index_action]
                 #print("Action: ", action)
    
                
                # pdb.set_trace()
                
    # policy_net.train()
    if cfg.REACTIVE == True: action = 5 #Forcefully set reactive robot     
       
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
            decision_rate = 100
             
    else:
        decision_rate = cfg.DECISION_RATE
        prev_decision_rate = " "
        
    if decision_cont % decision_rate == 0:  
        action_selected = select_action(state,phase)
        flag_decision = True 
    else:
        action_selected = 5
        flag_decision = False
        
    # print("RANDOM NUMBER: ",decision_rate)
    # pdb.set_trace()
    return action_selected, flag_decision, prev_decision_rate
    

# TRAINING OPTIMIZATION FUNCTION
# ----------------------------------


def optimize_model(phase):
    """
    Executes an iteration of the optimization algorithm.    

    Parameters
    ----------
    phase: (str) phase of the training. If 'train', backpropagate gradients, else, just save the loss value.

    Returns
    -------
    None.

    """
    
    t_batch_size = min(len(memory),BATCH_SIZE)
    
    #print("len memory: ", len(memory))
    #print("batch: ", BATCH_SIZE)
    
    policy_net.train()
    
    
    if len(memory) < BATCH_SIZE:
        return           
        
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
    
    criterion = nn.SmoothL1Loss() #A mixture of MSE/L1

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # print("LOSS: ", loss)
    
    if phase == 'train':
        optimizer.zero_grad()
        loss.backward()
        
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1,1)

        optimizer.step()
    
    episode_loss.append(loss.detach().item())
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

total_UA_related_epoch_train = []
total_UA_unrelated_epoch_train = []
total_UA_related_epoch_val = []
total_UA_unrelated_epoch_val = []


#RRRRRRR
total_idle_epoch_train = []
total_idle_epoch_val = []

prev_decision_rate = 1
steps_done = 0 


# Get minimum and maximum time from dataset

video_max_times = []
video_min_times = []


root = "./video_annotations/5folds/"+cfg.TEST_FOLD+"/train/*" #!
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
maximum_time = sum(video_max_times) #* cfg.BETA 




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
    
    #Call the LR scheduler 1 time per epoch
    scheduler.step()
    NO_ACTION_PROBABILITY = NO_ACTION_PROBABILITY * 0.99
    
    # Each epoch has a training and validation phase
    print("| ----------- EPOCH " + str(i_epoch) + " ----------- ")
    for phase in ['train', 'val']:
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
        total_UA_related = []
        total_UA_unrelated = []
        total_interaction_time_epoch = []
        
        #RRRRRRRRRRr
        total_idle = []
        
        videos_mejorables = []
        total_times_execution = []
        if phase == 'train':
            policy_net.train()  # Set model to training mode
            target_net.eval()
        else:
            policy_net.eval()   # Set model to evaluate mode
            target_net.eval()

        for i_episode in range(0, NUM_EPISODES):
            if(args.display): print("| EPISODE #", i_episode , end='\n')
            else: print("| EPISODE #", i_episode , end='\r')
        
            state = torch.tensor(env.reset(), dtype=torch.float, device=device).unsqueeze(0)
            # pdb.set_trace()
            episode_loss = []
            done = False
            to_optim = True
            decision_cont = 0
            
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
                next_state_, reward, done, optim, flag_pdb, reward_time, reward_energy, hri_time, correct_action, type_threshold, error_pred, total_pred = env.step(array_action)
        
                reward = torch.tensor([reward], device=device)

                next_state = torch.tensor(env.state, dtype=torch.float, device=device).unsqueeze(0)
                reward_energy_ep += reward_energy
                reward_time_ep += reward_time
                error_pred_ep += error_pred
                total_pred_ep += total_pred
                

                if optim:                 
                    
                    reward = torch.tensor([reward], device=device)

                    to_optim = True                    

                    memory.push(decision_state, action_, next_state, reward)
                    
                    optimize_model(phase)
                    num_optim += 1

                if not done: 
                    state = next_state
        
                else: 
                    next_state = None 
                
                if done: 
                    
                    if episode_loss: 
                        total_loss.append(mean(episode_loss))
                       
                        
                    total_reward.append(env.get_total_reward())
                    total_reward_energy_ep.append(reward_energy_ep)
                    total_reward_time_ep.append(reward_time_ep)
                    total_reward_error_pred.append(error_pred_ep/total_pred_ep)                    
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
                    total_interaction_time_epoch.append(hri_time)                    
                    total_UA_related.append(env.UA_related)
                    total_UA_unrelated.append(env.UA_unrelated)      

                    total_idle.append(env.anticipation)                           

                    break #Finish episode
        
            #print(scheduler.optimizer.param_groups[0]['lr']) #Print LR (to check scheduler)           
            
            
            if i_episode % TARGET_UPDATE == 0: #Copy the Policy Network parameters into Target Network
                target_net.load_state_dict(policy_net.state_dict())
                #scheduler.step()                


        interaction_time = sum(total_interaction_time_epoch)
        
            
        data = {'video': maximum_time,
        'interaction': interaction_time,
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
            if i_epoch % 5 == 0:
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
                    weight_prob = ''
                else:
                    weight_prob = '_NO_ACTION_PROBABILITY_EXPLORATION_' + str(cfg.NO_ACTION_PROBABILITY)
                    
                if cfg.DECISION_RATE == 'random':
                    decision_rate_name = '_DECISION_RATE_random_'
                else:
                    decision_rate_name = '_DECISION_RATE_'+str(cfg.DECISION_RATE)
                    
                if REPLAY_MEMORY > BATCH_SIZE:
                    batch_name = '_CHANGING_BATCH_SIZE_MEMORY_'
                else: 
                    batch_name = ''
                    
                if n_states < 1157:
                    z_name = '_WITHOUT_Z_'
                else: 
                    z_name = ''

                # path = os.path.join(ROOT, EXPERIMENT_NAME + '_' + dt_string  +'_PENALTY_ENERGY_FACTOR_'+str(cfg.FACTOR_ENERGY_PENALTY)+z_name+batch_name+'_EPS_START_'+str(cfg.EPS_START) + decision_rate_name +weight_prob +'_LR_'+str(LR)+ pre + freeze + '_GAMMA_'+str(GAMMA))
                path = os.path.join(ROOT, EXPERIMENT_NAME + '_' + dt_string  +'_'+cfg.TEST_FOLD+'_PENALTY_ENERGY_FACTOR_'+str(cfg.FACTOR_ENERGY_PENALTY)+z_name+batch_name+'_EPS_START_'+str(cfg.EPS_START) + decision_rate_name +weight_prob +'_LR_'+str(LR)+ pre + freeze + '_GAMMA_'+str(GAMMA))
                                  
                
                # path = os.path.join(ROOT, EXPERIMENT_NAME)
                save_path = os.path.join(path, "Graphics") 
                save_path_hist = os.path.join(save_path, "Histograms") 
                model_name = 'model_' + str(i_epoch) + '.pt'
                if not os.path.exists(path): os.makedirs(path)
                if not os.path.exists(save_path): os.makedirs(save_path)
                # if not os.path.exists(save_path_hist): os.makedirs(save_path_hist)
 
         
                print("Saving model at ", os.path.join(path, model_name))
                torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': i_epoch,
                'loss': total_loss,
                'steps': steps_done            
                }, os.path.join(path, model_name))


            ex_rate.append(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))

            if total_loss: total_loss_epoch_train.append(sum(total_loss))
            total_reward_epoch_train.append(sum(total_reward))
            total_reward_energy_epoch_train.append(sum(total_reward_energy_ep))
            total_reward_time_epoch_train.append(sum(total_reward_time_ep))

            total_time_execution_epoch_train.append(interaction_time)

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
            
            total_UA_related_epoch_train.append(sum(total_UA_related))
            total_UA_unrelated_epoch_train.append(sum(total_UA_unrelated))
            
            
            # total_idle_epoch_train.append(sum(total_idle)) #RRRRRRRR
            total_idle_epoch_train.append(np.mean(total_idle))
            
            total_results_train = [total_CA_intime_epoch_train,total_CA_late_epoch_train,total_IA_intime_epoch_train,
            total_IA_late_epoch_train,
            total_UAC_intime_epoch_train,
            total_UAC_late_epoch_train,
            total_UAI_intime_epoch_train,
            total_UAI_late_epoch_train,
            total_CI_epoch_train,
            total_II_epoch_train,
            total_UA_related_epoch_train, 
            total_UA_unrelated_epoch_train] 
            

 
            #PLOT TRAIN
            if i_epoch % 30 == 0: plot_each_epoch(i_epoch, phase,save_path,
            minimum_time,
            total_results_train,
            total_loss_epoch_train,
            total_reward_epoch_train,
            maximum_time,
            total_time_execution_epoch_train,
            total_reward_energy_epoch_train,
            total_reward_time_epoch_train,
            ex_rate, idle_frames=total_idle_epoch_train)
            
            
            
            
            # if i_epoch == NUM_EPOCH-1:
            data_train = {
            'CA_intime': total_CA_intime_epoch_train,
            'CA_late':total_CA_late_epoch_train,
            'IA_intime': total_IA_intime_epoch_train,
            'IA_late':total_IA_late_epoch_train,
            'UAC_intime': total_UAC_intime_epoch_train,
            'UAC_late': total_UAC_late_epoch_train,
            'UAI_intime': total_UAI_intime_epoch_train,
            'UAI_late': total_UAI_late_epoch_train,
            'CI': total_CI_epoch_train,
            'II': total_II_epoch_train,
            'prediction error': np.mean(total_reward_error_pred)
            }


        elif phase=='val':
            # print(len(total_loss))
            # pdb.set_trace()
            total_loss_epoch_val.append(sum(total_loss))
            total_reward_epoch_val.append(sum(total_reward))
            total_reward_energy_epoch_val.append(sum(total_reward_energy_ep))
            total_reward_time_epoch_val.append(sum(total_reward_time_ep))

            #total_time_execution_epoch_val.append(sum(total_time_interaction))
            total_time_execution_epoch_val.append(interaction_time)

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
            
            total_UA_related_epoch_val.append(sum(total_UA_related))
            total_UA_unrelated_epoch_val.append(sum(total_UA_unrelated))
            
            # total_idle_epoch_val.append(sum(total_idle)) #RRRRRRR
            total_idle_epoch_val.append(np.mean(total_idle))
            
            total_results = [total_CA_intime_epoch_val,total_CA_late_epoch_val,total_IA_intime_epoch_val,
            total_IA_late_epoch_val,
            total_UAC_intime_epoch_val,
            total_UAC_late_epoch_val,
            total_UAI_intime_epoch_val,
            total_UAI_late_epoch_val,
            total_CI_epoch_val,
            total_II_epoch_val,
            total_UA_related_epoch_val, 
            total_UA_unrelated_epoch_val]
            #PLOT VALIDATION            
            
            if i_epoch % 20== 0: plot_each_epoch(i_epoch, phase,save_path,
            minimum_time, 
            total_results,
            total_loss_epoch_val,
            total_reward_epoch_val,
            maximum_time,
            total_time_execution_epoch_val,
            total_reward_energy_epoch_val,
            total_reward_time_epoch_val, idle_frames=total_idle_epoch_val)            
            
            
            #---------------------------------------------------------------------------------------
            
            #PLOT TOGETHER
            if i_epoch % 25 == 0: plot_each_epoch_together(i_epoch,save_path,
            minimum_time,
            total_results_train,
            total_loss_epoch_train,
            total_reward_epoch_train,
            maximum_time,
            total_time_execution_epoch_train,
            total_reward_energy_epoch_train,
            total_reward_time_epoch_train,
            ex_rate,
            total_results,
            total_loss_epoch_val,
            total_reward_epoch_val,
            total_time_execution_epoch_val,
            total_reward_energy_epoch_val,
            total_reward_time_epoch_val)
            
            # if i_epoch == NUM_EPOCH-1:
            data_val = {
            'CA_intime': total_CA_intime_epoch_val,
            'CA_late':total_CA_late_epoch_val,
            'IA_intime': total_IA_intime_epoch_val,
            'IA_late':total_IA_late_epoch_val,
            'UAC_intime': total_UAC_intime_epoch_val,
            'UAC_late': total_UAC_late_epoch_val,
            'UAI_intime': total_UAI_intime_epoch_val,
            'UAI_late': total_UAI_late_epoch_val,
            'CI': total_CI_epoch_val,
            'II': total_II_epoch_val,
            'prediction error': np.mean(total_reward_error_pred)
            }
            

            
            

t2 = time.time() - t1 #Tak


print("\nTraining completed in {:.1f}".format(t2), "seconds.\n")
if PRETRAINED:
    with open(path +'/model_used.txt', 'w') as f: f.write(path_model.split('/')[-1])

