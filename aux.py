import numpy as np
import torch
import random
import PySimpleGUI as sg #Graphic Interface

import glob
import time
import os
from statistics import mean
import matplotlib.pyplot as plt
import config as cfg
import pandas as pd

N_ATOMIC_ACTIONS = cfg.N_ATOMIC_ACTIONS
N_OBJECTS = cfg.N_OBJECTS

"""
In this script, some auxiliary functions that are used in the environment setup (./gym-basic/gym_basic/envs/main.py) are implemented.
There are three types of functions:
    1) General purpose: regarding the management of array variables.
    2) Get state: as interface functions between the input systems and the environment. Right now using video_annotations. In the future, these functions will be used to retrieve the outputs of the Action Prediction system (among others).
    3) Reward: user interfaces to get the reward value. 
"""


# 1) GENERAL PURPOSE FUNCTIONS
#Management of vectors: one hot encoding, softmax, concatenations and deconcatenations.
#---------------------------------------------------------------------------

def one_hot(x, n):
    """
    One hot encoding of an integer variable x with n number of possible values.
    Input:
        x: (int) integer value.    
        n: (int) max number of states.
    Output:
        vector: (numpy array) one-hot encoded vector with all 0s and 1 in position x.     
    """
    
    vector = np.zeros((n))
    vector[x] = 1    
    
    return vector

def undo_one_hot(vector):
    """
    Reverts a one-hot encoded vector and returns an integer as the argmax of the vector.
    Input:
        vector: (numpy array) one-hot encoded vector with all zeros except for a 1 at any position.
    Output:
        x: (int) argmax of the vector.    
    
    """
    x = np.argmax(vector)

    return int(x)
    
def softmax(x):
    """
    Performs the softmax operation in a vector.
    Input:
        x: (numpy array) vector or matrix.
    Output:
        Softmaxed vector/matrix.    
    """
    e_x = np.exp(x - np.max(x))
    
    return e_x / e_x.sum(axis=0)    



def concat_vectors(a, b):
    """
    Concatenates two vectors along axis 0.
    
    Input:
        a: vector (numpy array) of dimension N_a.
        b: vector (numpy array) of dimension N_b.
    Output:
        v: vector (numpy array) of dimension (N_a + N_b).    
    """
    v = np.concatenate((a, b), axis=0)
    
    return v

def concat_3_vectors(a, b, c):
    """
    Concatenates two vectors along axis 0.
    
    Input:
        a: vector (numpy array) of dimension N_a.
        b: vector (numpy array) of dimension N_b.
    Output:
        v: vector (numpy array) of dimension (N_a + N_b).    
    """
    v = np.hstack((a,b,c)).ravel()
    
    return v


#State: NA + AO
def undo_concat_state(state):
    """
    Separates a full state vector into the Next Action vector and the Active Objects vector.
    Input:
        State: (numpy array) representation of the full state as a Next Action vector and the Active Object vector.
    Output:
        next_action: (numpy array) vector of dimensions N_ATOMIC_ACTIONS.
        ao: (numpy array) vector of dimensions N_OBJECTS.
    """

    if cfg.VERSION == 2:
        next_action = state[0:N_ATOMIC_ACTIONS]
        ao = state[N_ATOMIC_ACTIONS:]    
        assert next_action.shape[0] == N_ATOMIC_ACTIONS, f"Next action vector has expected number of elements"
        return next_action, ao   
    
    elif cfg.VERSION == 3: 
        next_action = state[0:N_ATOMIC_ACTIONS]
        ao = state[N_ATOMIC_ACTIONS:N_OBJECTS+N_ATOMIC_ACTIONS]    
        oit = state[N_OBJECTS+N_ATOMIC_ACTIONS:]
        assert next_action.shape[0] == N_ATOMIC_ACTIONS, f"Next action vector has expected number of elements"
        return next_action, ao, oit
    

"""
MOVING AVERAGE
"""

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# 3) GET REWARDS    
#User interfaces/Reward functions.
#---------------------------------------------------------------------------


def get_reward_keyboard():
    """
    Returns an integer value read from keyboard input.
    Output:
        reward (int): Input value or 0 if the user did not provide a valid number.
    """
    reward = input("Input reward value...\n")
    
    try:
        return int(reward)
    except: 
        print("ERROR: invalid reward value.")
        return 0    
    

def get_emotion():
    """
    Returns an integer value based on the facial expression, as a reward signal.
    Output:
        reward (int): reward value provided by the facial expression recognition system. 
    
    """
    reward = ... #whatever function calls the emotion recognition system
    try:
        return int(reward)
    except:
        print("Invalid value")
        return 0
    

def get_reward_GUI():
    """
    Gets a reward signal from a Graphical User Interface with three buttons: Negative (-1), Neutral (0) and Positive (+1).
    
    Output:
        reward (int): reward value provided by the user. 
    
    """
    button_size = (25, 15)    
    reward = 0
    
    #Button layout (as a matrix)
    interface = [[
    sg.Button('NEGATIVE', size=button_size, key='Negative', button_color='red'), 
    sg.Button('NEUTRAL', key='Neutral', size=button_size, button_color='gray'), 
    sg.Button('POSITIVE', key='Positive', size=button_size, button_color='blue')
    ]]
    
    #Generate window with the button layout
    window = sg.Window('Interface', interface, background_color='black', return_keyboard_events=True).Finalize()    
    
    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == 'Negative':
            reward = -1
            break
        
        elif event == sg.WIN_CLOSED or event == 'Neutral':
            reward = 0
            break
        
        elif event == sg.WIN_CLOSED or event == 'Positive':
            reward = +1
            break    
        
        elif event == '1:10': #Keyboard press 1
            window['Negative'].click()
        
        elif event == '2:11': #Keyboard press 2
            window['Neutral'].click()
        
        elif event == '3:12': #Keyboard press 3
            window['Positive'].click()
            
        elif event == 'q:24': #Keyboard press q
            break    
            
    window.close()
    
    return reward


def reward_confirmation_perform(action):
    """
    Performs an action after receiving confirmation (POSITIVE reward) or cancels the operation if received NEGATIVE/NEUTRAL reward. The reward is offered through a Graphical User Interface with three buttons: Negative (-1), Neutral (0) and Positive (+1).
    
    Input:
        action (int): action-output of the DQN (according to an exploration-exploitation policy).
    
    Output:
        reward (int): reward value provided by the user. 
    
    """
    button_size = (25, 15)    
    reward = 0
    
    #Button layout (as a matrix)
    interface = [[
    sg.Button('NEGATIVE', size=button_size, key='Negative', button_color='red'), 
    sg.Button('NEUTRAL', key='Neutral', size=button_size, button_color='gray'), 
    sg.Button('POSITIVE', key='Positive', size=button_size, button_color='blue')
    ]]
    
    #Generate window with the button layout    
    
    if action != 18:
        window = sg.Window('Interface', interface, background_color='black', return_keyboard_events=True).Finalize()    
        print("\nROBOT: I'm going to", cfg.ROBOT_ACTIONS_MEANINGS[action])
    
        while True:
            event, values = window.read()
        
            if event == sg.WIN_CLOSED or event == 'Negative':
                reward = -1
                break
        
            elif event == sg.WIN_CLOSED or event == 'Neutral':
                reward = 0
                break
        
            elif event == sg.WIN_CLOSED or event == 'Positive':
                reward = +1
                break    
                
            elif event == '1:10': #Keyboard press 1
                window['Negative'].click()
                
            elif event == '2:11': #Keyboard press 2
                window['Neutral'].click()
            
            elif event == '3:12': #Keyboard press 3
                window['Positive'].click()    
        window.close()
    
        global frame
    
        #Confirmation
        if reward == 1:
            print("PERFORMING ACTION - ", cfg.ROBOT_ACTIONS_MEANINGS[action], "\n")
            for i in range(10):
                print("."*(i+1), end='\r')
                time.sleep(0.25)
            #time.sleep(1)
            frame += 300
        
        else: 
            #print("Or maybe not.")
            pass    
    
    
    return reward




def plot_each_epoch(i_epoch, phase,save_path,minimum_time, total_results,total_loss_epoch,total_reward_epoch,maximum_time,total_time_execution_epoch,total_reward_energy_epoch,total_reward_time_epoch,ex_rate=0):
                
    n = int(cfg.NUM_EPOCH*0.05)
    if i_epoch >= 2*n: 
        plot_detailed_results(n, total_results, save_path, phase)

    n = 10
    if i_epoch >= 2*n: 
        plot_detailed_results(n, total_results, save_path, phase)
        

    
    if phase == 'train':
    	fig1 = plt.figure(figsize=(20, 6))
    	plt.subplot(121)
    	plt.title(phase+" Loss")
    	plt.xlabel("Epoch")
    	plt.ylabel("MSE")
    	plt.plot(total_loss_epoch, 'r')
    	plt.subplot(122)
    	plt.title(phase+" Exploration rate")
    	plt.xlabel("Epoch")
    	plt.ylabel("Epsilon")
    	plt.plot(ex_rate, 'c')
    	fig1.savefig(save_path+'/'+phase+'_LOSS_EXPLORATION.jpg')
    plt.close('all')
    
    
    
    # --------- ACTIONS ---------------------------
    fig1 = plt.figure(figsize=(25, 8))

    
    plt.subplot2grid((2,6), (0,0))
    plt.title("Short-term correct actions (in time)")
    plt.plot(total_results[0], 'springgreen')


    plt.subplot2grid((2,6), (1,0))
    plt.title("Short-term correct actions (late)")
    plt.plot(total_results[1], 'springgreen')
    plt.xlabel("Epoch")

       
    plt.subplot2grid((2,6), (0,4))
    plt.title("Incorrect actions (in time)")
    plt.plot(total_results[2], 'red')


    plt.subplot2grid((2,6), (1,4))
    plt.title("Incorrect actions (late)")
    plt.plot(total_results[3], 'red')
    plt.xlabel("Epoch")

      
    plt.subplot2grid((2,6), (0,1))
    plt.title("Long-term correct actions(in time)")
    plt.plot(total_results[4], 'springgreen')


    plt.subplot2grid((2,6), (1,1))
    plt.title("Long-term correct actions (late)")
    plt.plot(total_results[5], 'springgreen')
    plt.xlabel("Epoch")

    
    
    plt.subplot2grid((2,6), (0,3))
    plt.title("Unnecessary actions (in time)")
    plt.plot(total_results[6], 'red')


    plt.subplot2grid((2,6), (1,3))
    plt.title("Unnecessary actions (late)")
    plt.plot(total_results[7], 'red')
    plt.xlabel("Epoch")

    
    plt.subplot2grid((2,6), (0,2))
    plt.title("Correct inactions")
    plt.plot(total_results[8], 'springgreen')


    plt.subplot2grid((2,6), (1,2))
    plt.title("Incorrect inactions")
    plt.plot(total_results[9], 'red')
    plt.xlabel("Epoch")
    
    plt.subplot2grid((2,6), (0,5))
    plt.title("Unn recipe-related")
    plt.plot(total_results[10], 'royalblue')
    
    plt.subplot2grid((2,6), (1,5))
    plt.title("Unn recipe-unrelated")
    plt.plot(total_results[11], 'crimson')
    plt.xlabel("Epoch")


    fig1.savefig(save_path+'/'+phase+'_ACTIONS.jpg')
    plt.close(fig1)
    
    
    # ---------- INTERACTION TIME ------------------------------
    #total_time_video_epoch = [sum(total_time_video)]*len(total_time_execution_epoch)
    
    
    fig1 = plt.figure(figsize=(15, 6))
    plt.axhline(y=maximum_time, color='k')
    plt.plot(total_time_execution_epoch, 'b--')
    plt.axhline(y=minimum_time, color='r')
    plt.legend(["Video","Interaction", "Minimum"])
    plt.xlabel("Epoch")
    plt.ylabel("Frames")
    plt.title(phase+" | Interaction time")

    fig1.savefig(save_path+'/'+phase+'_INTERACTION_TIME.jpg')
    plt.close(fig1)
    
    
    
    #---------- REWARDs ---------------------------
    fig1 = plt.figure(figsize=(20, 6))
     
    plt.subplot2grid((1,3), (0,0))
    plt.plot(total_reward_energy_epoch, 'b:')
    plt.legend(["Energy reward"])
    plt.xlabel("Epoch")

    plt.title(phase+" | Energy reward")
    
    plt.subplot2grid((1,3), (0,1))
    plt.plot(total_reward_time_epoch, 'b:')
    plt.legend(["Time reward"])
    plt.xlabel("Epoch")

    plt.title(phase+" | Time reward")
    
    
    plt.subplot2grid((1,3), (0,2))
    plt.plot(total_reward_epoch, 'b-.')
    plt.xlabel("Epoch")
    plt.legend(["Total reward"])
    plt.title("Total reward")

    fig1.savefig(save_path+'/'+phase+'_REWARD.jpg')
    plt.close(fig1)
    
    plt.close('all')
    
    
    
    
def plot_each_epoch_together(i_epoch,save_path,minimum_time, total_results_train,total_loss_epoch_train,total_reward_epoch_train,maximum_time,total_time_execution_epoch_train,total_reward_energy_epoch_train,total_reward_time_epoch_train,ex_rate,total_results,total_loss_epoch_val,total_reward_epoch_val,total_time_execution_epoch_val,total_reward_energy_epoch_val,total_reward_time_epoch_val):
                 
    
    
    #------------- ACTIONS --------------------------------
    
    
    #123456 -> 6 columnas
    fig1 = plt.figure(figsize=(25, 8))
    
    plt.subplot2grid((2,6), (0,0))
    plt.title("Short-term correct actions (in time)")
    plt.plot(total_results_train[0],'b',label='train')
    plt.plot(total_results[0], 'm',label='val')


    plt.legend()
    plt.subplot2grid((2,6), (1,0))
    plt.title("Short-term correct actions (late)")
    plt.plot(total_results_train[1],'b',label='train')
    plt.plot(total_results[1],'m',label='val')
    plt.xlabel("Epoch")

    plt.legend()
    
    
    plt.subplot2grid((2,6), (0,4))
    plt.title("Incorrect actions (in time)")
    plt.plot(total_results_train[2],'b',label='train')
    plt.plot(total_results[2],'m',label='val')


    plt.legend()
    plt.subplot2grid((2,6), (1,4))
    plt.title("Incorrect actions (late)")
    plt.plot(total_results_train[3],'b',label='train')
    plt.plot(total_results[3],'m',label='val')
    plt.xlabel("Epoch")

    plt.legend()
    
    
    plt.subplot2grid((2,6), (0,1))
    plt.title("Long-term correct actions (in time)")
    plt.plot(total_results_train[4],'b',label='train')
    plt.plot(total_results[4],'m',label='val')


    plt.legend()
    plt.subplot2grid((2,6), (1,1))
    plt.title("Long-term correct actions (late)")
    plt.plot(total_results_train[5],'b',label='train')
    plt.plot(total_results[5],'m',label='val')
    plt.xlabel("Epoch")

    plt.legend()
    
    
    plt.subplot2grid((2,6), (0,3))
    plt.title("Unnecessary actions (in time)")
    plt.plot(total_results_train[6],'b',label='train')
    plt.plot(total_results[6],'m',label='val')


    plt.legend()
    plt.subplot2grid((2,6), (1,3))
    plt.title("Unnecessary actions (late)")
    plt.plot(total_results_train[7],'b',label='train')
    plt.plot(total_results[7],'m',label='val')
    plt.xlabel("Epoch")

    plt.legend()
    
    
    plt.subplot2grid((2,6), (0,2))
    plt.title("Correct inactions")
    plt.plot(total_results_train[8],'b',label='train')
    plt.plot(total_results[8],'m',label='val')


    plt.legend()
    plt.subplot2grid((2,6), (1,2))
    plt.title("Incorrect inactions")
    plt.plot(total_results_train[9],'b',label='train')
    plt.plot(total_results[9],'m',label='val')
    plt.xlabel("Epoch")

    plt.legend()
    
    
    #123456 {
    plt.subplot2grid((2,6), (0,5))
    plt.title("Unn recipe-related")    
    plt.plot(total_results_train[10], 'b', label='train')
    plt.plot(total_results[10], 'm', label='val')
    plt.legend()
    
    
    plt.subplot2grid((2,6), (1,5))
    plt.title("Unn recipe-unrelated")
    plt.plot(total_results_train[11], 'b', label='train')
    plt.plot(total_results[11], 'm', label='val')
    plt.xlabel("Epoch")
    
    plt.legend()   
    
    
    # }
    
    
    
    
    fig1.savefig(save_path+'/01_ACTIONS.jpg')
    plt.close(fig1)
    
   
    #------------------ INTERACTION TIME------------------------------------------
    
    fig1 = plt.figure(figsize=(15, 6))
    plt.axhline(y=maximum_time,color='k', label='Video')
    plt.plot(total_time_execution_epoch_train,'b--',label='Train')
    plt.plot(total_time_execution_epoch_val,'m--',label='Validation')
    plt.axhline(y=minimum_time, color='r', label='Minimum')
    
    plt.legend(["Video","Train Interaction","Val Interaction"])
    plt.xlabel("Epoch")
    plt.ylabel("Frames")
    plt.title("Interaction time")

    plt.legend()
    fig1.savefig(save_path+'/02_INTERACTION_TIME.jpg')
    plt.close(fig1)
    

    
    
    # -------------------- REWARDS ---------------------------------------------
    fig1 = plt.figure(figsize=(20, 6))
    
    plt.subplot2grid((1,3), (0,0))
    
    plt.plot(total_reward_energy_epoch_train, 'b:', label='train')
    plt.plot(total_reward_energy_epoch_val,'m:',label='val')
    plt.xlabel("Epoch")

    plt.title("Energy reward")
    plt.legend()
    
     
    plt.subplot2grid((1,3), (0,1))
    
    plt.plot(total_reward_time_epoch_train,'b:',label='train')
    plt.plot(total_reward_time_epoch_val,'m:', label='val')
    
    plt.xlabel("Epoch")

    plt.title("Time reward")
    plt.legend()

    
    plt.subplot2grid((1,3), (0,2))    
    
    plt.plot(total_reward_epoch_train,'b-.',label='train')
    plt.plot(total_reward_epoch_val,'m-.',label='val')
    plt.xlabel("Epoch")
    plt.title("Total reward")
    plt.legend()
    
    
    fig1.savefig(save_path+'/03_REWARDS.jpg')
    plt.close(fig1)
    
    plt.close('all')
    
    
def plot_detailed_results (n, total_results, save_path, MODE): 
    
    total_CA_intime = total_results[0]
    total_CA_late = total_results[1]
    total_IA_intime = total_results[2]
    total_IA_late = total_results[3]
    total_UAC_intime = total_results[4]
    total_UAC_late = total_results[5]
    total_UAI_intime = total_results[6]
    total_UAI_late = total_results[7]
    total_CI = total_results[8]
    total_II = total_results[9]
    
    x_axis = np.arange(0,len(total_CA_intime)-1,n).tolist()
    n_total_CA_intime = [sum(total_CA_intime[i:i+n])/n for i in range(0,len(total_CA_intime)-1,n)]
    n_total_CA_late = [sum(total_CA_late[i:i+n])/n for i in range(0,len(total_CA_late)-1,n)]
    n_total_IA_intime = [sum(total_IA_intime[i:i+n])/n for i in range(0,len(total_IA_intime)-1,n)]
    n_total_IA_late = [sum(total_IA_late[i:i+n])/n for i in range(0,len(total_IA_late)-1,n)]
    n_total_UAC_intime = [sum(total_UAC_intime[i:i+n])/n for i in range(0,len(total_UAC_intime)-1,n)]
    n_total_UAC_late = [sum(total_UAC_late[i:i+n])/n for i in range(0,len(total_UAC_late)-1,n)]
    n_total_UAI_intime = [sum(total_UAI_intime[i:i+n])/n for i in range(0,len(total_UAI_intime)-1,n)]
    n_total_UAI_late = [sum(total_UAI_late[i:i+n])/n for i in range(0,len(total_UAI_late)-1,n)]
    n_total_CI = [sum(total_CI[i:i+n])/n for i in range(0,len(total_CI)-1,n)]
    n_total_II = [sum(total_II[i:i+n])/n for i in range(0,len(total_II)-1,n)]


    fig1 = plt.figure(figsize=(25, 8))
   
    
    plt.subplot2grid((2,5), (0,0))
    plt.title("Short-term correct actions (in time)",fontsize=14)
    plt.plot(x_axis,n_total_CA_intime)  


    plt.subplot2grid((2,5), (1,0))
    plt.title("Short-term actions (late)",fontsize=14)
    plt.plot(x_axis,n_total_CA_late)
    plt.xlabel("Epoch")

    
    
    plt.subplot2grid((2,5), (0,4))
    plt.title("Incorrect actions (in time)",fontsize=14)
    plt.plot(x_axis,n_total_IA_intime)


    plt.subplot2grid((2,5), (1,4))
    plt.title("Incorrect actions (late)",fontsize=14)
    plt.plot(x_axis,n_total_IA_late)
    plt.xlabel("Epoch")

    
    
    plt.subplot2grid((2,5), (0,1))
    plt.title("Long-term correct actions (in time)",fontsize=14)
    plt.plot(x_axis,n_total_UAC_intime)


    plt.subplot2grid((2,5), (1,1))
    plt.title("Long-term correct actions (late)",fontsize=14)
    plt.plot(x_axis,n_total_UAC_late)
    plt.xlabel("Epoch")

    
    
    plt.subplot2grid((2,5), (0,3))
    plt.title("Unnecessary actions incorrect (in time)",fontsize=14)
    plt.plot(x_axis,n_total_UAI_intime)

    plt.ylabel("Amount action")
    plt.subplot2grid((2,5), (1,3))
    plt.title("Unnecessary actions incorrect (late)",fontsize=14)
    plt.plot(x_axis,n_total_UAI_late)
    plt.xlabel("Epoch")

    
    
    plt.subplot2grid((2,5), (0,2))
    plt.title("Correct inactions",fontsize=14)
    plt.plot(x_axis,n_total_CI)


    plt.subplot2grid((2,5), (1,2))
    plt.title("Incorrect inactions",fontsize=14)
    plt.plot(x_axis,n_total_II)
    plt.xlabel("Epoch")

    # plt.show()

    fig1.savefig(save_path+'/'+MODE+'_ACTIONS_average'+str(n)+'.jpg')
    plt.close(fig1)
    
    plt.close('all')
 
def get_estimations_action_time_human():
    
    # Get the list of all files and directories
    path = os.path.abspath(os.getcwd())
    dir_list = os.listdir(path+'/video_annotations/Real_data/train/')
    
    duration_action_compilation_list = [[] for _ in range(33)]
    
    print(dir_list)
    
    for idx in dir_list:
        
        with open(path+'/video_annotations/Real_data/train/'+idx+'/labels_updated.pkl', 'rb') as f:
                data = np.load(f, allow_pickle=True)
                for i in range(len(data)):
                    frame_duration = data['frame_end'][i]-data['frame_init'][i]
                    duration_action_compilation_list[data['label'][i]].append(frame_duration)
                print(data)
            
    avg_list = []
    for idx in range(len(duration_action_compilation_list)):
        if duration_action_compilation_list[idx]: 
            avg_list.append(mean(duration_action_compilation_list[idx]))
        else: 
            avg_list.append(0)
            
        
    atomic_actions = ['other manipulation',
                                'pour milk',
                                'pour water',
                                'pour coffee',
                                'pour Nesquik',
                                'pour sugar',
                                'put microwave',
                                'stir spoon',
                                'extract milk fridge',
                                'extract water fridge',
                                'extract sliced bread',
                                'put toaster',
                                'extract butter fridge',
                                'extract jam fridge',
                                'extract tomato sauce fridge',
                                'extract nutella fridge',
                                'spread butter',
                                'spread jam',
                                'spread tomato sauce',
                                'spread nutella',
                                'pour olive oil',
                                'put jam fridge',
                                'put butter fridge',
                                'put tomato sauce fridge',
                                'put nutella fridge',
                                'pour milk bowl',
                                'pour cereals bowl',
                                'pour nesquik bowl',
                                'put bowl microwave',
                                'stir spoon bowl',
                                'put milk fridge',
                                'put sliced bread plate',
                                'TERMINAL STATE',
                                ]
            
    # Calling DataFrame constructor after zipping
    # both lists, with columns specified
    df = pd.DataFrame(list(zip(atomic_actions, avg_list)),
                    columns =['atomic_actions', 'avg_frame'])

    ROBOT_ACTION_DURATIONS = {}
    for idx in range(len(cfg.ROBOT_ACTIONS_MEANINGS)):
        ROBOT_ACTION_DURATIONS[idx] = 0
        
    for idx_AR, value_AR in cfg.ROBOT_ACTIONS_MEANINGS.items():
        current_object = value_AR.split(" ")[1]
        for index, row in df.iterrows():
            if current_object in row['atomic_actions']:
                if 'bring' in value_AR:
                    if 'extract' in row['atomic_actions']: 
                        ROBOT_ACTION_DURATIONS[idx_AR] = row['avg_frame']
                elif 'put' in (row['atomic_actions'] and value_AR):
                    ROBOT_ACTION_DURATIONS[idx_AR] = row['avg_frame']
                
    return ROBOT_ACTION_DURATIONS
       





