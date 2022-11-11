import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import time
from aux import *
import config as cfg
import numpy as np
import glob 
import pdb
from numpy import random
import pandas as pd

#CONFIGURATION GLOBAL ENVIRONMENT VARIABLES
ACTION_SPACE = cfg.ACTION_SPACE
N_ATOMIC_ACTIONS = cfg.N_ATOMIC_ACTIONS
N_OBJECTS = cfg.N_OBJECTS

ATOMIC_ACTIONS_MEANINGS = cfg.ATOMIC_ACTIONS_MEANINGS
OBJECTS_MEANINGS = cfg.OBJECTS_MEANINGS
ROBOT_ACTIONS_MEANINGS = cfg.ROBOT_ACTIONS_MEANINGS
ROBOT_ACTION_DURATIONS = cfg.ROBOT_ACTION_DURATIONS

VERSION = cfg.VERSION

#ANNOTATION-RELATED VARIABLES
root = "./video_annotations/train/*"
videos = glob.glob(root)
random.shuffle(videos)
total_videos = len(videos)
video_idx = 0 #Index of current video
action_idx = 0 #Index of next_action
frame = 0 #Current frame

annotations = np.load(videos[video_idx], allow_pickle=True)
print(annotations)

class BasicEnv(gym.Env):
    message = "Custom environment for recipe preparation scenario."
    
    def __init__(self, display=False, test=False):
        self.action_space = gym.spaces.Discrete(ACTION_SPACE) #[0, ACTION_SPACE-1]
        
        if VERSION == 1:
            self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS) #State as Next Action
        elif VERSION == 2:
            self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS+N_OBJECTS) #State as Next Action + VWM     

        self.state = [] #One hot encoded state        
        self.total_reward = 0
        
        self.prev_state = []
        
        self.action_repertoire = ROBOT_ACTIONS_MEANINGS
        self.next_atomic_action_repertoire = ATOMIC_ACTIONS_MEANINGS
        
        self.display = display
        
        global root, videos, total_videos, annotations
        
        if test:
        	print("==== TEST SET ====")
        	root = "./video_annotations/test/*"
        	videos = glob.glob(root)
        	random.shuffle(videos)
        	total_videos = len(videos)
        	annotations = np.load(videos[video_idx], allow_pickle=True)
        
        self.CA_intime = 0
        self.CA_late = 0
        self.IA_intime = 0
        self.IA_late = 0	
        self.UA_intime = 0
        self.UA_late = 0
        self.CI = 0
        self.II = 0
        
    
    def get_action_meanings(self):
        return self.action_repertoire
    def get_state_meanings(self):
        return self.state_repertoire
    
    def energy_robot_reward (self, action):
         
        energy = -ROBOT_ACTION_DURATIONS[action]/100 
         
        return energy 
        
    def time_wait_reward (self):
        global cum 
        
        cum += -1
        
    def select_inaction_sample (self, inaction):
        random_position = random.randint(len(inaction))
        
        self.prev_state = inaction[random_position][1]
        self.state = inaction[random_position][1] # esto se hace para que el next_state sea el siguiente al guardado
        reward = inaction[random_position][2]
        
        return reward
        
    def select_correct_action (self, action): 
        
        global frame
        
        length = len(annotations['label']) -1 
        last_frame = int(annotations['frame_end'][length])
        
        for idx, val in cfg.ROBOT_ACTION_DURATIONS.items(): 
            reward, _ = self._take_action(idx)
            if reward > -1: 
                correct_action = idx
                duration_action = val
        
        print("Acción tomada: ",cfg.ROBOT_ACTIONS_MEANINGS[action])
        print("Corrección de accion: ",cfg.ROBOT_ACTIONS_MEANINGS[correct_action])
        
        
        new_threshold = duration_action + frame 
        if new_threshold > last_frame: 
            new_threshold = last_frame
            

        return new_threshold, correct_action
                
    def update(self, update_type, fr_execution=0): 
        global frame, action_idx, inaction
        
        length = len(annotations['label']) - 1 
        fr_init_next = int(annotations['frame_init'][action_idx])
        
        if update_type == "action":
            if action_idx + 1 <= length: 
                frame = int(annotations['frame_end'][action_idx]) 
                action_idx = action_idx + 1
                inaction = []
        if update_type == "unnecesary":
            if action_idx + 1 <= length:
                if fr_execution >= fr_init_next:
                    frame = fr_init_next - 1
                    action_idx = action_idx + 1
                    inaction = []
                else:
                    inaction = []
                # action_idx = action_idx + 1
        #   QUE SE HACE SI ESTAMOS EN EL ESTADO TERMINAL? 
        
        
    def time_course (self, action, flag_decision):
        global frame, action_idx, inaction
        
        flag_no_action = False 
        
        if action_idx == 0: 
            action_idx = 1
        length = len(annotations['label']) -1 
        fr_execution = cfg.ROBOT_ACTION_DURATIONS[int(action)] + frame
        fr_end = int(annotations['frame_end'][action_idx-1])
        fr_init_next = int(annotations['frame_end'][action_idx]) 
        last_frame = int(annotations['frame_init'][length])
        
        # pdb.set_trace()
        if action != 18: 
            if fr_execution > last_frame: 
                threshold = last_frame
            else:     
                threshold = max(fr_execution, fr_end)
        else: 
            
            if frame == fr_end - 1 or frame == fr_init_next - 1: 
                if flag_decision == True:
                    flag_no_action = True
                else:
                    if len(inaction) > 0:
                        flag_decision = True
                        flag_no_action = True
                        # pdb.set_trace()
                
                if frame == fr_init_next - 1: 
                    threshold = fr_init_next 
                else:
                    threshold = fr_end 
           
            else: 
                threshold = frame
                if len(inaction) > 0:
                    print("Times no action selected: ", len(inaction))
                    
                    
        return threshold, fr_execution, fr_end, flag_no_action, flag_decision 
    
    def evaluation (self, action, fr_execution, fr_end, flag_no_action, frame_post):
        global frame, action_idx, cum, flag_evaluated, inaction, new_energy
        
        optim = True
        simple_reward, action_robot = self._take_action(action)
        new_threshold = 0 
        flag_break = False
        flag_pdb = False
        energy  = self.energy_robot_reward(action)
        reward =  energy

        if flag_evaluated == 'Incorrect action' or flag_evaluated == "Incorrect inaction":   
            
            if frame == fr_execution: 
               
                flag_evaluated == 'Not evaluated'
                self.update("action")
                
             
                reward = energy + new_energy + cum
                flag_break = True
                # if flag_evaluated == "Incorrect inaction":
                #     # flag_pdb = True
             
            else: 
                self.time_wait_reward()

                    
        else: 
           
            if action != 18: 
                # CORRECT ACTION
                if simple_reward > -1: 
                    # In time
                    if fr_execution <= fr_end: 
                       
                        # pdb.set_trace()
                        if frame == fr_execution: 
                            # print("FROZEN REWARD: ",frozen_reward)
                            frame_post.append(frame)
                            reward =  energy
                            
                        if frame == fr_end:
                        
                            self.CA_intime += 1
                            print("Action idx: ", action_idx)
                            print("*************** CORRECT ACTION (in time) ***************")
                            inaction.append([action, frame, reward]) 
                            self.update("action")
                        
                    # Late
                    else: 
                        if frame == fr_execution: 
                        
                            self.CA_late += 1
                            print("Action idx: ", action_idx)
                            print("*************** CORRECT ACTION (late) ***************")
                            inaction.append([action, frame, reward])
                            frame_post.append(frame)
                            self.update("action")                          
                            reward = energy + cum

                        if frame >=  fr_end: 
                        
                            self.time_wait_reward()
                        
                # # INCORRECT
                else: 
              
                    # INCORRECT ACTION
                    if action_robot == True: 
                        
                        if fr_execution <= fr_end: 
                            if frame == fr_execution: 
                                frame_post.append(frame)
                                
                            if frame == fr_end: 
                                self.IA_intime += 1
                                print("Action idx: ", action_idx)
                                print("*************** INCORRECT ACTION (in time) ***************")
                                inaction.append([action, frame, reward])
                                new_threshold, correct_action = self.select_correct_action(action)
                                flag_evaluated = 'Incorrect action'
                                new_energy = self.energy_robot_reward(correct_action)
                                
                                
                        else: 
                            if frame >= fr_end:
                                self.time_wait_reward()
                            if frame == fr_execution: 
                                self.IA_late += 1
                                print("Action idx: ", action_idx)
                                print("*************** INCORRECT ACTION (late) ***************")
                                inaction.append([action, frame, reward])
                                new_threshold, correct_action = self.select_correct_action(action)
                                flag_evaluated = 'Incorrect action'
                                frame_post.append(frame)
                                new_energy = self.energy_robot_reward(correct_action)

                        # aumentar el tiempo, seleccionando la accion correcta + unos frmaes mas para devolver objeto
                        
                    # UNNECESARY ACTION 
                    else: 
                        if fr_execution <= fr_end: 
                            if frame == fr_execution:
                                frame_post.append(frame)
                                reward = energy
                                self.update("unnecesary")
                            if frame == fr_end: 
                                self.UA_intime += 1
                                print("Action idx: ", action_idx)
                                print("*************** UNNECESARY ACTION (in time) ***************")
                                inaction.append([action, frame, reward])
                        else: 
                            
                            if frame >= fr_end:
                                self.time_wait_reward()
                              
                            if frame == fr_execution: 
                                self.UA_late += 1
                                print("Action idx: ", action_idx)
                                print("*************** UNNECESARY ACTION (late) ***************")
                                inaction.append([action, frame, reward])
                                frame_post.append(frame)
                                self.update("unnecesary", fr_execution)
                                reward =  energy + cum
                                flag_break = True

                               
                        
            # if action == 18 (no se hace nada)
            else:
                inaction.append([action, self.state, reward])
                if flag_no_action == True:
                    if frame == fr_end: 
                        # CORRECT INACTION
                        if simple_reward > -1: 
                            # no se actualiza como antes 
                            self.CI += 1
                            print("Action idx: ", action_idx)
                            print("*************** CORRECT INACTION ***************")
                            reward = self.select_inaction_sample(inaction)
                  
                        # INCORRECT INACTION
                        else: 
                            self.II += 1
                            new_threshold, correct_action = self.select_correct_action(action)
                            new_energy = self.energy_robot_reward(correct_action)
                            flag_evaluated = "Incorrect inaction"
                            print("Action idx: ", action_idx)
                            print("*************** INCORRECT INACTION ***************")
                        frame_post.append(frame)

                        # flag_pdb = True
                else: 
                    optim = False


        return reward, new_threshold, flag_break, optim, flag_pdb, frame_post
        
    def prints_terminal(self, action, frame_prev, frame_post, reward):
        
        global annotations
        
        person_states_index = annotations['label']
        fr_init = annotations['frame_init']
        fr_end = annotations['frame_end']
        
        person_states = []
        
        for idx,val in enumerate(person_states_index):
            person_states.append(ATOMIC_ACTIONS_MEANINGS[val])
        
        data = {"States": person_states, "Frame init": fr_init, "Frame end": fr_end}
        
        df = pd.DataFrame(data)
        

        accion_robot = ROBOT_ACTIONS_MEANINGS[action]
        
        data_robot = {"Robot action":accion_robot, "Frame init": int(frame_prev), "Frame end": str(frame_post), "Reward": reward}
        # pdb.set_trace()
        df_robot = pd.DataFrame(data_robot, index=[0])
        
        print("----------------------------------- Video -----------------------------------")
        print(df)
        print("\n----------------------------------- Robot -----------------------------------")
        print(df_robot)
        
    def step(self, action):
        """
        Transition from the current state (self.state) to the next one given an action.
        
        Input:
            action: (int) action taken by the agent.
        Output:
            next_state: (numpy array) state transitioned to after taking action.
            reward: (int) reward received. 
            done: (bool) True if the episode is finished (the recipe has reached its end).
            info:    
        """
        global frame, action_idx, annotations, cum, flag_evaluated, inaction, new_energy
        
        """
        print("\nann: ", annotations)
        print("F: ", frame)
        print("Previous end frame. ", annotations['frame_end'][action_idx-1])
        
        print("ACTION: ", action)
        
        print("Execution frame: ", self.perform_action_time(action))
        print("idx: ", action_idx)
        print("ANNOTATION RIGHT NOW: ", annotations['label'][action_idx])
        """
        
        flag_decision = action[1]
        action = action[0]
        assert self.action_space.contains(action)
        
        reward = 0
        cum = 0
        new_energy = 0
        done = False
        optim = False
        flag_pdb = False
        freeze_state = False 
        flag_break = False
        flag_evaluated = 'Not evaluated' 
        
        threshold, fr_execution, fr_end, flag_no_action, flag_decision = self.time_course(action, flag_decision)
        

        print('Frame prev: ', frame)
        # print("Prev State: ", cfg.ATOMIC_ACTIONS_MEANINGS[undo_one_hot(self.prev_state[:33])])
        # # print("Threshold: ", threshold)
        frame_prev = frame 
        prev_state = self.prev_state
        frame_post = []
        while frame <= threshold:
            
            current_state = self.state #Current state

            if flag_decision==False:   
                # se transiciona de estado pero no se hace ninguna acción 
                freeze_state = False
                print("\n NO decision yet")
            else: 
                
                optim = True
                reward, new_threshold, flag_break, optim, flag_pdb, frame_post = self.evaluation(action, fr_execution, fr_end, flag_no_action, frame_post)
                if new_threshold != 0: 
                    threshold = new_threshold
                    fr_execution = new_threshold
                    flag_evaluated == True
                    frame_post.append(threshold)
                freeze_state = True

            if frame == threshold:
                freeze_state = False
                
            self.transition(freeze_state) #Transition to a new state
               
            next_state = self.state
            
            if undo_one_hot(self.state) == N_ATOMIC_ACTIONS-1: #If the next action is nothing (==terminal state), finish episode
                #print("TERMINAL STATE.")
                done = True
            # print("Frame: ", frame)
            #PRINT STATE-ACTION TRANSITION & REWARD
            if self.display: self.render(current_state, next_state, action, reward, self.total_reward)
            
            if flag_break: 
                break

        if optim == True:   
            self.prints_terminal(action, frame_prev, frame_post, reward)

            
        return prev_state, next_state, reward, done, optim, flag_pdb      
        
        
    def get_total_reward(self):
        return self.total_reward
    
    
    def perform_action_time(self, action):

        global annotations, action_idx, frame
        
        length = len(annotations['label'])
        last_frame = int(annotations['frame_end'][length-1])
        
        exe_frame = cfg.ROBOT_ACTION_DURATIONS[action] + frame    
        
        return min(last_frame, exe_frame)
    
    def reset(self):
        """
        Resets the environment to an initial state.
        """
        super().reset()
        
        global video_idx, action_idx, annotations, frame, inaction
        
        inaction = []
        annotations = np.load(videos[video_idx], allow_pickle=True)
        
        if video_idx+1 < total_videos:
            video_idx += 1
        else:
            video_idx = 0
            print("EPOCH COMPLETED.")
        
        action_idx = 1   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! START AT 0 OR 1? 
        frame = 0
        self.total_reward = 0    
        
        #First Next_action and Active Object    
        na = one_hot(annotations['label'][action_idx], N_ATOMIC_ACTIONS)
        ao = np.zeros((N_OBJECTS))
        
        ao_idx = annotations['object_label'][action_idx]    
        if type(ao_idx) == int:
            pass
        else:
            for idx in ao_idx:
                ao = 0.6 * ao
                ao[idx] = 1
        
        if VERSION == 1:
            self.state = na
            self.prev_state = one_hot(annotations['label'][0], N_ATOMIC_ACTIONS)
        elif VERSION == 2:
            self.state = concat_vectors(na, ao)
            self.prev_state = concat_vectors(one_hot(annotations['label'][0], N_ATOMIC_ACTIONS), ao)

        self.CA_intime = 0
        self.CA_late = 0
        self.IA_intime = 0
        self.IA_late = 0	
        self.UA_intime = 0
        self.UA_late = 0
        self.CI = 0
        self.II = 0
        
        return self.state


    def _take_action(self, action): 
        """
        Version of the take action function that considers a unique correct robot action for each state, related to the required object and its position (fridge or table). 
                
        Input:
            action: (int) from the action repertoire taken by the agent.
        Output:
            reward: (int) received from the environment.
        
        """

        if VERSION == 1:
            state = undo_one_hot(self.state) #If the state is the Next Action vector, undo the O-H to obtain the integer value.
        elif VERSION == 2: #If the state is NA + VWM, first separate the two variables and then obtain the value of the state from the Next Action.
            na, ao = undo_concat_state(self.state)
            state = undo_one_hot(na) #Only consider the Next Action as the state.    
        
        reward = 0
        positive_reward = 0
        action_robot = False
        
        if state == 1: #'pour milk'
            
            if action == 18: #'bring milk'
                reward = positive_reward
                
            else: reward = -1    
        
        elif state == 2: #'pour water'
           
            if action == 18: #'bring water'
                reward = positive_reward
            else: reward = -1
        
        elif state == 3: #'pour coffee'
            if action == 18: #'do nothing' -> *coffee is at arm's reach
                reward = positive_reward
            else: reward = -1                    
        
        elif state == 4: #'pour Nesquik'
            if action == 18:
                reward = positive_reward
            else: reward = -1
        
        elif state == 5: #'pour sugar'
            if action == 18:
                reward = positive_reward
            else: reward = -1
        
        elif state == 6: #'put microwave'
            if action == 18:
                reward = positive_reward
            else: reward = -1    
        
        elif state == 7: #'stir spoon
            if action == 18:
                reward = positive_reward
            else: reward = -1
        
        elif state == 8: #'extract milk fridge'
            action_robot = True
            if action == 8:
                reward = positive_reward
            else: reward = -1                    
        
        elif state == 9: #'extract water fridge'
            action_robot = True
            if action == 17:
                reward = positive_reward
            else: reward = -1
        
        elif state == 10: #'extract sliced bread'
            if action == 18:
                reward = positive_reward
            else: reward = -1
        
        elif state == 11: #'put toaster'
            if action == 18:
                reward = positive_reward
            else: reward = -1    
        
        elif state == 12: #'extract butter fridge'
            action_robot = True
            if action == 1: #'bring butter'
                reward = positive_reward
            else: reward = -1
        
        elif state == 13: #'extract jam fridge'
            action_robot = True
            if action == 6: #'bring jam'
                reward = positive_reward
            else: reward = -1                    
        
        elif state == 14: #'extract tomato sauce fridge'
            action_robot = True
            if action == 16: #'bring tomato sauce'
                reward = positive_reward
            else: reward = -1
        
        elif state == 15: #'extract nutella fridge'
            action_robot = True
            if action == 10: #'bring nutella'
                reward = positive_reward
            else: reward = -1            
        
        elif state == 16: #'spread butter'
            if action == 18:
                reward = positive_reward
            else: reward = -1    
        
        elif state == 17: #'spread jam'
            if action == 18:
                reward = positive_reward
            else: reward = -1
        
        elif state == 18: #'spread tomato sauce'
            if action == 18:
                reward = positive_reward
            else: reward = -1                    
        
        elif state == 19: #'spread nutella'
            if action == 18:
                reward = positive_reward
            else: reward = -1
        
        elif state == 20: #'pour olive oil'
            if action == 18:
                reward = positive_reward
            else: reward = -1
            
        elif state == 21: #'put jam fridge'
            if action == 18:
                reward = positive_reward
            else: reward = -1    
        
        elif state == 22: #'put butter fridge'
            action_robot = True
            if action == 20:
                reward = positive_reward
            else: reward = -1
        
        elif state == 23: #'put tomato sauce fridge'
            action_robot = True
            if action == 21:
                reward = positive_reward
            else: reward = -1                    
        
        elif state == 24: #'put nutella fridge'
            action_robot = True
            if action == 22:
                reward = positive_reward
            else: reward = -1
        
        elif state == 25: #'pour milk bowl'
            if action == 18:
                reward = positive_reward
            else: reward = -1
            
        elif state == 26: #'pour cereals bowl'
            if action == 18:
                reward = positive_reward
            else: reward = -1    
        
        elif state == 27: #'pour nesquik bowl'
            if action == 18:
                reward = positive_reward
            else: reward = -1
        
        elif state == 28: #'put bowl microwave'
            if action == 18:
                reward = positive_reward
            else: reward = -1                    
        
        elif state == 29: #'stir spoon bowl'
            if action == 18:
                reward = positive_reward
            else: reward = -1
        
        elif state == 30: #'put milk fridge'
            action_robot = True
            if action == 23:
                reward = positive_reward
            else: reward = -1
            
        elif state == 31: #'put sliced bread plate'
            if action == 18:
                reward = positive_reward
            else: reward = -1    
        
        else:
            if action == 18: #'do nothing'
                reward = positive_reward
            else: reward = -1                    
        #------------------------------        
                
        self.total_reward += reward            
    
        return reward, action_robot

    
    def transition(self, freeze_state):
        """
        Gets a new observation of the environment based on the current frame and updates the state.
        
        Global variables:
            frame: current time step.
            action_idx: index of the NEXT ACTION (state as the predicted action). *The action_idx points towards the next atomic action at the current frame.
            annotations: pickle with the annotations, in the form of a table. 
        
        """
        
        global action_idx, frame, annotations, inaction
        

            
        frame += 1 #Update frame
        length = len(annotations['label']) - 1 #Length from 0 to L-1 (same as action_idx). "Duration of the recipe, in number of actions"
        
        #print("Frmae. ", frame, end='\r')
        
        
        # 1)
        #GET TIME STEP () (Updates the action_idx)
        #We transition to a new action index when we surpass the init frame of an action (so that we point towards the next one).    
        if freeze_state == False: 
            # frame >= annotations['frame_init'][action_idx]
            if  frame >= annotations['frame_init'][action_idx]:
                action_idx += 1
                inaction = []
        
        
        # 2) GET NA & AO FROM ANNOTATIONS        
        # Check if the action_idx is pointing towards nothing == the person is performing the last action of the recipe and there is no NEXT ACTION.
        if action_idx >= length+1: #If we finish, code TERMINAL STATE
            na = one_hot(-1, N_ATOMIC_ACTIONS) #Code a TERMINAL STATE as [0, 0, ..., 1]
            ao = np.zeros((N_OBJECTS)) #Code Active Object as zeros.
            ao_prev = ao
            na_prev = annotations['label'][action_idx-1]
            na_prev = one_hot(na_prev, N_ATOMIC_ACTIONS)
            var = 0.01
        # If we haven't finished the recipe, then we update the STATE by getting the NEXT ACTION and the ACTIVE OBJECT.
        else:
            #Generate a random number between 0 and 1. 
            p = random.uniform(0, 1)
            
            diff = (annotations['frame_init'][action_idx] - frame)/annotations['frame_init'][action_idx] 
            var = diff**3 #Noise variance
            
            # 5% chance of erroneously coding another action that does not correspond to the annotations.
            if p>0.97:
                na = random.randint(0, N_ATOMIC_ACTIONS-2) #Random label with 5% prob (from action 0 to N_AA-2, to avoid coding the TERMINAL STATE)
                na_prev = na
            # 95 % chance of coding the proper action
            else:
                na = annotations['label'][action_idx] #Correct label
                na_prev = annotations['label'][action_idx-1]
                
            na = one_hot(na, N_ATOMIC_ACTIONS) #From int to one-hot vector.
            na_prev = one_hot(na_prev, N_ATOMIC_ACTIONS)
            
            # Generate gaussian noise with 0 mean and 1 variance.
            noise = np.random.normal(0, 1, N_ATOMIC_ACTIONS)
            na_noisy = na + var*noise #Add the noise to the NEXT ACTION vector. The multiplication factor modulates the amplitude of the noise. Right now, 0.1 is not very aggresive but it is noticeable.
            na_norm = (na_noisy + abs(np.min(na_noisy))) / (np.max(na_noisy) - np.min(na_noisy)) #Normalize so that the vector represents the probability of each action. 
            na = na_norm / np.sum(na_norm)
            
            na_prev_noisy = na_prev + var*noise
            na_prev_norm = (na_prev_noisy + abs(np.min(na_prev_noisy))) / (np.max(na_prev_noisy) - np.min(na_prev_noisy)) 
            na_prev = na_prev_norm / np.sum(na_prev_norm)
            
            # This is an invented variable that codes the active objects of the NEXT ACTION.
            ao = np.zeros((N_OBJECTS))
            ao_prev = ao
            ao_idx = annotations['object_label'][action_idx] #Indices of the objects, for example [1, 4, 5] 
            ao_idx_prev = annotations['object_label'][action_idx-1] 
            if type(ao_idx) == int:
                pass
            else:
                for idx in ao_idx: # This generates different activation values, so that it is not coded with 0s and 1s.
                    ao = 0.6 * ao
                    ao[idx]=1 # In the example with [1, 4, 5], the AO would look like [0, 0.36, 0, 0, 0.6, 1, 0, ..., 0]
    
            if type(ao_idx_prev) == int:
                pass
            else:
                for idx in ao_idx_prev: # This generates different activation values, so that it is not coded with 0s and 1s.
                    ao_prev = 0.6 * ao_prev
                    ao_prev[idx]=1 # In the example with [1, 4, 5], the AO would look like [0, 0.36, 0, 0, 0.6, 1, 0, ..., 0]
    
        #Either take the NEXT ACTION as the STATE, or also concatenate the ACTIVE OBJECT
        if VERSION == 1:
            state = na
            prev_state = na_prev
        elif VERSION == 2:
            state = concat_vectors(na, ao)    
            prev_state = concat_vectors(na_prev, ao_prev) 
        
        self.state = state
        self.prev_state = prev_state

    
    def render(self, state, next_state, action, reward, total_reward):
        """
        Prints the environment.        
        Input:
            state: (numpy array) current state of the environment.
            next_state: (numpy array) state transitioned to after taking action.
            action: (int) action taken in current state.
            reward: (int) reward received in current state by taking action.
            total_reward: (int) cumulative reward of the episode.
        """
        if VERSION == 1:
            state = undo_one_hot(state)
            next_state = undo_one_hot(next_state)        
        elif VERSION == 2:
            na, vwm = undo_concat_state(state)
            next_na, next_vwm = undo_concat_state(next_state)
            state = undo_one_hot(na)
            next_state = undo_one_hot(next_na)
        
        #Numerical version
        #print('| State: {:>3g}'.format(state), ' | Action: {:>3g}'.format(action), ' | New state: {:>3g}'.format(next_state), ' | Reward {:>3g}'.format(reward), ' | Total reward {:>3g}'.format(total_reward), ' |')
        #print('='*82)
        
        #Version with dictionary meanings (provided that the state/action spaces are shorter than or equal to the dictionary.
        print('| STATE: {0:>29s}'.format(self.next_atomic_action_repertoire[state]), ' | ACTION: {0:>20s}'.format(self.action_repertoire[action]), ' | NEW STATE: {0:>29s}'.format(self.next_atomic_action_repertoire[next_state]), ' | REWARD {:>3g}'.format(reward), ' | TOTAL REWARD {:>3g}'.format(total_reward), ' |')
        #print('='*151)


        
        
    def summary_of_actions(self):
    	print("\nCORRECT ACTIONS (in time): ", self.CA_intime)
    	print("CORRECT ACTIONS (late): ", self.CA_late)
    	print("INCORRECT ACTIONS (in time): ", self.IA_intime)
    	print("INCORRECT ACTIONS (late): ", self.IA_late)
    	print("UNNECESSARY ACTIONS (in time): ", self.UA_intime)
    	print("UNNECESSARY ACTIONS (late): ", self.UA_late)
    	print("CORRECT INACTIONS: ", self.CI)
    	print("INCORRECT INACTIONS: ", self.II)
    	print("")
    
    def close(self):
        pass
