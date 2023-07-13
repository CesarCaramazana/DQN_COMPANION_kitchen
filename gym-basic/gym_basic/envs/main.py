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
import random
from collections import Counter
import copy 
from natsort import natsorted, ns
import pickle


#random.seed(42)
SEED = 48

#CONFIGURATION GLOBAL ENVIRONMENT VARIABLES
# 1) Dimensionality of input variables
ACTION_SPACE = cfg.ACTION_SPACE
N_ATOMIC_ACTIONS = cfg.N_ATOMIC_ACTIONS
N_OBJECTS = cfg.N_OBJECTS
VWM = (N_OBJECTS-1)*2
TEMPORAL_CTX = ACTION_SPACE +1

ATOMIC_ACTIONS_MEANINGS = cfg.ATOMIC_ACTIONS_MEANINGS
OBJECTS_MEANINGS = cfg.OBJECTS_MEANINGS
ROBOT_ACTIONS_MEANINGS = copy.deepcopy(cfg.ROBOT_ACTIONS_MEANINGS)
ROBOT_ACTION_DURATIONS = cfg.ROBOT_ACTION_DURATIONS
ROBOT_POSSIBLE_INIT_ACTIONS = cfg.ROBOT_POSSIBLE_INIT_ACTIONS 
OBJECTS_INIT_STATE = copy.deepcopy(cfg.OBJECTS_INIT_STATE)

VERSION = cfg.VERSION
POSITIVE_REWARD = cfg.POSITIVE_REWARD

Z_hidden_state = cfg.Z_hidden_state
Z_HIDDEN = cfg.Z_HIDDEN

INTERACTIVE_OBJECTS_ROBOT = cfg.INTERACTIVE_OBJECTS_ROBOT

#ANNOTATION-RELATED VARIABLES

#root_realData = "./video_annotations/Real_data/fold1/train/*" #!
root_realData = "./video_annotations/5folds/"+cfg.TEST_FOLD+"/train/*" 

#List of videos
videos_realData = glob.glob(root_realData) #Folders

random.shuffle(videos_realData)

total_videos = len(videos_realData)


video_idx = 0 #Index of current video
action_idx = 0 #Index of next_action
frame = 0 #Current frame
recipe = ''

correct_action = -1 # esto para que es

# labels_pkl = 'labels_updated.pkl'
# path_labels_pkl = os.path.join(videos_realData[video_idx], labels_pkl)

# annotations = np.load(path_labels_pkl, allow_pickle=True)




class BasicEnv(gym.Env):
    message = "Custom environment for recipe preparation scenario."      

    
    def __init__(self, display=False, test=False, debug=False):
        """        
        Initializes the environment.
            - Sets the dimensionality of the input and output space.
            - Sets the initial values of all the variables.
            - Sets the dataset
        
        Args:
            display: (bool) if True, display information.
            test: (bool) if True, the test set will be used instead of the training.


        """
        self.action_space = gym.spaces.Discrete(ACTION_SPACE) #[0, ACTION_SPACE-1]
        
        print("ACTION_SPACE: ", ACTION_SPACE)
        print("N_OBJECTS: ", N_OBJECTS)
        print("N_ATOMIC_ACTIONS: ", N_ATOMIC_ACTIONS)
        print("Z HIDDEN: ", Z_HIDDEN)
        
        
        if cfg.TEMPORAL_CONTEXT:
            if Z_hidden_state:
                self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS*2 + VWM + N_OBJECTS + TEMPORAL_CTX + Z_HIDDEN) #[Ac pred + Ac reg + VWM + OiT + Ac duration + Z]
            else:
                self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS*2 + VWM + N_OBJECTS + TEMPORAL_CTX) #[Ac pred + Ac reg + VWM + OiT + Ac duration]
        else:
            if Z_hidden_state:
                self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS*2 + VWM + N_OBJECTS + Z_HIDDEN) #[Ac pred + Ac reg + VWM + OiT + Z]
            else:
                self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS*2 + VWM + N_OBJECTS) #[Ac pred + Ac reg + VWM + OiT]

            
        self.state = []               
        
        self.action_repertoire = ROBOT_ACTIONS_MEANINGS
        self.next_atomic_action_repertoire = ATOMIC_ACTIONS_MEANINGS
        
        self.display = display
        
        self.test = test
        
        self.flags = {'freeze state': False, 'decision': False, 'threshold': " ",'evaluation': "Not evaluated", 'action robot': False,'break':False,'pdb': False}
        
        self.person_state = "other manipulation"
        self.robot_state = "idle"
        
        self.total_reward = 0
        self.reward_energy = 0
        self.reward_time = 0
        
        self.time_execution = 0
        self.mode = 'train'
        
        self.objects_in_table = OBJECTS_INIT_STATE.copy()
        
        global root, videos_realData, total_videos, annotations
        
        if self.test:
            random.seed(SEED)
            print("==== TEST SET ====")
            if debug:
                print("== Debug version!")
                root_realData = "./video_annotations/5folds/" + cfg.TEST_FOLD + "/test_debug/*"
            else:
                root_realData = "./video_annotations/5folds/"+cfg.TEST_FOLD+"/test/*" 
                
            videos_realData = glob.glob(root_realData) 
        
        else:
            print("==== TRAIN SET ====")
            root_realData = "./video_annotations/5folds/"+cfg.TEST_FOLD+"/train/*" 
            videos_realData = glob.glob(root_realData) 
            random.shuffle(videos_realData)
        
        print("List of videos\n", *videos_realData, sep='\n')       
        
        total_videos = len(videos_realData)
        
        labels_pkl = 'labels_updated.pkl'
        path_labels_pkl = os.path.join(videos_realData[video_idx], labels_pkl)
        
        self.video_ID = str(path_labels_pkl.split("/")[-2]) #Title of the video

        annotations = np.load(path_labels_pkl, allow_pickle=True)
        
        remaining_frames_pkl = os.path.join(videos_realData[video_idx], 'remaining_frames')
        self.remaining_frames = np.load(remaining_frames_pkl, allow_pickle=True).numpy().squeeze()
        
        print(annotations)
        
        self.CA_intime = 0
        self.CA_late = 0
        self.IA_intime = 0
        self.IA_late = 0 
        self.UAC_intime = 0
        self.UAC_late = 0
        self.UAI_intime = 0
        self.UAI_late = 0
        self.CI = 0
        self.II = 0
        
        self.UA_related = 0
        self.UA_unrelated = 0
        
        self.prediction_error = 0
        self.total_prediction = 0
        
        self.r_history = []
        self.h_history = []
        self.rwd_history = []
        self.rwd_time_h = []
        self.rwd_energy_h = []
        
        
        self.duration_action = 0
        
        #RRRRRRRRRRRRRR
        self.action_repertoire_durations = [[] for x in range(cfg.ACTION_SPACE)] #Empty list of lists
        self.idles = 0
        self.idles_list = []
        self.anticipation = 0

        
    def get_frame(self):
        global frame
        return frame 
    def get_action_idx(self):
        global action_idx
        return action_idx
    def get_annotations(self):
        global annotations
        return annotations
    
    def get_action_meanings(self):
        return self.action_repertoire
    def get_state_meanings(self):
        return self.state_repertoire
    
   
    def energy_robot_reward(self, action):
        """
        Establish the energy consumption of an action proportional to its duration.
        
        Args:
            action: (int) robot action.
            
        """        
         
        self.reward_energy = -self.duration_action * cfg.FACTOR_ENERGY_PENALTY
    
    def get_possibility_objects_in_table (self):
        global annotations
        # el update de los objetos de la mesa se hace antes de que pase, cambiarlo es un rollo
        # memory_objects_table[-1]
        # si en frame, el (o los) objeto(s) esta(n) o no en la mesa, genial.
        
        # 1 - saber con que objetos interatua el robot en el video 
        # 2 - cuando esta bien que este en la mesa o no 
        
        # realmente podemos inferir que accion esta haciendo viendo el historial, 
        # como se actualiza antes de que pase, ya sabemos que objeto cambia 
        
        # puedo hacer una tabla con, objeto, cuando puede o no estar en la mesa (0, 1),
        # fr init, fr end
        
        # para eso tengo que coger de las anotaciones, las acciones que involucren al robot, 
        # y el fr_init (que pasa a ser el end) y el fr_end de la primera accion pasa a ser el init de la otra
        # si no hay acciones antes que involucren un mismo objeto, se pone desde 0 el fr init
        # print("?) Entro en get_possibility_objects_in_table")
        person_states = annotations['label']

        objects_video = []
        in_table_video = []
        fr_init_video = []
        fr_end_video = []
        index_annotation = []
        
        for idx,value in enumerate(person_states):
            for obj in INTERACTIVE_OBJECTS_ROBOT:
                if obj in ATOMIC_ACTIONS_MEANINGS[value]: 
                    if 'extract' in ATOMIC_ACTIONS_MEANINGS[value]:
                        objects_video.append(obj)
                        in_table_video.append(1)
                        fr_init_video.append(0)
                        fr_end_video.append(annotations['frame_init'][idx])
                        index_annotation.append(idx)
                        
        video_bring = {"Object": objects_video, "In table": in_table_video, "Frame init": fr_init_video,"Frame end": fr_end_video, "Index": index_annotation}
        
        df_bring = pd.DataFrame(video_bring)            
        # tengo que cerar 
        for idx,value in enumerate(person_states):
            for obj in INTERACTIVE_OBJECTS_ROBOT: 
                if obj in ATOMIC_ACTIONS_MEANINGS[value]:
                    if 'put' in ATOMIC_ACTIONS_MEANINGS[value]: 
                        objects_video.append(obj)
                        in_table_video.append(0)
                       
                        df_current_object=df_bring[df_bring["Object"] == obj]
                        # este caso es distinto
                        fr_init_video.append(annotations['frame_end'][int(df_current_object['Index'])])
                        fr_end_video.append(annotations['frame_init'][idx])
        
        data_video =  {"Object": objects_video, "In table": in_table_video, "Frame init": fr_init_video,"Frame end": fr_end_video}
        df_video = pd.DataFrame(data_video)  
        person_states_print = []
        for idx,val in enumerate(person_states):
            person_states_print.append(ATOMIC_ACTIONS_MEANINGS[val])

        return df_video
       
        
    def get_minimum_execution_times(self):
        
        global annotations
        df_video = self.get_possibility_objects_in_table()
        
        # print("?) Entro en get_minimum_execution")
        
        name_actions = []
        for index, row in df_video.iterrows():
            if row['In table'] == 1:
                name_actions.append("bring "+row['Object'])
            else:
                name_actions.append("put "+row['Object']+ ' fridge')
            
        df_video['Actions'] = name_actions
        
        keys = list(df_video['Actions'])
        video_dict = {}
        for i in keys:
            video_dict[i] = 0
            
        person_states = annotations['label']
        
       
        total_minimum_time_execution = annotations['frame_end'][len(annotations)-1]
        # total_minimum_time_execution = 0
        df_video_dataset = df_video.copy()
        # df_video_dataset = df_video
        df_video_dataset['Max_time_to_save'] = [0]*len(df_video_dataset)
        
        if not df_video_dataset.empty:
            # print(df_video)
            
            for idx,value in enumerate(person_states):
                for obj in INTERACTIVE_OBJECTS_ROBOT:
                    if obj in ATOMIC_ACTIONS_MEANINGS[value]: 
                        
                        if 'extract' in ATOMIC_ACTIONS_MEANINGS[value]:
                            if idx != 0: 
                                action_name = 'bring '+ obj
                                fr_init = annotations['frame_init'][idx-1]
                                
                                df_video_dataset['Frame init'].loc[df_video_dataset['Actions']==action_name] = fr_init
                                keys = [k for k, v in ROBOT_ACTIONS_MEANINGS.items() if v == action_name]
                                df_video_dataset['Max_time_to_save'].loc[df_video_dataset['Actions']==action_name] = annotations['frame_end'][idx] - annotations['frame_end'][idx-1] 

                        elif 'put' in ATOMIC_ACTIONS_MEANINGS[value]: 
                            action_name = 'put '+ obj +' fridge'
                            fr_init = annotations['frame_init'][idx-1]
                            # df_selected = df_video_dataset.loc[df_video_dataset['Actions']==action_name]
                            df_video_dataset['Frame init'].loc[df_video_dataset['Actions']==action_name] = fr_init
                            keys = [k for k, v in ROBOT_ACTIONS_MEANINGS.items() if v == action_name]
                            
                            df_video_dataset['Max_time_to_save'].loc[df_video_dataset['Actions']==action_name] = annotations['frame_end'][idx] - annotations['frame_end'][idx-1] 
                            # df_video_dataset['Max_time_to_save'].loc[df_video_dataset['Actions']==action_name] = annotations['frame_end'][idx] - annotations['frame_end'][idx-1] - ROBOT_ACTION_DURATIONS[keys[0]]
                 
                
                
            df_video_dataset.sort_values("Frame init")
            # print(df_video_dataset)
            total_minimum_time_execution = annotations['frame_end'][len(annotations)-1] - df_video_dataset['Max_time_to_save'].sum()
        return total_minimum_time_execution,df_video_dataset
    
    def get_energy_robot_reward(self,action):
        """
        Get the energy cost (reward_energy) of an action based on its duration and on its correctness.
        Checks if an action is useful (either short or long-term) and sets its energy accordingly (useful actions are not penalized).
        
        Args:
            action: (int) action taken by the robot.        
        
        """
        global memory_objects_in_table, frame
        
        df_video = self.get_possibility_objects_in_table()
        energy_reward = 0

        variations_in_table = len(memory_objects_in_table)
        if variations_in_table < 2:
            oit_prev = memory_objects_in_table[0]
            oit = memory_objects_in_table[0]
        else:
            oit_prev = memory_objects_in_table[variations_in_table-2]
            oit = memory_objects_in_table[variations_in_table-1]

        objects_prev_print = []
        for key,value in OBJECTS_MEANINGS.items():
            if oit_prev[key] == 1: 
                objects_prev_print.append(value)
            
        objects_print = []
        for key,value in OBJECTS_MEANINGS.items():
            if oit[key] == 1: 
                objects_print.append(value)
                
        set1 = set(objects_prev_print)
        set2 = set(objects_print)
        
        missing = list(sorted(set1 - set2))
        added = list(sorted(set2 - set1))
        
        if len(missing) > 0:
            # print("PUT ")
            if (df_video['Object'] == missing[0]).any():
                df = df_video.loc[df_video['Object'] == missing[0]]
                if (df['In table'] == 0).any():
                    df_ = df.loc[df['In table'] == 0]
                    fr_init = int(df_['Frame init'])
                    fr_end = int(df_['Frame end'])
                    if fr_init < frame < fr_end:
                        energy_reward = 0
                    else:
                        energy_reward = 1
                else:
                    energy_reward = 1
            else:
                 energy_reward = 1
            
        elif len(added) > 0:            
            # print("BRING ")
            if (df_video['Object'] == added[0]).any():
                df = df_video.loc[df_video['Object'] == added[0]]
                if (df['In table'] == 1).any():
                    df_ = df.loc[df['In table'] == 1]
                    fr_init = int(df_['Frame init'])
                    fr_end = int(df_['Frame end'])
                    
                    if fr_init < frame < fr_end:
                        energy_reward = 0
                    else:
                        energy_reward = 1
                else:
                    energy_reward = 1
            else:
                 energy_reward = 1

        #GET THE ENERGY COST OF THE ACTION AS A FUNCTION OF ITS DURATION
        self.energy_robot_reward(action)
        
        #OBTAIN THE FINAL ENERGY COST DEPENDING ON ITS USEFULNESS
        self.reward_energy = energy_reward * self.reward_energy
        

        
    def update_objects_in_table (self, action):
        """
        Updates the objects in the table according to the robot's action, as it can bring (and put away)* objects.
        
        Args:
            action: (int) robot action.


        *Putting away objects was deprecated.
        
        """
        
        # print("?) Entro en update_objects_in_table")
        meaning_action = ROBOT_ACTIONS_MEANINGS.copy()[action]
        
        # print("Action: ", action)
        # print(meaning_action)
        for obj in INTERACTIVE_OBJECTS_ROBOT:
            if obj in meaning_action:
                
                if 'bring' in meaning_action:
                    self.objects_in_table[obj] = 1
                elif 'put' in meaning_action:
                    self.objects_in_table[obj] = 0
                    
        # pdb.set_trace()
        
        
        
    def possible_actions_taken_robot (self):
        """
        Calculates the possible actions for the robot to perform based on the objects at the table.
        The robot cannot bring an object twice if the object is already at the table.
        
        Returns:
            possible_actions: (list) list of the same length as the robot action repertoire with 1s on the indices of possible actions.
            
        """

        # print("?) Entro en possible_actions_taken_robot")
       
        bring_actions = []
        for x in INTERACTIVE_OBJECTS_ROBOT:
            bring_actions.append(''.join('bring '+ x))
            
        put_actions = []
        for x in INTERACTIVE_OBJECTS_ROBOT:
            put_actions.append(''.join('put '+ x + ' fridge'))
            
        position_bring_actions = [ele for ele in [key for key,value in ROBOT_ACTIONS_MEANINGS.copy().items() if value in bring_actions]]
        order_objects_bring = [ele for ele in [value for key,value in ROBOT_ACTIONS_MEANINGS.copy().items() if value in bring_actions]]
        position_put_actions = [ele for ele in [key for key,value in ROBOT_ACTIONS_MEANINGS.copy().items() if value in put_actions]]
        order_objects_put = [ele for ele in [value for key,value in ROBOT_ACTIONS_MEANINGS.copy().items() if value in put_actions]]
      
        dict_bring = {}
        for idx,value in enumerate(order_objects_bring):
            for obj in INTERACTIVE_OBJECTS_ROBOT:
                if obj in value: 
                    dict_bring[obj] = position_bring_actions[idx]
            
        dict_put = {}
        for idx,value in enumerate(order_objects_put):
            for obj in INTERACTIVE_OBJECTS_ROBOT:
                if obj in value: 
                    dict_put[obj] = position_put_actions[idx]
        
        # dict_put = {objects[i]: position_put_actions[i] for i in range(len(objects))}
        possible_actions = [0]*len(ROBOT_ACTIONS_MEANINGS)
        
        for key,value in ROBOT_ACTIONS_MEANINGS.copy().items():
            for obj in INTERACTIVE_OBJECTS_ROBOT: 
                if obj in value:
                    if self.objects_in_table[obj] == 0:
                        idx = dict_bring.copy()[obj]
                        possible_actions[idx] = 1
                    else:
                        try: 
                            idx_put = dict_put.copy()[obj]
                            possible_actions[idx_put] = 1
                        except: pass    
           
                   
        # print(self.objects_in_table)
        # print(dict_bring)
        # print(dict_put)
        possible_actions[5] = 1     
        # pdb.set_trace()
        
        
        #if possible_actions[0] == 1: 
        #    pdb.set_trace()
            
            
        return possible_actions

    def select_inaction_sample (self, inaction):
        """
        Gets the reward associated to one of the steps in which the robot stayed idle.
        
        Args:
            inaction: (list) list of samples in which the robot decided to 'do nothing' and their corresponding delayed rewards.
            
        Returns:
            reward: (int) single reward at a random position on the list.
        
        """

        random_position = random.randint(0,len(inaction)-1)
        
        self.state = inaction[random_position][1] # esto se hace para que el next_state sea el siguiente al guardado
        reward = inaction[random_position][2]
        
        return reward
        
    def select_correct_action (self): 
        """
        After an incorrect action was performed, get the correct action that the robot should've done.
        
        Args:
            action: (int)
            
        Returns:
            new_threshold: (int) ending frame of the corrected action.
            correct_action: (int) corrected action.
        
        """
                        
        global frame, annotations
        
        length = len(annotations['label']) -1 
        last_frame = int(annotations['frame_end'][length])
        
        for idx, val in cfg.ROBOT_ACTION_DURATIONS.items(): 
            reward = self.simple_reward(idx)
            if reward > -1 and idx!=5: 
                correct_action = idx
                duration_action = val
        

        new_threshold = duration_action + frame 
        if new_threshold > last_frame: 
            new_threshold = last_frame
            
        return new_threshold, correct_action
               
    def select_correct_action_video (self, action_idx): 
        
        global frame, annotations
        
        length = len(annotations['label']) -1 
        last_frame = int(annotations['frame_end'][length])
        
        real_state = annotations['label'][action_idx]
        
        for idx, val in cfg.ROBOT_ACTION_DURATIONS.items(): 
            reward = self.simple_reward(idx, real_state)
            if reward > -1: 
                correct_action = idx
                duration_action = val

        new_threshold = duration_action + frame 
        if new_threshold > last_frame: 
            new_threshold = last_frame
            
        return new_threshold, correct_action
    
    def update(self, update_type): 
        """
        Updates the development of the recipe as a consequence of the robot's actions.
        
        Args:
            update_type: (str) an update type from the possible: 'action', 'unnecesary', 
        """
        global frame, action_idx, inaction, annotations
        
        # print("?) Entro en update() -> type: ", update_type)
        
        length = len(annotations['label']) - 1 
        fr_init_next = int(annotations['frame_init'][action_idx])
        fr_end = int(annotations['frame_end'][action_idx-1])
        
        # If the current threshold is at the last frame of the recipe (thus, we end the recipe already)
        if self.flags['threshold'] == "last":           
            self.flags['break'] = True

        # Else, the normal case, in which there are still actions ahead.
        else:             

            if update_type == "action":
                 
                # frame = int(annotations['frame_end'][action_idx])
                # print("FRame in update: ", frame)
                # pdb.set_trace()
                if action_idx + 1 <= length:
                    action_idx = action_idx + 1
                    
                    # !! TO ACCOUNT FOR THE CASE IN WHICH TWO ACTIONS THAT REQUIRE AN OBJECT HAPPEN IN A ROW !! 12345
                    new_task = annotations['label'][action_idx] #This is the action the person jumps to. Check if this action is one that requires an object: 
                    # 8: 'extract milk', 12: 'extract jam', 13: 'extract butter', 14: 'extract tomato sauce', 15: 'extract nutella'                       
                    if (new_task == 8 and self.objects_in_table['milk'] == 1) or (new_task == 13 and self.objects_in_table['jam'] == 1) or (new_task == 12 and self.objects_in_table['butter'] == 1) or (new_task == 14 and self.objects_in_table['tomato sauce'] == 1) or (new_task == 15 and self.objects_in_table['nutella'] == 1):
                        action_idx = action_idx + 1
                    
                    frame = int(annotations['frame_end'][action_idx-1])+1 #At the end of the previous?
                    # frame = int(annotations['frame_init'][action_idx]) #Or at the beginning of the new one?

                    # pdb.set_trace()
                inaction = []
                    
            if update_type == "unnecesary":
                # pdb.set_trace()

                if self.flags['threshold'] == ('second' or 'next action init'): #?!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #if self.flags['threshold'] == 2 or self.flags['threshold'] == 4:    
                    frame = fr_init_next #+1
                    
                    if action_idx + 1 <= length:
                        action_idx = action_idx + 1
                
                #elif self.flags['threshold'] == 1 or self.flags['threshold'] == 3:
                elif self.flags['threshold'] == ('first' or 'next action'):
                    # print(self.flags)
                    # pdb.set_trace()
                    # print(frame)
                    # pdb.set_trace()
                    if frame > fr_end:
                        frame = fr_end #+1
                 
                inaction = []

    def time_course(self, action):
        """
        Determines the execution time of the action and its evaluation frame based on the human behavior.
        
        Args: 
            action: (int) action to be performed by the robot.
        
        Returns:
            threshold: (int) evaluation frame. It is either fr_execution or fr_end.
            fr_execution: (int) frame at which the robot finishes the action.
            fr_end: (int) frame at which the person finishes the action.
        
        """
        
        global frame, action_idx, inaction
        
        # print("2) Entro a time_course (con frame %5i)" %frame)
        # print("Entro a time_course con accion: ", action, " | Y el flag de decision: ", self.flags['decision'])
        
        
        
        # ==== 0 Check how many decisions are idle before an action is taken === 
        
        if (action==5) and (self.flags['decision']):
            self.idles += 1
        elif (action != 5) and (self.flags['decision']):
            # print("Idles before action: ", self.idles)
            self.idles_list.append(self.idles)
            self.idles = 0
                    
        #=======================================================================
        
        

        # ===== 1 GET ACTION DURATION ==========================================================================================
        # 1.1 Get the frame at which the robot will finish the action (fr_execution)
        if self.test: random.seed(SEED)
        sample = random.random()
        if sample < cfg.ERROR_PROB:            
            self.duration_action = int(random.gauss(1.5*cfg.ROBOT_ACTION_DURATIONS[int(action)], 0.2*cfg.ROBOT_ACTION_DURATIONS[int(action)]))
            fr_execution = self.duration_action + frame
            
        else:            
            self.duration_action = int(random.gauss(cfg.ROBOT_ACTION_DURATIONS[int(action)], 0.2*cfg.ROBOT_ACTION_DURATIONS[int(action)]))
            fr_execution = self.duration_action + frame       
            
        
        #RRRRRRRRRRRRRRRRRRRRRRRRRRR    
        # 1.2 Save sampled duration in the list of lists
        if action != 5: 
            #self.action_repertoire_durations[action].append(self.duration_action)   
            
            #Running mean
            self.action_repertoire_durations[action] = 0.95 * self.action_repertoire_durations[action] + (1-0.95)*self.duration_action
            # print("DURATION: ", self.duration_action)
        
        # =========================================================================================================================
    

        #print("In time course:\n", annotations)
        #print("Action idx: ", action_idx)
        
        # =============== 2 Possible EVALUATION FRAMES: person's beginning or end frames   ========================================
        # =========================================================================================================================
        last_frame = int(annotations['frame_end'].iloc[-1]) #5) The last frame of the recipe
        fr_end = int(annotations['frame_end'][action_idx-1]) # 1) End of the person's current action
        fr_init_next = int(annotations['frame_init'][action_idx]) # 2) Beginning of the person's next action
        fr_end_next = int(annotations['frame_end'][action_idx]) #3) End of the person's next action
        try:
            fr_init_next_next = int(annotations['frame_init'][action_idx+1]) # 4) Beginning of the two-steps-ahead action (if it exists)
            fr_end_next_next = int(annotations['frame_end'][action_idx+1])
            # if fr_init_next_next == fr_end_next:
            #     fr_init_next_next += 10
        except:
            fr_init_next_next = last_frame  #5) The last frame 
            fr_end_next_next = last_frame
        
        # print("CURRENT FRAME: ", frame)
        # print("1) Fr end: ", fr_end)
        # print("2) Fr init next: ", fr_init_next)
        # print("3) Fr end next: ", fr_end_next)
        # print("4) Fr init next next: ", fr_init_next_next)
        # print("4.2) Fr end next next: ", fr_end_next_next)
        # print("5) Last frame: ", last_frame)
        # print("Fr execution (cuando el robot acaba): ", fr_execution)
        
        # if fr_init_next == fr_end: fr_init_next += 10
        # if fr_init_next_next == fr_end_next: fr_init_next_next += 10
        
        # =========================================================================================================================        
   
        if self.flags['decision']:
            self.flags['freeze state'] = True
        else:
            self.flags['freeze state'] = False
                     
        #print("Freeze state: ", self.flags['freeze state'])
        self.flags['threshold'] = ''
        # pdb.set_trace()
        
        #If the robot has taken an action -------------------------------------
        if action !=5: 
            _,df_video = self.get_minimum_execution_times()     
            #filtro para quedarme solo con las acciones que realmente se pueden hacer 
            df_video_filtered = df_video[df_video['In table']==1]   
            
            # print("Df filtered: ", df_video_filtered)
            
            # pdb.set_trace()
            
            #5) If the robot execution frame surpasses the last frame of the recipe, hard-limit the EVALUATION to the last frame
            if fr_execution > last_frame: 
                threshold = last_frame 
                fr_execution = last_frame 
                self.flags['threshold'] = 'last'
            
            #1) If the robot execution frame is earlier than the person's current action, the EVALUATION happens when the person finishes.
            if fr_execution < fr_end:
                    threshold = fr_end
                    #self.flags['threshold'] = 1 #"first"
                    self.flags['threshold'] = 'first'
            
            
            # If the robot finishes later than the person's current action, there are some possibilities:
            # -- The person's NEXT ACTION does not require an object, so we can relax the EVALUATION frame.
            # -- The person's NEXT ACTION requires an object, so we cannot relax the EVALUATION frame.
            else:
                # _,df_video = self.get_minimum_execution_times()                
                # df_video_filtered = df_video[df_video['In table']==1]                
                
                increase_threshold = True
                
                # print("Self objects in table: ", self.objects_in_table)
                

                for index, row in df_video_filtered.iterrows():
                    # print("Row object: ", row['Object'])   
                    # print("Frmae Ini: ", row['Frame init'])
                    # print("Frame end: ", row['Frame end'])
                    # pdb.set_trace()
                    
                    if row['Frame init'] <= frame < row['Frame end']: #??????? Esto estaba antes
                    # if frame < row['Frame end']:    #Esto lo he cambiado
                        # print("Row object: ", row['Object'])   
                        if self.objects_in_table[row['Object']] != row['In table']:  # nunca llega a hacer la 1 accion, que no se pare
                            if row['Frame end'] != int(annotations['frame_init'][0]):
                                increase_threshold = False
                                # print("No se puede")
                                # pdb.set_trace()
                    if row['Frame init'] <= fr_execution < row['Frame end']: #?????????? Esto estaba antes
                    # if fr_execution < row['Frame end']:     
                        if self.objects_in_table[row['Object']] != row['In table']: # nunca llega a hacer la 1 accion, que no se pare
                            if row['Frame end'] != int(annotations['frame_init'][0]):
                                # print("No se puede")
                                # pdb.set_trace()
                                increase_threshold = False
                               
                    if fr_execution >= row['Frame end']:
                        if self.objects_in_table[row['Object']] != row['In table']: # nunca llega a hacer la 1 accion, que no se pare
                            if row['Frame end'] != int(annotations['frame_init'][0]):
                                increase_threshold = False


                # WE CAN INCREASE THE EVALUATION FRAME. How much?
                # =============================================================================================
                if increase_threshold:
                    self.flags['freeze state'] = False #??? Do we freeze the state if the threshold is increased?
                    
                    # 2) Until the beginning of the NEXT ACTION    - · - · - · - · - · - · - · - · - · - · - · - · - · -                 
                    if fr_execution <= fr_init_next: 
                        threshold = fr_init_next
                        fr_end = fr_init_next
                        #self.flags['threshold'] = 2 #"second"
                        self.flags['threshold'] = 'second'
                        # self.flags['freeze state'] = False #????
                        # print("Second")
                        # pdb.set_trace()
                        
                    # 3) Until the end of the NEXT ACTION    - · - · - · - · - · - · - · - · - · - · - · - · - · - 
                    elif fr_init_next < fr_execution <= fr_end_next:
                        threshold = fr_end_next
                        fr_end = fr_end_next #?
                        
                        #self.flags['threshold'] = 3 #"next action"?
                        self.flags['threshold'] = 'next action'
                        # self.flags['freeze state'] = False #?
                        # print("Next action")
                        # pdb.set_trace()
                    
                    # 4) Until the beginning of the FOLLOWING TO NEXT ACTION  - · - · - · - · - · - · - · - · - · - · - · - · - · - 
                    elif fr_end_next < fr_execution <= fr_init_next_next:
                    # else:
                        threshold = fr_init_next_next
                        fr_end = fr_init_next_next #?                        
                        self.flags['threshold'] = 'next action init'
                        # self.flags['freeze state'] = False #??????
                        # print("Next action init")
                        # pdb.set_trace()
                        
                    # 5) Until the end of the FOLLOWING TO NEXT ACTION - · - · - · - · - · - · - · - · - · - · - · - · - ·
                    elif fr_init_next_next < fr_execution <= fr_end_next_next:
                        threshold = fr_end_next_next
                        fr_end = fr_end_next_next
                        self.flags['threshold'] = 'next action init'
                        # print("Next action end")
                        # pdb.set_trace()
                    
                    else:
                        threshold = fr_end_next_next
                        fr_end = fr_end_next_next
                        self.flags['threshold'] = 'next action init' #?

                    
                    # elif fr_end_next < fr_execution <= fr_init_next_next:
                    #     threshold = fr_init_next_next
                    #     fr_end = fr_init_next_next
                        
                    #     self.flags['threshold'] = 4
                    #     self.flags['freeze state'] = False #?
                      
                    
                
                # WE CANNOT INCREASE THE THRESHOLD so the EVALUATION frame is hard-limited (time penalty incoming) =========================
                else:
                    # print("NO SE INCREMENTA")
                    # pdb.set_trace()
                    # self.flags['freeze state'] = True
                    # threshold = fr_init_next
                    # self.flags['threshold'] = 2
                    # self.flags['threshold'] = 'second' #¿¿¿¿¿
                    
                    
                    #Si estamos en ese else es porque fr_execution > fr_end, sino no se hubiera comprobado si hacía falta incrementar el threshold
                    
                    if frame < fr_end:                        
                        threshold = max(fr_execution, fr_end)
                        self.flags['threshold'] = 'first' #1
                    else:
                        threshold = max(fr_execution, fr_init_next)
                        fr_end = fr_init_next
                        
                        self.flags['threshold'] = 'second' #2
                          

        # Else: the robot does not take an action. Stays 'idle'/'do nothing'       
        else: 
            if frame == fr_end - 1 or frame == fr_init_next - 1: 
                if len(inaction) > 0:
                    if "action" not in inaction:
                        # flag_decision = True
                        self.flags['decision'] = True
                        self.flags['freeze state'] = True

                # Before the beginning of the next action
                if frame == fr_init_next - 1: 
                    threshold = fr_init_next 
                    fr_end = fr_init_next
                    
                    #Set decision = True
                    #The last frame before an inaction is considered a decision (so we can always act in between actions even if the gap is smaller than the decision rate).
                    self.flags['decision'] = True
                    self.flags['threshold'] = 'second'
                else:
                    threshold = fr_end 
                    self.flags['threshold'] = 'first'
            else: 
                
                threshold = frame
                # if len(inaction) > 0:
                    # if "action" not in inaction:
                    #     print("Times no action selected: ", len(inaction))

        
        # print("\nAT THE END THE THRESHOLD IS SET AT: ", threshold)
        # print("Salgo de time course con threshold %5i y frame %5i " %(threshold, frame))
        # print("salgo de time course con freeze state: ", self.flags['freeze state'])        
        # print("Which corresponds to case: ", self.flags['threshold'])
        # print("DECISION FLAG: ", self.flags['decision'])
        
        return threshold, fr_execution, fr_end 
    
    def evaluation(self, action, fr_execution, fr_end, frame_post):
        """
        Args:
            action: (int) action performed by the robot.
            fr_execution: (int) frame in which the robot finishes the action.
            fr_end: (int) frame in which the person finishes the current action. #######¿¿¿¿¿¿¿¿¿ Shouldn't this be 'threshold'?
            frame_post: (list)
        
        Returns:
            reward: (int) reward associated to action. 
            new_threshold: (int) updated EVALUATION frame in case the robot needs to perform a correction.
            optim: (bool) if True, the state-action-reward tuple will be used to optimize the DQN.
            frame_post: (list)
            correct_action: (int) updated correct action, simulating an ASR command from the person.        
        
        """
        global frame, action_idx, inaction, new_energy, correct_action, recipe
        
        # print("3)Entro a evaluation() |||| Frame: ", frame, "||| Exe: ", fr_execution, " ||| End: ", fr_end)
        # pdb.set_trace()
        # print("Y los objetos en la mesa: ", self.objects_in_table)
        
        optim = True        
        #Obtain the simple_reward factor as the correspondence between state-action (regardless of time delays)
        simple_reward = self.simple_reward(action)
        
        #Initialize new_threshold and reward
        new_threshold = 0 
        reward = 0
        
        # print("and the flag evaluation is: ", self.flags['evaluation'])
        
        #print("IN EVALUATION\nFrame: ", frame)
        if self.flags['evaluation'] == 'Correct Inaction':
            if frame == fr_execution: 
                #print("Frame: in ev: ", frame)
                reward = 0
                self.CI += 1
                # pdb.set_trace()
                self.update("action")    
                self.flags['evaluation'] = 'Not evaluated'
                self.flags['break'] = True
                
        elif (self.flags['evaluation'] == 'Incorrect action' or self.flags['evaluation'] == "Incorrect inaction"):               
            
            if frame == fr_execution: 
                # if self.flags['evaluation'] == 'Incorrect inaction':
                    # print('Incorrect inaction')
                    # print('new thr: ', fr_execution)
                    # pdb.set_trace()
                self.flags['evaluation'] = 'Not evaluated'
                frame_post.append(frame)
                self.update("action")
                reward = self.reward_energy + self.reward_time
                self.flags['break'] = True
             
            else: 
                self.reward_time += -1
                    
        else: 
                # pdb.set_trace()

                if simple_reward == 5:
                    self.flags['evaluation'] = 'Correct Inaction'
                    
                    if self.flags['threshold'] == '':
                    #if self.flags['threshold'] == 0:    
                        new_threshold = fr_end

                #If the object is already at the table and does something else.        
                elif simple_reward == -5:                    
                    if fr_execution <= fr_end: 
                        
                        if frame == fr_execution:
                            frame_post.append(frame)
    
                        if frame == fr_end: 
                            
                            inaction.append("action")
                            #self.energy_robot_reward(action)
                            self.get_energy_robot_reward(action)
                            reward = self.reward_energy  
                            if reward == 0:
                                self.UAC_intime += 1
                            else:
                                self.UAI_intime += 1
                                
                                if recipe == 'c' or recipe == 'd':
                                    if action == 2: self.UA_related += 1
                                    else: self.UA_unrelated += 1
                                elif recipe == 't':
                                    if action in [0, 1, 3, 4]: self.UA_related += 1
                                    else: self.UA_unrelated += 1    
                                
                            self.update("action")
                    else: 
    
                        if frame > fr_end:
                            self.reward_time += -1
                          
                        if frame == fr_execution: 
                            # print("*************** UNNECESARY ACTION (late) ***************")
                            inaction.append("action")
                            frame_post.append(frame)
                            # self.energy_robot_reward(action)
                            self.get_energy_robot_reward(action)
                            reward =  self.reward_energy + self.reward_time
                            if self.reward_energy == 0:
                                self.UAC_late += 1
                            else:
                                self.UAI_late += 1                                
                                
                            self.update("action")
                            
                            self.flags['break'] = True
                
                
                
                else:  
                    if action !=5: 
                        
                        #  ============= CORRECT ACTION ======================
                        # ====================================================
                        if simple_reward > -1: 
                            # 1) In time
                            if fr_execution <= fr_end:                                
                                # pdb.set_trace()
                                if frame == fr_execution:                                     
                                    frame_post.append(frame)
                                    self.reward_energy = 0
                                    reward = 0
                                    
                                if frame == fr_end:                                
                                    self.CA_intime += 1
                                    # print("*************** CORRECT ACTION (in time) ***************")
                                    inaction.append("action") 
                                    self.update("action")
                                    # self.flags['pdb'] = True
                            # 2) Late
                            else: 
                                if frame == fr_execution: 
                                    # print("*************** CORRECT ACTION (late) ***************")
                                    self.CA_late += 1
                                    self.reward_time += -1
                                    
                                    inaction.append("action")
                                    frame_post.append(frame)
                                    self.update("action")   
                                    self.reward_energy = 0
                                    reward = self.reward_time 
                                   
                                    # self.flags['pdb'] = True
        
                                if frame >=  fr_end: 
                                
                                    self.reward_time += -1
                            
                        # =================== INCORRECT OR UNNECESSARY ACTIONS ==================================
                        # =======================================================================================
                        else: 
                            #self.flags['freeze state'] = True #??????????'
                            # print("Entro en INCORRECT OR UNNEC with simple reward: ", simple_reward)
                            # print("Y self flags action robot: ", self.flags["action robot"])
                            
                            # INCORRECT ACTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            if self.flags["action robot"] == True: 
                                # pdb.set_trace()

                                # 1) In time ----------------------------------------------------
                                if fr_execution <= fr_end: 
                                    if frame == fr_execution: 
                                        frame_post.append(frame)
                                        
                                    if frame == fr_end: 
                                        # print("*************** INCORRECT ACTION (in time) ***************")
                                        self.IA_intime += 1                                        
                                        inaction.append("action")
                                        
                                        #Get the correct action --------
                                        new_threshold, correct_action = self.select_correct_action()
                                        
                                        self.flags['evaluation'] = 'Incorrect action'
        
                                        #self.energy_robot_reward(action)
                                        self.get_energy_robot_reward(action)
                                        prev_energy = self.reward_energy
                                        self.energy_robot_reward(correct_action)
                                        self.reward_energy = self.reward_energy + prev_energy
                                        # self.flags['pdb'] = True
                                        
                                # 2) Late ----------------------------------------        
                                else: 
                                    # print("ENTRO EN INCORRECT LATE")
                                    if frame > fr_end:
                                        self.reward_time += -1
                                        
                                    if frame == fr_execution: 
                                        # print("*************** INCORRECT ACTION (late) ***************")
                                        self.IA_late += 1                                        
                                    
                                        frame_post.append(frame)
                                        new_threshold, correct_action = self.select_correct_action()                                        
                                                   
                                        # self.flags['pdb'] = True
                                        self.flags['evaluation'] = 'Incorrect action'
                                  
                                        self.get_energy_robot_reward(action)
                                        prev_energy = self.reward_energy
                                        self.energy_robot_reward(correct_action)
                                        self.reward_energy = self.reward_energy + prev_energy
        
                            # UNNECESARY ACTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            else:
                                # 1) In time ----------------------------------------------------
                                if fr_execution <= fr_end: 
                                    # pdb.set_trace()
                                    # if self.flags['threshold'] == 'last':
                                        
                                    #     print(frame)
                                    if frame == fr_execution:
                                        frame_post.append(frame)
        
                                    if frame == fr_end: 
                                        # print("*************** UNNECESARY ACTION ***************")                                        
                                        inaction.append("action")                                        
                                        #self.energy_robot_reward(action)
                                        self.get_energy_robot_reward(action)

                                        reward = self.reward_energy  
                                        if reward == 0:
                                            self.UAC_intime += 1
                                        else:
                                            self.UAI_intime += 1
                                            
                                            if recipe == 'c' or recipe == 'd':
                                                if action == 2: self.UA_related += 1
                                                else: self.UA_unrelated += 1
                                            elif recipe == 't':
                                                if action in [0, 1, 3, 4]: self.UA_related += 1
                                                else: self.UA_unrelated += 1
     
                                        self.update("unnecesary")
                                
                                # 2) Late -------------------------------------------------        
                                else: 
        
                                    if frame >= fr_end:
                                        self.reward_time += -1
                                      
                                    if frame == fr_execution:                                         
                                        # print("*************** UNNECESARY ACTION (late) ***************")
                                        inaction.append("action")
                                        frame_post.append(frame)
                                        #self.energy_robot_reward(action)
                                        self.get_energy_robot_reward(action)
                                        reward =  self.reward_energy + self.reward_time
                                        if self.reward_energy == 0:
                                            self.UAC_late += 1
                                        else:
                                            self.UAI_late += 1

                                            if recipe == 'c' or recipe == 'd':
                                                if action == 2: self.UA_related += 1
                                                else: self.UA_unrelated += 1
                                            elif recipe == 't':
                                                if action in [0, 1, 3, 4]: self.UA_related += 1
                                                else: self.UA_unrelated += 1
                                            
                                            
                                        if  self.flags['threshold'] == 'next action init': 
                                        #if self.flags['threshold'] == 4:
                                            #self.flags['threshold'] = 3
                                            self.flags['threshold'] = 'next action'
                                        self.update("unnecesary")
                                        
                                        self.flags['break'] = True
                                        # flag_break = True
                                # flag_pdb = True
                                   
                    # ELSE: 'IDLE'/'DO NOTHING'        
                    else:
                        inaction.append([action, self.state, reward])
                        if self.flags['decision'] == True:
                            if frame == fr_end: 
                                # CORRECT INACTION
                                if simple_reward > -1: 
                                    # print("*************** CORRECT INACTION ***************")
                                    self.CI += 1                                    
                                    reward = self.select_inaction_sample(inaction)
                          
                                # INCORRECT INACTION
                                else: 
                                    # print("*************** INCORRECT INACTION ***************")
                                    self.flags['freeze state'] = True
                                    self.II += 1
                                    new_threshold, correct_action = self.select_correct_action()
                                    self.energy_robot_reward(correct_action)
                                    reward = self.reward_energy
                                    self.flags['evaluation'] = 'Incorrect inaction'
                                    # pdb.set_trace()

                                frame_post.append(frame)        
        
                            else: 
                                optim = False
                        else: 
                            optim = False

        # print("Salgo de evaluation con reward: ", reward)
        # self.flags['freeze state'] = False

        return reward, new_threshold, optim, frame_post, correct_action
        
    def prints_terminal(self, action, frame_prev, frame_post, reward):
        """
        Prints the reward obtained for a given action.
        
        Args:
            action: (int) action executed by the robot.
            frame_prev: (int) start of the action.
            frame_post: (int) end of the action.
            reward: (int) obtained reward.
        
        """
        
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
        
        data_robot = {"Robot action":accion_robot, "Frame init": int(frame_prev), "Frame end": str(frame_post), "Reward": reward, "Reward time": self.reward_time, "Reward energy": self.reward_energy, "Time execution": self.time_execution}
        # pdb.set_trace()
        df_robot = pd.DataFrame(data_robot, index=[0])
        
        print("----------------------------------- Video -----------------------------------")
        print(df)
        print("\n----------------------------------- Robot -----------------------------------")
        print(df_robot)
        
    def prints_debug(self, action):
        global annotations, action_idx, memory_objects_in_table, frame
        
        person_states_index = annotations['label']
        fr_init = annotations['frame_init']
        fr_end = annotations['frame_end']
        
        person_states = []
        
        for idx,val in enumerate(person_states_index):
            person_states.append(ATOMIC_ACTIONS_MEANINGS[val])
        
        data = {"States": person_states, "Frame init": fr_init, "Frame end": fr_end}
        
        df = pd.DataFrame(data)
        state_prev = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
        state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]]
        
        variations_in_table = len(memory_objects_in_table)
        if variations_in_table < 2:
            oit_prev = memory_objects_in_table[0]
            oit = memory_objects_in_table[0]
        else:
            oit_prev = memory_objects_in_table[variations_in_table-1]
            oit = memory_objects_in_table[variations_in_table]
            
        objects_prev_print = []
        for key,value in OBJECTS_MEANINGS.items():
            if oit_prev[key] == 1: 
                objects_prev_print.append(value)
            
        objects_print = []
        for key,value in OBJECTS_MEANINGS.items():
            if oit[key] == 1: 
                objects_print.append(value)
                
        action_robot = ROBOT_ACTIONS_MEANINGS.copy()[action]
        
        set1 = set(objects_prev_print)
        set2 = set(objects_print)
        
        missing = list(sorted(set1 - set2))
        added = list(sorted(set2 - set1))
        
        if len(missing) > 0:
            change = missing[0]
        elif len(added) > 0:
            change = added[0]
        else:
            change = 'nothing'
        
        # pdb.set_trace()
                
        print("Frame: ", str(frame))
        # print("----------------------------------- Video -----------------------------------")
        # print(df)
        # print("-------- Pre State Video --------")
        print("     ",state_prev)
        print("\nOBJECTS IN TABLE:")
        for obj in objects_prev_print: 
            if obj == change and action !=5:
                print('----> ' + obj + ', ')
            else:
                print(obj + ', ')
        # print(*objects_prev_print, sep = ", ")
        print("\n ROBOT ACTION: \n", action_robot)
        print("-------- State Video --------")
        print("     ",state)
        print("OBJECTS IN TABLE:")
        for obj in objects_print: 
            if obj == change and action !=5:
            
                print('----> ' + obj + ', ')
            else:
                print(obj + ', ')
        # print(*objects_print,sep = ", ")


        
    def step(self, action_array):
        """
        *This is the key function in all the environment.*
        Transition from the current state (self.state) to the next one given an action. The order in which functions are called is: 1) time_course(), to get the execution time of the robot and the evaluation frame,
        2) evaluation(), to get the reward, 3) transition(), to update the temporal variables.
        
        Args:
            action_array: (list) tuple of 0: action taken by the agent, 1: decision flag (True if the robot took a decision), 2: mode 
            
        Returns: self.state, reward, done, optim,  self.flags['pdb'], self.reward_time, self.reward_energy, self.time_execution, action, self.flags['threshold'], self.prediction_error, self.total_prediction 
            state: (numpy array) state of the environment after executing the action.
            reward: (int) obtained reward signal.
            done: (bool) flag indicating that the recipe is finished.
            optim: (bool) flag indicating that the sample will be used to optimize the DQN.
            flags['pbd']: (bool)
            reward_time: (int) time penalty component in the reward signal.
            reward_energy: (int) energy penalty component in the reward signal.
            time_execution: (int)
            action: (int) executed action by the robot.
            flags['threshold']: (str) indication of the type of threshold that was established for evaluation, depending on the human needs.
            prediction_error: 
            total_prediction:
            
        """
        global frame, action_idx, annotations, inaction, memory_objects_in_table, correct_action, path_labels_pkl       

        # I. Unzip the action array into: action (int), decision (bool) and mode (str)
        #============================================================================================
        action = action_array[0]
        self.flags['decision'] = action_array[1]
        self.mode = action_array[2]
        assert self.action_space.contains(action)
        
        # II. Initialize reward, flags and other variables
        #===========================================================================================
        reward = 0
        self.reward_energy = 0
        self.reward_time = 0
        path_env = 0
        done = False
        optim = False
        
        self.flags['freeze state'] = False
        self.flags['pdb'] = False 
        self.flags['break'] = False
        self.flags['evaluation'] = 'Not evaluated' 
        self.flags['threshold'] = ''

        len_prev = 1
        
        min_time = 0 #Optimal time for recipe
        max_time = 0 #Human time without HRI
        hri_time = 0 #Time of HRI
        frames_waiting = 0 #Waiting time of the person
        
        frame_prev = frame #Save the frame at which we entered
        frame_post = []
        
        
        # print("\n=============================\nFrame: ", frame)
        # print("1)Entro en step()")
        # print("Y la acción es: ", action)
        # print("Los objetos en la mes: ", self.objects_in_table)       
        
        # if frame > 1100:
        #     pdb.set_trace()
        
        
        # III. Run the loop to obtain the reward and the transitioned state 
        #============================================================================================        
        
        # 1) TIME COURSE
        #Get the evaluation point (THRESHOLD) of the action that finishes at fr_execution ==============================================
        threshold, fr_execution, fr_end = self.time_course(action)
        
        # print("THRESHOLD: ", threshold)
        # print("FRE EXE: ", fr_execution)
        # print("FRE END: ", fr_end)
        # pdb.set_trace()
        
        
        duration_action = self.duration_action #Save the duration of the action in a local variable        
        
        if action !=5:
            self.update_objects_in_table(action)
            memory_objects_in_table.append(list(self.objects_in_table.values()))
            # print("Threshold: ",threshold)

        # If the person is performing the last action of the recipe
        # ======END OF THE RECIPE ==========================================================================================================
        # ==================================================================================================================================
        if frame >= annotations['frame_init'].iloc[-1]:  
            #...just let the person finish but do nothing ('idle')            
            while frame <= annotations['frame_end'].iloc[-1]:   
                if annotations['frame_init'][action_idx] <= frame <= annotations['frame_end'][action_idx]:
                    self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]]
                self.robot_state = "idle"                

                # 3) TRANSITION()  
                self.transition() 
                
                self.rwd_history.append([reward])
                self.h_history.append([self.person_state])
                self.r_history.append([self.robot_state])
                self.rwd_time_h.append([self.reward_time])
                self.rwd_energy_h.append([self.reward_energy])   
                 
            # execution_times.append(annotations['frame_end'].iloc[-1])
            hri_time = self.time_execution
            self.flags['pdb'] = True
            # pdb.set_trace()
                        
            next_state = self.state
            done = True
            # print("Time execution: ", self.time_execution)
            
        # ==================================================================================================================================    
    
  
        #Else: there are still actions left
        # ==================================================================================================================================
        else: 
            #UNTIL THE EVALUATION IS FINISHED       
            # * * * * * * * * * BEGIN LOOP * * * * * * * * * * * * * 
            while frame <= threshold:
                # print(frame)
                # print(ATOMIC_ACTIONS_MEANINGS[undo_one_hot(self.state[0:33])] )
                current_state = self.state #Current state
                
                # print("(Step) Decision flag: ", self.flags['decision'])
    
                # If the robot didn't take a decision (frames in between DECISION_RATE multiples)
                # I. =================================================================================================================================
                if self.flags['decision'] == False:   
                    # se transiciona de estado pero no se hace ninguna acción 
                    self.flags['freeze state'] = False
                    self.robot_state = "idle"               
                    
                    if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                        self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                    else:
                        self.person_state = "other manipulation"
                
                # Else: THE ROBOT TOOK A DECISION AND HAS TO BE EVALUATED   
                # II. =================================================================================================================================
                else:                     
                    optim = True
                    
                    # 2) EVALUATION ()
                    # ************ GET REWARD AND (POSSIBLE) EXTENDED THRESHOLD (due to incorrect actions)
                    reward, new_threshold, optim, frame_post, correct_action = self.evaluation(action, fr_execution, fr_end, frame_post)
                    # reward, new_threshold, optim, frame_post, correct_action = self.evaluation(action, fr_execution, threshold, frame_post)
                    
                        
                    # ************ THE THRESHOLD IS EXTENDED    
                    if new_threshold != 0: 
                        threshold = new_threshold
                        fr_execution = new_threshold
                        # pdb.set_trace()
                        if self.flags['evaluation'] != 'Correct Inaction': 
                            action = correct_action
                            self.update_objects_in_table(action)
                            memory_objects_in_table.append(list(self.objects_in_table.values()))
                            len_prev = 2
                    
                    # Get meaning of the robot action    
                    if action != 5:
                        self.robot_state = ROBOT_ACTIONS_MEANINGS[action]  
                    else: 
                        self.robot_state = "idle" #Change 'do nothing' to 'idle'
                    
                    # *. If the robot has finished before the person...
                    if fr_execution <= fr_end:                         
                        # Check what the person was doing
                        if annotations['frame_init'][action_idx-1] < frame <= annotations['frame_end'][action_idx-1]:
                            self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                        else:
                            self.person_state = "other manipulation"
                            #self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]]
                        
                        # Check if the robot is finished because it took an action or because it didn't.    
                        if frame > fr_execution: 
                            if action != 5: 
                                self.robot_state = "Waiting..."
                            else:
                                self.robot_state = "idle"
                            frames_waiting += 1
                    
                    # **. Else: the person finishes before the robot...
                    elif fr_execution > fr_end: 
                        # print("frame: ", frame)
                        # print("F exe: ", fr_execution)
                        # print("fr end: ", fr_end)
                        # The person has already finished -> wait for the robot
                        if frame >= fr_end: 
                            if action != 5:
                                self.person_state = "Waiting for robot..."
                            else:
                                self.person_state = "other manipulation"
                            # print("??????? nono?")
                            # pdb.set_trace()
                        
                        # The person has yet to finish -> do current action
                        elif frame < fr_end: 

                            if annotations['frame_init'][action_idx-1] < frame <= annotations['frame_end'][action_idx-1]:
                                self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                            else:
                                self.person_state = "other manipulation"
                                              

                
                if frame == threshold :
                    self.flags['freeze state']  = False
                    
                    
                # 3) TRANSITION ========================================================
                self.transition() #Transition to a new frame -> new action_idx                
                
                # Save immediate states and rewards for visualization
                self.rwd_history.append([reward])
                self.h_history.append([self.person_state])
                self.r_history.append([self.robot_state])
                self.rwd_time_h.append([self.reward_time])
                self.rwd_energy_h.append([self.reward_energy])                
                                
                # print("] In step's loop ", frame, "Human: ", self.person_state, " |  Robot: ", self.robot_state, " | Self decision flag: ", self.flags['decision'], " | Action idx: ", action_idx, " | Freeze: ", self.flags['freeze state'])
                       
                next_state = self.state
                
                #PRINT STATE-ACTION TRANSITION & REWARD
                if self.display: self.render(current_state, next_state, action, reward, self.total_reward)
                
                # Check if we have reached the last frame and set 'done' flag to True if that's the case
                if frame >= annotations['frame_end'].iloc[-1]:
                    done = True
                    
                if self.flags['break'] == True: 
                    break
            
            # * * * * * * * * END LOOP * * * * * * * * * * * * *     
        # ==================================================================================================================================        

        # If a decision has been made, the objects in the table may have been updated    
        if self.flags['decision'] == True:
            self.state[110:133] = memory_objects_in_table[len(memory_objects_in_table)-1]
    
        # if self.flags['pdb']:
        #     self.prints_terminal(action, frame_prev, frame_post, reward)
        #     # self.prints_debug(action)
        #     print(frame)
        #     pdb.set_trace()
             
        # if optim:
            # self.prints_terminal(action, frame_prev, frame_post, reward)

            
        #If we have finished the recipe, save the history of states-actions-rewards
        if done:
            self.save_history()
            
            #SAVE MIN and MAX times of HRI (oracle and reactive)
            if cfg.REACTIVE:                
                total_minimum_time_execution, _ = self.get_minimum_execution_times()
                path_to_save = videos_realData[video_idx] + '/human_times'            
                print("Path: ", path_to_save)            
                human_times = {'min': total_minimum_time_execution, 'max': self.time_execution}            
                with open(path_to_save, 'wb') as handle:
                    pickle.dump(human_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Esto es para comprobar
            # path = videos_realData[video_idx] + '/human_times'
            # human_times = np.load(path, allow_pickle=True)
            # min_time = human_times['min']
            # max_time = human_times['max']
            # print("\nMin time on loop: ", total_minimum_time_execution)
            # print("Max time on loop (self.time execution): ", self.time_execution)
            # print("\nMIN TIME according to annotation: ", min_time)
            # print("MAX TIME according to annotation: ", max_time)
            
            # print(self.action_repertoire_durations)
            
            
            # print("How many idles before taking action\n", self.idles_list)
            #self.anticipation = np.sum(self.idles_list)# * cfg.DECISION_RATE
            if self.idles_list:
                self.anticipation = np.mean(self.idles_list) * cfg.DECISION_RATE / 30 #In seconds
            else:
                self.anticipation = 0.0
            # print("ANTICIPATION IN MAIN: ", self.anticipation)


        self.total_reward += reward 
        
        # print("=============================================FIN DE STEP()")    
        
            

        return self.state, reward, done, optim,  self.flags['pdb'], self.reward_time, self.reward_energy, self.time_execution, action, self.flags['threshold'], self.prediction_error, self.total_prediction 
        
        
    def get_total_reward(self):
        """
        Returns the reward value of the environment.
        
        Returns:
            total_reward: (int) reward signal.
        """
        return self.total_reward
    
    
    
    def save_history(self):
        """
        Saves the history of states/actions for the robot and the human in a .npz file.
        """
        
        if len(self.h_history) > 0 and self.test:
            path = './temporal_files/History_Arrays/'
            if not os.path.exists(path): os.makedirs(path)
            
            file_name = "{0}.npz".format(video_idx)
            np.savez(os.path.join(path, file_name), 
            h_history=self.h_history, 
            r_history=self.r_history, 
            rwd_history=self.rwd_history,
            rwd_time_h = self.rwd_time_h,
            rwd_energy_h = self.rwd_energy_h,
            video_title = self.video_ID
            )
    
    
    def reset(self):
        """
        Resets the environment to an initial state.
        """
        super().reset()
        
        global video_idx, action_idx, annotations, frame, inaction, memory_objects_in_table, path_labels_pkl, recipe
        
        inaction = []
        memory_objects_in_table = []
        
        self.time_execution = 0
        self.reward_energy = 0
        self.reward_time = 0
        self.total_reward = 0   
        
        

        if video_idx+1 < total_videos:
            video_idx += 1

        else:
            video_idx = 0
            #print("EPOCH COMPLETED.")
      
        # action_idx is initialized as 1 to point towards the second action of the person (we cannot anticipate/predict the first action).
        action_idx = 1 
        frame = 0
        
        
        # 0) Identify the recipe: the first letter of the file is c: cereals, d: drink or t: toast.        
        recipe = videos_realData[video_idx].split("/")[-1][0]
        
        # 1) Read labels and store it in annotation_pkl
        labels_pkl = 'labels_updated.pkl'
        path_labels_pkl = os.path.join(videos_realData[video_idx], labels_pkl)
        
        # print(path_labels_pkl)

        # Save the file name        
        self.video_ID = str(path_labels_pkl.split("/")[-2])        
        
        annotations = np.load(path_labels_pkl, allow_pickle=True)
        annotations['frame_end'] -= 2
        
        # print(annotations)
        
        
        
        # print(annotations)
        
        # pdb.set_trace()
        
        
        # 2) Read initial state        
        frame_pkl = 'frame_0000' #Initial frame pickle        
        path_frame_pkl = os.path.join(videos_realData[video_idx], frame_pkl)        
        
        read_state = np.load(path_frame_pkl, allow_pickle=True)
        
        ac_pred = (read_state['data'][0:33] - cfg.MEAN_ACTION_PREDICTION) / cfg.STD_ACTION_PREDICTION
        ac_rec = (read_state['data'][33:66] - cfg.MEAN_ACTION_RECOGNITION) / cfg.STD_ACTION_RECOGNITION
        vwm = (read_state['data'][66:110] - cfg.MEAN_VWM) / cfg.STD_VWM
        
        z = (read_state['z'] - cfg.MEAN_Z) / cfg.STD_Z

        oit = list(OBJECTS_INIT_STATE.values())
        oit = (oit - np.mean(oit)) / np.std(oit)
        
        #RRRRRR temporal ctx
        # remaining_frames_pkl = os.path.join(videos_realData[video_idx], 'remaining_frames')
        # self.remaining_frames = np.load(remaining_frames_pkl, allow_pickle=True).numpy().squeeze()        
                

        if cfg.TEMPORAL_CONTEXT:
                
            #FROM CONFIG (for testing, as we don't save the ML in external memory) (Since it is a ML estimate, with enough iterations, they are almost the same)
            action_durations_ML = list(cfg.ROBOT_ACTION_DURATIONS.values())
            action_durations_ML = (action_durations_ML - np.mean(action_durations_ML)) / np.std(action_durations_ML)

            #FROM RUNNING MEAN
            #action_durations_ML = self.action_repertoire_durations
            
            #print("Remainigng frames en transition: ", self.remaining_frames[frame_to_read])
            # human_action_estimate = [self.remaining_frames[frame_to_read]] #This will be output by the ACTION PREDICTION MODULE
            
            #Using ANNOTATIONS
            human_action_estimate = np.zeros(1)
            human_action_estimate[0] = (((annotations['frame_end'][action_idx] - frame) / 3513) - 0.5)*2
            
            
            # temp_ctx = concat_vectors(action_durations_ML, human_action_estimate)
            
            
            # W/O Z-hidden state of LSTM
            if Z_hidden_state:
                # print("WITH Z")                
                self.state = np.concatenate((ac_pred, ac_rec, vwm, oit, human_action_estimate, action_durations_ML, z))                    
                
            else:
                # print("WITHOUT Z")
                # self.state = concat_3_vectors(data,oit, temp_ctx)
                
                self.state = np.concatenate((ac_pred, ac_rec, vwm, oit, human_action_estimate, action_durations_ML))
                # print(len(self.state))
        
        else:
            # print("WITHOUT TEMPORAL CTX")
            # W/O Z-hidden state of LSTM
            if Z_hidden_state:
                # print("WITH Z")
                # self.state = concat_3_vectors(data, oit, z)
                
                self.state = np.concatenate((ac_pred, ac_rec, vwm, oit, z))
                
                
                # print(len(self.state))
            else:
                # print("WIHOUT Z")
                # self.state = concat_vectors(data,oit)
                # print(len(self.state))
                self.state = np.concatenate((ac_pred, ac_rec, vwm, oit))

            
        # if Z_hidden_state:
        #     self.state = concat_3_vectors(data, list(OBJECTS_INIT_STATE.values()), z)
        # else:
        #     self.state = concat_vectors(data, list(OBJECTS_INIT_STATE.values()))
        
        self.CA_intime = 0
        self.CA_late = 0
        self.IA_intime = 0
        self.IA_late = 0    
        self.UAC_intime = 0
        self.UAC_late = 0 
        self.UAI_intime = 0
        self.UAI_late = 0
        self.CI = 0
        self.II = 0
        
        self.UA_related = 0
        self.UA_unrelated = 0
        
        self.prediction_error = 0
        self.total_prediction = 0        
        
        self.r_history = []
        self.h_history = []
        self.rwd_history = []
        self.rwd_time_h = []
        self.rwd_energy_h = []
        
        self.anticipation = 0
        self.duration_action = 0
        
        self.objects_in_table = OBJECTS_INIT_STATE.copy()
        memory_objects_in_table.append(list(self.objects_in_table.values()))
        
        #RRRRRRRRRRRRRRRRRRRR
        self.idles = 0
        self.idles_list = []
        
        
        #RRRRRRRRRRRRRR
        self.action_repertoire_durations = [[] for x in range(cfg.ACTION_SPACE)] #Empty list of lists

        #For the ML estimates
        #self.action_repertoire_durations = np.zeros(cfg.ACTION_SPACE)
        
        self.flags = {'freeze state': False, 'decision': False, 'threshold': " ",'evaluation': "Not evaluated", 'action robot': False,'break':False,'pdb': False}
        
        self.person_state = "other manipulation"
        self.robot_state = "idle"
        
        
        return self.state


    def simple_reward(self, action, state = []): 
        global annotations, action_idx
        """
        Version of the take action function that considers a unique correct robot action for each state, related to the required object and its position (fridge or table). 
        Simple reward understood as a correspondence between the ground truth next action of the person and the robot's decision.
                
        Input:
            action: (int) from the action repertoire taken by the agent.
        Output:
            simple_reward: (int) received from the environment.
        
        """

        global memory_objects_in_table
        if state == []:
            # state = undo_one_hot(self.state[0:33]) #Next action prediction
            state = annotations['label'][action_idx] #RRRRRRRRRRRRRRR
                            
        object_before_action = memory_objects_in_table[len(memory_objects_in_table)-1]
        
        # print("In simple_reward() object before action: ", object_before_action)

        reward = 0
        positive_reward = cfg.POSITIVE_REWARD
        
        self.total_prediction += 1
        
        if annotations['label'][action_idx] != state:
            self.prediction_error += 1
            state = annotations['label'][action_idx]
            # pdb.set_trace()
        
        self.flags["action robot"] = False

        if state == 1: #'pour milk'
            
            if action ==5: #'bring milk'
                reward = positive_reward
                
            else: reward = -1    
        
        elif state == 2: #'pour water'
           
            if action ==5: #'bring water'
                reward = positive_reward
            else: reward = -1
        
        elif state == 3: #'pour coffee'
            if action ==5: #'do nothing' -> *coffee is at arm's reach
                reward = positive_reward
            else: reward = -1                    
        
        elif state == 4: #'pour Nesquik'
            if action ==5:
                reward = positive_reward
            else: reward = -1
        
        elif state == 5: #'pour sugar'
            if action ==5:
                reward = positive_reward
            else: reward = -1
        
        elif state ==6: #'put microwave'
            if action ==5:
                reward = positive_reward
            else: reward = -1    
        
        elif state == 7: #'stir spoon
            if action ==5:
                reward = positive_reward
            else: reward = -1
        
        elif state == 8: #'extract milk fridge'
            
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'milk'][0]
            if action == 2:
                # print("ENTRA")
                reward = positive_reward
           
            elif object_before_action[key] == 1:           
                self.flags["action robot"] = False
                if action ==5: 
                    # self.update("action")
                    
                    reward = 5
                else:
                    reward = -5
            else: reward = -1                    
        
        elif state == 9: #'extract water fridge'
            # self.flags["action robot"] = True
            if action ==5:
                reward = positive_reward
            else: reward = -1
        
        elif state == 10: #'extract sliced bread'
            if action ==5:
                reward = positive_reward
            else: reward = -1
        
        elif state == 11: #'put toaster'
            if action ==5:
                reward = positive_reward
            else: reward = -1    
        
        elif state == 12: #'extract butter fridge'
            # pdb.set_trace()
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'butter'][0]
            # key2 = [k for k, v in OBJECTS_MEANINGS.items() if v == 'jam'][0]
            
            if action == 0: #'bring butter'
                reward = positive_reward
                #pdb.set_trace()
                
            elif (object_before_action[key] == 1): # or (object_before_action[key2] == 1): #Bring jam & bring butter often appear together
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5: 
                    reward = 5
                else:
                    reward = -5
            else: reward = -1
        
        elif state == 13: #'extract jam fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'jam'][0]
            # key2 = [k for k, v in OBJECTS_MEANINGS.items() if v == 'butter'][0]

            if action == 1: #'bring jam'
                reward = positive_reward
                
            elif (object_before_action[key] == 1): # or (object_before_action[key2] == 1): #Bring jam & bring butter often appear together
                self.flags["action robot"] = False
                if action ==5: 
                    reward = 5
                else:
                    reward = -5
            else: reward = -1    
                        
        
        elif state == 14: #'extract tomato sauce fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'tomato sauce'][0]
            if action == 4: #'bring tomato sauce'
                reward = positive_reward
                
            elif object_before_action[key] == 1:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action == 5: 
                    reward = 5
                else:
                    reward = -5
            else: reward = -1
        
        elif state == 15: #'extract nutella fridge'
            self.flags["action robot"] = True
            # print("State 15 extract nutella")
            # pdb.set_trace()
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'nutella'][0]
            if action == 3: #'bring nutella'
                reward = positive_reward
                
            elif object_before_action[key] == 1:
                self.flags["action robot"] = False
                if action == 5: 
                    reward = 5
                else:
                    reward = -5
            else: reward = -1            
        
        elif state == 16: #'spread butter'
            if action == 5:
                reward = positive_reward
            else: reward = -1    
        
        elif state == 17: #'spread jam'
            if action == 5:
                reward = positive_reward
            else: reward = -1
        
        elif state == 18: #'spread tomato sauce'
            if action ==5:
                reward = positive_reward
            else: reward = -1                    
        
        elif state == 19: #'spread nutella'
            if action ==5:
                reward = positive_reward
            else: reward = -1
        
        elif state == 20: #'pour olive oil'
            if action ==5:
                reward = positive_reward
            else: reward = -1
            
        elif state == 21: #'put jam fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'jam'][0]
            if action ==5:
                reward = positive_reward
            
            else: reward = -1    
            """    
            elif object_before_action[key] == 0:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5: 
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1    
        
        elif state == 22: #'put butter fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'butter'][0]
            if action ==5:
                reward = positive_reward
            
            else: reward = -1    
            """   
            elif object_before_action[key] == 0:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5: 
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1
        
        elif state == 23: #'put tomato sauce fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'tomato sauce'][0]
            if action ==5:
                reward = positive_reward
            
            else: reward = -1    
            """    
            elif object_before_action[key] == 0:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5: 
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1                    
        
        elif state == 24: #'put nutella fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'nutella'][0]
            if action ==5:
                reward = positive_reward
            
            else: reward = -1    
            
            """   
            elif object_before_action[key] == 0:
                # pdb.set_trace()
                self.flags["action robot"] = False
                if action ==5: 
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1
        
        elif state == 25: #'pour milk bowl'
            if action ==5:
                reward = positive_reward
            else: reward = -1
            
        elif state == 26: #'pour cereals bowl'
            if action ==5:
                reward = positive_reward
            else: reward = -1    
        
        elif state == 27: #'pour nesquik bowl'
            if action ==5:
                reward = positive_reward
            else: reward = -1
        
        elif state == 28: #'put bowl microwave'
            if action ==5:
                reward = positive_reward
            else: reward = -1                    
        
        elif state == 29: #'stir spoon bowl'
            if action ==5:
                reward = positive_reward
            else: reward = -1
        
        elif state == 30: #'put milk fridge'
            #self.flags["action robot"] = True
            #key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'milk'][0]
            if action ==5:
                reward = positive_reward
            
            else: reward = -1    
            """    
            elif object_before_action[key] == 0:
                self.flags["action robot"] = False
                if action ==5: 
                    reward = 5
                else:
                    reward = -5
            """
            #else: reward = -1
            
        elif state == 31: #'put sliced bread plate'
            if action ==5:
                reward = positive_reward
            else: reward = -1    
        
        else:
            if action ==5: #'do nothing'
                reward = positive_reward
            else: reward = -1                    

        
        #Obtain object keys
        objs = ['jam', 'butter', 'tomato sauce', 'nutella', 'milk']
        keys = [k for k, v in OBJECTS_MEANINGS.items() if v in objs]
        
        # for key in keys:
        #     if object_before_action[key] == 1:
        #         print("????????AKSDJKLASJDLKASJDLKJSLD")
        #         pdb.set_trace()
        #         self.flags['action robot'] = False
        
        # print("KEYS: ", keys)
        # print("Flag action robot: ", self.flags['action robot'])
        
        # pdb.set_trace()

        return reward

    
    def transition(self):
        """
        Gets a new observation of the environment based on the current frame and updates the state.
        
        Global variables:
            frame: current time step.
            action_idx: index of the NEXT ACTION (state as the predicted action). *The action_idx points towards the next atomic action at the current frame.
            annotations: pickle with the annotations, in the form of a table. 
        
        """
        
        global action_idx, frame, annotations, inaction, memory_objects_in_table, video_idx

        # print("4)Entro a transition en el frame: ", frame)    
        frame += 1 #Update frame        
        
        if frame <= annotations['frame_end'].iloc[-1]:
            self.time_execution += 1
        
        length = len(annotations['label']) - 1        
        
           
        # 1)
        #GET TIME STEP () (Updates the action_idx)
        #We transition to a new action index when we surpass the init frame of an action (so that we point towards the next one).    
        if  self.flags['freeze state'] == False:        
            # frame >= annotations['frame_init'][action_idx]
            if  frame >= annotations['frame_init'][action_idx]+1: #LAXTIONS
            #if frame >= annotations['frame_end'][action_idx-1]:
                if action_idx <= length-1: 
                    action_idx += 1
                    inaction = []
            
        
        if frame % cfg.DECISION_RATE != 0: return
        
        # 2) GET STATE FROM OBSERVED DATA            
        # 2.1 ) Read the pickle
        frame_to_read = int(np.floor(frame/6)) 
        frame_pkl = 'frame_' + str(frame_to_read).zfill(4) # Name of the pickle (without the .pkl extension)
        path_frame_pkl = os.path.join(videos_realData[video_idx], frame_pkl) # Path to pickle as ./root_realData/videos_realData[video_idx]/frame_pkl
            
        #print("This is the path to the frame picke: ", path_frame_pkl)
            
        read_state = np.load(path_frame_pkl, allow_pickle=True) # Contents of the pickle. In this case, a 110 dim vector with the Pred Ac., Reg. Ac and VWM          
        #print("Contenido del pickle en el frame", frame, "\n", read_state)
            
        
        # 2.2 ) Generate state
        data = read_state['data'][:110] #[Ac pred + Ac reg + VWM]
        
        ac_pred = (read_state['data'][0:33] - cfg.MEAN_ACTION_PREDICTION) / cfg.STD_ACTION_PREDICTION
        ac_rec = (read_state['data'][33:66] - cfg.MEAN_ACTION_RECOGNITION) / cfg.STD_ACTION_RECOGNITION
        vwm = (read_state['data'][66:110] - cfg.MEAN_VWM) / cfg.STD_VWM
        
        z = (read_state['z'] - cfg.MEAN_Z) / cfg.STD_Z
        # pre_softmax = read_state['pre_softmax']             
        # data[0:33] = pre_softmax              
        
            
        # OBJECTS IN TABLE            
        variations_in_table = len(memory_objects_in_table)
        if variations_in_table < 2:
            oit = memory_objects_in_table[0]
        else:
            oit = memory_objects_in_table[variations_in_table-1]
        
        oit = (oit - np.mean(oit)) / np.std(oit)
        
        # print("\nAct pred: ", ac_pred.shape)
        # print("\nAct recon: ", ac_rec.shape)
        # print("\nVwm: ", vwm.shape)
        # print("\nOIT: ", oit.shape)    

        if cfg.TEMPORAL_CONTEXT:
            # print("WITH TEMPORAL CTX")
            # if self.test:
            #     action_durations_ML = list(cfg.ROBOT_ACTION_DURATIONS.values())
            # else:
            #     action_durations_ML = [np.array(ad).mean() if ad else 0 for ad in self.action_repertoire_durations]
                
            #FROM CONFIG
            action_durations_ML = list(cfg.ROBOT_ACTION_DURATIONS.values())
            action_durations_ML = (action_durations_ML - np.mean(action_durations_ML)) / np.std(action_durations_ML)
            
            
            #print("Remainigng frames en transition: ", self.remaining_frames[frame_to_read])
            # human_action_estimate = [self.remaining_frames[frame_to_read]] #This will be output by the ACTION PREDICTION MODULE
            
            #Using ANNOTATIONS
            human_action_estimate = np.zeros(1)
            human_action_estimate[0] = (((annotations['frame_end'][action_idx] - frame) / 3513) - 0.5)*2
               
            # temp_ctx = concat_vectors(action_durations_ML, human_action_estimate)
            
            
            # W/O Z-hidden state of LSTM
            if Z_hidden_state:
                # print("WITH Z")                
                self.state = np.concatenate((ac_pred, ac_rec, vwm, oit, human_action_estimate, action_durations_ML, z))                    
                
            else:
                # print("WITHOUT Z")
                self.state = np.concatenate((ac_pred, ac_rec, vwm, oit, human_action_estimate, action_durations_ML))
        
        else:
            # print("WITHOUT TEMPORAL CTX")
            # W/O Z-hidden state of LSTM
            if Z_hidden_state:
                # print("WITH Z")
                self.state = np.concatenate((ac_pred, ac_rec, vwm, oit, z))
                
                
                # print(len(self.state))
            else:
                # print("WIHOUT Z")
                self.state = np.concatenate((ac_pred, ac_rec, vwm, oit))
                
        # print("STATE\n", self.state[0:33])
        # print(self.state[33:66])
        # print(self.state[66:110])
        # print(self.state[110:144])       
        


    def CreationDataset(self):
        #ver porque los states aveces tienen un elemento, de resto creo que esta todo ok
        global frame, action_idx, annotations
        
        guarda = 20 
        done = False
        state = []
        action = []
        no_action_state = []
        no_actions = []
        
        
        self.transition()
        fr_end = int(annotations['frame_end'][action_idx-1])
       
        # print("          Frame: ", frame)
        # print("Action idx: ",action_idx)
        # print(annotations['frame_end'].iloc[-2])
        
        if frame >= annotations['frame_end'].iloc[-2]:
            frame = annotations['frame_end'].iloc[-1]
            done = True
       
       
        df_video = self.get_possibility_objects_in_table()
        # pdb.set_trace()
        name_actions = []
        for index, row in df_video.iterrows():
            if row['In table'] == 1:
                name_actions.append("bring "+row['Object'])
            # else:
            #     name_actions.append("put "+row['Object']+ ' fridge')
            
        df_video_filtered = df_video[df_video['In table']==1]
        df_video_filtered['Actions'] = name_actions
        
        keys = list(df_video_filtered['Actions'])
        video_dict = {}
        for i in keys:
            video_dict[i] = 0
            
        person_states = annotations['label']
        
        df_video_dataset = df_video_filtered.copy()
        
        if not df_video_dataset.empty:
            # print(df_video)
            
            for idx,value in enumerate(person_states):
                for obj in INTERACTIVE_OBJECTS_ROBOT:
                    if obj in ATOMIC_ACTIONS_MEANINGS[value]: 
                        
                        if 'extract' in ATOMIC_ACTIONS_MEANINGS[value]:
                            if idx != 0: 
                                action_name = 'bring '+ obj
                                fr_init = annotations['frame_init'][idx-1]
                                
                                # df_selected = df_video_dataset.loc[df_video_dataset['Actions']==action_name]
                                df_video_dataset['Frame init'].loc[df_video_dataset['Actions']==action_name] = fr_init
                                # pdb.set_trace()
                                
                        # elif 'put' in ATOMIC_ACTIONS_MEANINGS[value]: 
                        #     action_name = 'put '+ obj +' fridge'
                        #     fr_init = annotations['frame_init'][idx-1]
                        #     # df_selected = df_video_dataset.loc[df_video_dataset['Actions']==action_name]
                        #     df_video_dataset['Frame init'].loc[df_video_dataset['Actions']==action_name] = fr_init
                      
                   
            df_video_dataset.sort_values("Frame init")
            # print(df_video_dataset)
            

        # print(frame)
        # print("fr end: ",fr_end)        
        while frame < fr_end:
            if not df_video_dataset.empty:
                
        
                for index, row in df_video_dataset.iterrows():
                    
                        correct_action = [k for k, v in ROBOT_ACTIONS_MEANINGS.items() if v == row['Actions']]
                        correct_action = correct_action[0]
                        duration = ROBOT_ACTION_DURATIONS[correct_action]
                        
                      
                        if row['Frame init'] + guarda < frame < row['Frame end'] - duration:
                            # pdb.set_trace()
                            while frame <  row['Frame end'] - duration:
                                if video_dict[ROBOT_ACTIONS_MEANINGS[correct_action]] == 0:
                                    if 'tomato' in ROBOT_ACTIONS_MEANINGS[correct_action]:
                                        current_obj = 'tomato sauce'
                                    else:
                                        current_obj = ROBOT_ACTIONS_MEANINGS[correct_action].split(" ")[1]
                                    if 'bring' in ROBOT_ACTIONS_MEANINGS[correct_action]:
                                        if self.objects_in_table[current_obj] == 1:
                                            self.objects_in_table[current_obj] = 0
                                            # print(ROBOT_ACTIONS_MEANINGS[correct_action])
                                            # print(self.objects_in_table)
                                            # print("SE HA REAJUSTADO PARA bring")
                                            # pdb.set_trace()
                                    # elif 'put' in ROBOT_ACTIONS_MEANINGS[correct_action]:
                                    #     if self.objects_in_table[current_obj] == 0:
                                    #         self.objects_in_table[current_obj] = 1
                                    #         # print(ROBOT_ACTIONS_MEANINGS[correct_action])
                                            # print(self.objects_in_table)
                                            # print("SE HA REAJUSTADO PARA PUT")
                                            # pdb.set_trace()
                                    memory_objects_in_table = list(self.objects_in_table.values())
                                    self.update_objects_in_table(correct_action)
                                    video_dict[ROBOT_ACTIONS_MEANINGS[correct_action]] = 1
                                    
                                if len(memory_objects_in_table) > 1:
                                    state_append = self.state
                                    state_append[110:133] = memory_objects_in_table
                                    state.append(state_append)
                                        
                                
                                action.append(correct_action)
                                self.transition()
                        else:
                            no_actions.append(5)
                            no_action_state.append(self.state)
                            
                       
                            self.transition()
     
            else:
                no_actions.append(5)
                no_action_state.append(self.state)
                self.transition()
                
                # frame = int(annotations['frame_init'][action_idx]) - 1
            
            # print("Action idx: ",action_idx)
        

        
        if len(action)>0:
            new_no_actions = []
            new_no_actions_state = []
            # number_of_actions = round(len(action)*0.6)
            # random_positions = random.sample(range(0,len(action)),number_of_actions)        
            # action = ([action[i] for i in random_positions])
            # state = ([state[i] for i in random_positions])
        else: 
            number_of_no_actions = round(len(no_actions)*0.05)
            random_positions = random.sample(range(0,len(no_actions)),number_of_no_actions)        
            new_no_actions = ([no_actions[i] for i in random_positions])
            new_no_actions_state = ([no_action_state[i] for i in random_positions])
            

            
        # print("Post: ",action)
        # print("Frame: ", frame)
        # print("")
        # # self.prints_terminal(18, 0, frame, 1)
        # print("State: ",len(state))
        # print("No actions: ",len(no_actions))
        # print("New no actions: ", len(new_no_actions))
        
       
        

        #     pdb.set_trace()
        
        # if (len(new_no_actions)==0 and len(state)==0):
        #     pdb.set_trace()
        # self.prints_terminal(5, 0, frame, 1)
        
        if len(action)>0:
            state_env = state
            action_env = action
            # print("accion: ",ROBOT_ACTIONS_MEANINGS[action_env[0]] )
            # if action_env[0] ==5:
                # pdb.set_trace()
            # for mem in memory_objects_in_table:
              
            #     for idx,obj in OBJECTS_MEANINGS.items():
            #         if mem[idx] == 1:
            #             print(obj)
            #     print("--------------------------")
                
            # variations_in_table = len(memory_objects_in_table)
            # if variations_in_table < 2:
            #     oit_prev = memory_objects_in_table[0]
            #     oit = memory_objects_in_table[0]
            # else:
            #     oit_prev = memory_objects_in_table[variations_in_table-2]
            #     oit = memory_objects_in_table[variations_in_table-1]
                
            # objects_prev_print = []
            # for key,value in OBJECTS_MEANINGS.items():
            #     if oit_prev[key] == 1: 
            #         objects_prev_print.append(value)
                
            # objects_print = []
            # for key,value in OBJECTS_MEANINGS.items():
            #     if oit[key] == 1: 
            #         objects_print.append(value)
                    
            # set1 = set(objects_prev_print)
            # set2 = set(objects_print)
            
            # missing = list(sorted(set1 - set2))
            # added = list(sorted(set2 - set1))
            # print("Frame: ",frame)
            # if len(missing) > 0:
            #     print("------> MISSING: ", missing[0])
            # if len(added) > 0:
            #     print("------> ADDED: ", added[0])
            # # pdb.set_trace()
            
            # for idx, state_ in enumerate(state_env): 
            #     print("Estado: ",ATOMIC_ACTIONS_MEANINGS[undo_one_hot(state_[:32])])
            #     objects_print = []
            #     for key,value in OBJECTS_MEANINGS.items():
            #         if state_[56:][key] == 1: 
            #             objects_print.append(value)
            #     print("Objects in table: ",*objects_print)
            # pdb.set_trace()
        else:
            state_env= new_no_actions_state
            action_env= new_no_actions
            

        return state_env, action_env, done
    
    def get_video_idx(self):
        """
        Gets the current video index.
        
        Returns:
            video_idx: (int) index of the current video in the data folder.
        """
        return video_idx
        

    
    def close(self):
        pass

