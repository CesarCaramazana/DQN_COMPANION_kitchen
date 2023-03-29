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

#CONFIGURATION GLOBAL ENVIRONMENT VARIABLES
ACTION_SPACE = cfg.ACTION_SPACE
N_ATOMIC_ACTIONS = cfg.N_ATOMIC_ACTIONS
N_OBJECTS = cfg.N_OBJECTS

ATOMIC_ACTIONS_MEANINGS = cfg.ATOMIC_ACTIONS_MEANINGS
OBJECTS_MEANINGS = cfg.OBJECTS_MEANINGS
ROBOT_ACTIONS_MEANINGS = copy.deepcopy(cfg.ROBOT_ACTIONS_MEANINGS)
ROBOT_ACTION_DURATIONS = cfg.ROBOT_ACTION_DURATIONS
ROBOT_POSSIBLE_INIT_ACTIONS = cfg.ROBOT_POSSIBLE_INIT_ACTIONS 
OBJECTS_INIT_STATE = copy.deepcopy(cfg.OBJECTS_INIT_STATE)

VERSION = cfg.VERSION
POSITIVE_REWARD = cfg.POSITIVE_REWARD

Z_hidden_state = cfg.Z_hidden_state

INTERACTIVE_OBJECTS_ROBOT = cfg.INTERACTIVE_OBJECTS_ROBOT

#ANNOTATION-RELATED VARIABLES

root_realData = "./video_annotations/Real_data/fold1/train/*" #!

#List of videos
videos_realData = glob.glob(root_realData) #Folders

random.shuffle(videos_realData)

total_videos = len(videos_realData)


video_idx = 0 #Index of current video
action_idx = 0 #Index of next_action
frame = 0 #Current frame
recipe = '' #12345

correct_action = -1 # esto para que es


labels_pkl = 'labels_updated.pkl'
path_labels_pkl = os.path.join(videos_realData[video_idx], labels_pkl)

annotations = np.load(path_labels_pkl, allow_pickle=True)




class BasicEnv(gym.Env):
    message = "Custom environment for recipe preparation scenario."
        

    
    def __init__(self, display=False, test=False):
        self.action_space = gym.spaces.Discrete(ACTION_SPACE) #[0, ACTION_SPACE-1]

        if Z_hidden_state:
            self.observation_space = gym.spaces.Discrete(1157) # Next Action + Action Recog + VWM + Obj in table + Z 
        else:   
            self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS*2+(N_OBJECTS-1)*2+ACTION_SPACE)
            
        self.state = [] #One hot encoded state        
        self.total_reward = 0
        #self.prev_state = []
        
        self.action_repertoire = ROBOT_ACTIONS_MEANINGS
        self.next_atomic_action_repertoire = ATOMIC_ACTIONS_MEANINGS
        
        self.display = display
        
        self.test = test
        
        self.flags = {'freeze state': False, 'decision': False, 'threshold': " ",'evaluation': "Not evaluated", 'action robot': False,'break':False,'pdb': False}
        
        self.person_state = "other manipulation"
        self.robot_state = "idle"
        
        self.reward_energy = 0
        self.reward_time = 0
        
        self.time_execution = 0
        self.mode = 'train'
        
        self.objects_in_table = OBJECTS_INIT_STATE.copy()
        
        global root, videos_realData, total_videos, annotations
        
        if self.test:
            print("==== TEST SET ====")

            root_realData = "./video_annotations/Real_data/fold1/test/*" #!
            videos_realData = glob.glob(root_realData)   
            
            print("Videos test\n", videos_realData)
            
            #random.shuffle(videos_realData)
            
            total_videos = len(videos_realData)
            
            labels_pkl = 'labels_updated.pkl'
            path_labels_pkl = os.path.join(videos_realData[video_idx], labels_pkl)
            

            
            annotations = np.load(path_labels_pkl, allow_pickle=True)
            
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
        
        #12345
        self.UA_related = 0
        self.UA_unrelated = 0
        
        self.prediction_error = 0
        self.total_prediction = 0
        
        self.r_history = []
        self.h_history = []
        self.rwd_history = []
        self.rwd_time_h = []
        self.rwd_energy_h = []
        
        #0000
        self.anticipation = []
        self.duration_action = 0
        
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
    
   
    def energy_robot_reward (self, action):
         
        self.reward_energy = -cfg.ROBOT_AVERAGE_DURATIONS[action]*cfg.FACTOR_ENERGY_PENALTY #ENERGY PENALTY
    
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

        self.energy_robot_reward(action)
        self.reward_energy = energy_reward * self.reward_energy
        

        
    def update_objects_in_table (self, action):
        


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
        random_position = random.randint(0,len(inaction)-1)
        
        #self.prev_state = inaction[random_position][1]
        self.state = inaction[random_position][1] # esto se hace para que el next_state sea el siguiente al guardado
        reward = inaction[random_position][2]
        
        return reward
        
    def select_correct_action (self, action): 
        
        global frame, annotations
        
        length = len(annotations['label']) -1 
        last_frame = int(annotations['frame_end'][length])
        
        for idx, val in cfg.ROBOT_ACTION_DURATIONS.items(): 
            reward = self._take_action(idx)
            if reward > -1 and idx!=5: 
                correct_action = idx
                duration_action = val
        
        # print("Acción tomada: ",cfg.ROBOT_ACTIONS_MEANINGS[action])
        # print("Corrección de accion: ",cfg.ROBOT_ACTIONS_MEANINGS[correct_action])
              
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
            reward = self._take_action(idx, real_state)
            if reward > -1: 
                correct_action = idx
                duration_action = val

        new_threshold = duration_action + frame 
        if new_threshold > last_frame: 
            new_threshold = last_frame
            
        return new_threshold, correct_action
    
    def update(self, update_type): 
        global frame, action_idx, inaction, annotations
        
        length = len(annotations['label']) - 1 
        fr_init_next = int(annotations['frame_init'][action_idx])
        fr_end = int(annotations['frame_end'][action_idx-1])
        
        if self.flags['threshold'] == "flipo":
           
            self.flags['break'] = True

        else:
              

            if update_type == "action":
                 
                frame = int(annotations['frame_end'][action_idx])
                # print("FRame in update: ", frame)
                # pdb.set_trace()
                if action_idx + 1 <= length:
                    action_idx = action_idx + 1
                inaction = []
                    
            if update_type == "unnecesary":
                # pdb.set_trace()

                if self.flags['threshold'] == ('second' or 'next action init'):
                    frame = fr_init_next 
                    
                    if action_idx + 1 <= length:
                        action_idx = action_idx + 1
                elif self.flags['threshold'] == ('first' or 'next action'):
                    # print(self.flags)
                    # pdb.set_trace()
                    # print(frame)
                    # pdb.set_trace()
                    if frame > fr_end:
                        frame = fr_end 
                    
                        
            
                inaction = []
        # print('despues: ', frame)
    def time_course (self, action):
        global frame, action_idx, inaction
        
        sample = random.random() #0000
        
        if sample < cfg.ERROR_PROB:
            #fr_execution = cfg.ROBOT_ACTION_DURATIONS[int(action)] + frame + cfg.ROBOT_ACTION_DURATIONS[int(action)]/2
            #self.duration_action = cfg.ROBOT_ACTION_DURATIONS[int(action)] + cfg.ROBOT_ACTION_DURATIONS[int(action)]/2
            
            self.duration_action = int(random.gauss(1.5*cfg.ROBOT_ACTION_DURATIONS[int(action)], 0.2*cfg.ROBOT_ACTION_DURATIONS[int(action)]))
            fr_execution = self.duration_action + frame
            
        else:
            #fr_execution = cfg.ROBOT_ACTION_DURATIONS[int(action)] + frame
            
            self.duration_action = int(random.gauss(cfg.ROBOT_ACTION_DURATIONS[int(action)], 0.2*cfg.ROBOT_ACTION_DURATIONS[int(action)]))
            fr_execution = self.duration_action + frame
            
        
        
        fr_end = int(annotations['frame_end'][action_idx-1])
        fr_init_next = int(annotations['frame_init'][action_idx]) 
        
        last_frame = int(annotations['frame_end'].iloc[-1])
        
        if self.flags['decision']:
            self.flags['freeze state'] = True
        else:
            self.flags['freeze state'] = False
            
            
        self.flags['threshold'] = ''
        # pdb.set_trace()
        if action !=5: 

            if fr_execution > last_frame: 
                threshold = last_frame 
                fr_execution = last_frame 
            
            # if frame < fr_end: 
            if fr_execution < fr_end:
                    threshold = fr_end
                    self.flags['threshold'] = "first"
            else:
                # aqui hay que comprobar si la siguiente accion realmente quiere algo la persona o no, sino lo evaluará en 
                # la siguiente accion
                _,df_video = self.get_minimum_execution_times()
                
                #filtro para quedarme solo con las acciones que realmente se pueden hacer 
                df_video_filtered = df_video[df_video['In table']==1]
                increase_threshold = True
                
                # _ = self._take_action(action)
                
                # pdb.set_trace()
                
               
                
                for index, row in df_video_filtered.iterrows():
                   if row['Frame init'] <= frame < row['Frame end']:
                       if self.objects_in_table[row['Object']] != row['In table']:  # nunca llega a hacer la 1 accion, que no se pare
                            if row['Frame end'] != int(annotations['frame_init'][0]):
                               increase_threshold = False
                           
                   if row['Frame init'] <= fr_execution < row['Frame end']:
                        if self.objects_in_table[row['Object']] != row['In table']: # nunca llega a hacer la 1 accion, que no se pare
                            if row['Frame end'] != int(annotations['frame_init'][0]):
                                increase_threshold = False
                   
                # print(df_video_filtered)
                # print(increase_threshold)
                # pdb.set_trace()
                # se puede ampliar el thr si fuera necesario (la siguiente accion no la tiene que hacer el robot)
                if increase_threshold:
                    
                    if fr_execution < fr_init_next: 
                        threshold = fr_init_next
                        fr_end = fr_init_next

                        self.flags['threshold'] = "second"
                    else:
                        self.flags['freeze state'] = False
                        # print("SE INCREMENTA")
 
                        threshold = int(annotations['frame_end'][action_idx])
                        self.flags['threshold'] = "next action"
     
                        if fr_execution > threshold:
                            if action_idx + 1 <= len(annotations['label']) - 1:
                                threshold = int(annotations['frame_init'][action_idx+1])
                                self.flags['threshold'] = "next action init"
                             
                        if action_idx + 1 <= len(annotations['label']) - 1:
                            if fr_execution > int(annotations['frame_init'][action_idx+1]):
                                threshold = int(annotations['frame_end'][action_idx+1])
                                self.flags['threshold'] = "next action"

                        
                        fr_end = threshold
                        if fr_execution > last_frame or action_idx == len(annotations['label']) - 1:
                            # print("AL FINAL NO")
                            self.flags['freeze state'] = True
                            threshold = last_frame 
                            fr_execution = last_frame 
                            fr_end = last_frame
                            self.flags['threshold'] = "flipo"
                        
                    
                    # pdb.set_trace()
                # si increase threshold == False quiere decir que estamos en momento de hacer algo, el th no se incrementa mas alla    
                else:
                    # print("NO SE INCREMENTA")
                    # self.flags['freeze state'] = True
                    if frame < fr_end:
                        
                        threshold = max(fr_execution, fr_end)
                        self.flags['threshold'] = "first"
                    else:
                        threshold = max(fr_execution, fr_init_next)
                        fr_end = fr_init_next
                        
                        self.flags['threshold'] = "second"


                
        else: 

            if frame == fr_end - 1 or frame == fr_init_next - 1: 

                if len(inaction) > 0:
                    if "action" not in inaction:
                        # flag_decision = True
                        self.flags['decision'] = True

                        # self.flags['freeze state'] = True

                if frame == fr_init_next - 1: 
                    threshold = fr_init_next 
                    fr_end = fr_init_next

                    self.flags['threshold'] = "second"
                else:
                    threshold = fr_end 
                    self.flags['threshold'] = "first"
            else: 
                
                threshold = frame
                # if len(inaction) > 0:
                    # if "action" not in inaction:
                    #     print("Times no action selected: ", len(inaction))
                    
        # if self.flags['decision']:        
        #     print(self.flags)
        return threshold, fr_execution, fr_end 
    
    def evaluation(self, action, fr_execution, fr_end, frame_post):
        global frame, action_idx, inaction, new_energy, correct_action, recipe
        
        optim = True
        simple_reward = self._take_action(action)
        new_threshold = 0 
        reward = 0

        
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

                if simple_reward == 5:
                    # print("EL OBJETO YA ESTA EN LA MESA")
                    # print("fr end (in evaluation): ", fr_end)
                    # print("Self flags in evaluation: ", self.flags)
                    self.flags['evaluation'] = 'Correct Inaction'
                    if self.flags['threshold'] == '':
                        new_threshold = fr_end
                        
                    #print(self.flags)
                    # pdb.set_trace()
                    #pdb.set_trace()
                    #if frame == fr_end: 
                       # print("Frame: in ev: ", frame)
                      #  reward = 0
                     #   self.CI += 1
                    #    self.update("action")
                        
                # se hace otra accion
                elif simple_reward == -5:
                    
                    # print("EL OBJETO YA ESTA EN LA MESA Y HACE OTRA COSA")
                    if fr_execution <= fr_end: 
                        
                        if frame == fr_execution:
                            frame_post.append(frame)
    
                        if frame == fr_end: 
                            
                            inaction.append("action")
                            self.energy_robot_reward(action)
                            self.get_energy_robot_reward(action)
                            reward = self.reward_energy  
                            if reward == 0:
                                self.UAC_intime += 1
                            else:
                                self.UAI_intime += 1
                                
                                #12345
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
                            
                            # print("Action idx: ", action_idx)
                            # print("*************** UNNECESARY ACTION (late) ***************")
                            inaction.append("action")
                            frame_post.append(frame)
                            self.energy_robot_reward(action)
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
                        
                        # CORRECT ACTION
                        if simple_reward > -1: 
                            # In time
                            if fr_execution <= fr_end: 
                               
                                # pdb.set_trace()
                                if frame == fr_execution: 
                                    
                                    frame_post.append(frame)
                                    # self.energy_robot_reward(action)
                                    # reward =  self.reward_energy + simple_reward
                                    self.reward_energy = 0
                                    reward = 0
                                    
                                if frame == fr_end:
                                
                                    self.CA_intime += 1
                                    # print("Action idx: ", action_idx)
                                    # print("*************** CORRECT ACTION (in time) ***************")
                                    inaction.append("action") 
                                    self.update("action")
                                    # self.flags['pdb'] = True
                            # Late
                            else: 
                                if frame == fr_execution: 
                                
                                    self.CA_late += 1
                                    self.reward_time += -1
                                    # print("Action idx: ", action_idx)
                                    # print("*************** CORRECT ACTION (late) ***************")
                                    inaction.append("action")
                                    frame_post.append(frame)
                                    self.update("action")   
                                    # self.energy_robot_reward(action)
                                    # reward = self.reward_energy + self.reward_time + simple_reward
                                    self.reward_energy = 0
                                    reward = self.reward_time 
                                   
                                    # self.flags['pdb'] = True
        
                                if frame >=  fr_end: 
                                
                                    self.reward_time += -1
                            
                        # # INCORRECT
                        else: 
                            self.flags['freeze state'] = True
                            # INCORRECT ACTION
                            if self.flags["action robot"] == True: 
                                
                                # if self.flags['threshold'] == 'next action':
                                #     print('FLIPO')
                                #     pdb.set_trace()
                                if fr_execution <= fr_end: 
                                    if frame == fr_execution: 
                                        frame_post.append(frame)
                                        
                                    if frame == fr_end: 
                                        self.IA_intime += 1
                                        # print("Action idx: ", action_idx)
                                        # print("*************** INCORRECT ACTION (in time) ***************")
                                        inaction.append("action")
                                        new_threshold, correct_action = self.select_correct_action(action)
                                        # print("new_threshold: ",new_threshold)
                                        
                                        # print(self.objects_in_table)
                                        # print('ACCION CORRECTA: ',ROBOT_ACTIONS_MEANINGS[correct_action])
                                        # pdb.set_trace()
                                        
                                        self.flags['evaluation'] = 'Incorrect action'
        
                                        self.energy_robot_reward(action)
                                        self.get_energy_robot_reward(action)
                                        prev_energy = self.reward_energy
                                        self.energy_robot_reward(correct_action)
                                        self.reward_energy = self.reward_energy + prev_energy
                                        # self.flags['pdb'] = True
                                        
                                else: 
                                    if frame > fr_end:
                                        self.reward_time += -1
                                    if frame == fr_execution: 
                                        self.IA_late += 1
                                        # print("Action idx: ", action_idx)
                                        # print("*************** INCORRECT ACTION (late) ***************")
                                    
                                        # inaction.append("action")
                                        frame_post.append(frame)
                                        new_threshold, correct_action = self.select_correct_action(action)
                                        
                                        # print(self.objects_in_table)
                                        # print('ACCION CORRECTA: ',ROBOT_ACTIONS_MEANINGS[correct_action])
                                        # pdb.set_trace()
                                        
                                        # self.flags['pdb'] = True
                                        self.flags['evaluation'] = 'Incorrect action'
                                  
                                        self.get_energy_robot_reward(action)
                                        prev_energy = self.reward_energy
                                        self.energy_robot_reward(correct_action)
                                        self.reward_energy = self.reward_energy + prev_energy
        
                            # UNNECESARY ACTION 
                            else: 
        
                                if fr_execution <= fr_end: 
                                    # if self.flags['threshold'] == 'flipo':
                                        
                                    #     print(frame)
                                    if frame == fr_execution:
                                        frame_post.append(frame)
        
                                    if frame == fr_end: 
                                        # print('actual frame: ',frame)
                                        # print("frame end: ", fr_end)
                                        # print('exe: ', fr_execution)
                                        # print("*************** UNNECESARY ACTION ***************")
                                        inaction.append("action")
                                        
                                        self.energy_robot_reward(action)
                                        self.get_energy_robot_reward(action)

                                        reward = self.reward_energy  
                                        if reward == 0:
                                            self.UAC_intime += 1
                                        else:
                                            self.UAI_intime += 1
                                            
                                        #12345
                                            if recipe == 'c' or recipe == 'd':
                                                if action == 2: self.UA_related += 1
                                                else: self.UA_unrelated += 1
                                            elif recipe == 't':
                                                if action in [0, 1, 3, 4]: self.UA_related += 1
                                                else: self.UA_unrelated += 1
                                                
                                            
                                            
                                            
                                            
                                        self.update("unnecesary")
                                        
                                else: 
        
                                    if frame > fr_end:
                                        self.reward_time += -1
                                      
                                    if frame == fr_execution: 
                                        
                                        # print("Action idx: ", action_idx)
                                        # print("*************** UNNECESARY ACTION (late) ***************")
                                        inaction.append("action")
                                        frame_post.append(frame)
                                        self.energy_robot_reward(action)
                                        self.get_energy_robot_reward(action)
                                        reward =  self.reward_energy + self.reward_time
                                        if self.reward_energy == 0:
                                            self.UAC_late += 1
                                        else:
                                            self.UAI_late += 1
                                            
                                            #12345
                                            if recipe == 'c' or recipe == 'd':
                                                if action == 2: self.UA_related += 1
                                                else: self.UA_unrelated += 1
                                            elif recipe == 't':
                                                if action in [0, 1, 3, 4]: self.UA_related += 1
                                                else: self.UA_unrelated += 1
                                            
                                            
                                        if  self.flags['threshold'] == 'next action init'   : 
                                            self.flags['threshold'] = 'next action'
                                        self.update("unnecesary")
                                        
                                        self.flags['break'] = True
                                        # flag_break = True
                                # flag_pdb = True
                                   
                            
                    else:
                        inaction.append([action, self.state, reward])
                        if self.flags['decision'] == True:
                            if frame == fr_end: 
                                # CORRECT INACTION
                                if simple_reward > -1: 
                                    # no se actualiza como antes 
                                    self.CI += 1
                                    # print("Action idx: ", action_idx)
                                    # print("*************** CORRECT INACTION ***************")
                                    reward = self.select_inaction_sample(inaction)
                          
                                # INCORRECT INACTION
                                else: 
                                    self.flags['freeze state'] = True
                                    self.II += 1
                                    new_threshold, correct_action = self.select_correct_action(action)
                                    self.energy_robot_reward(correct_action)
                                    reward = self.reward_energy
                                    self.flags['evaluation'] = 'Incorrect inaction'
                                    # pdb.set_trace()
                                    # print("Action idx: ", action_idx)
                                    # print(self.objects_in_table)
                                    # print("*************** INCORRECT INACTION ***************")
                                    # # print(self.flags)
                                    # print('reward: ',reward)
                                    # pdb.set_trace()
                                frame_post.append(frame)
        
        
                            else: 
                                optim = False
                        else: 
                            optim = False


        return reward, new_threshold, optim, frame_post, correct_action
        
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
        Transition from the current state (self.state) to the next one given an action.
        
        Input:
            action: (int) action taken by the agent.
        Output:
            next_state: (numpy array) state transitioned to after taking action.
            reward: (int) reward received. 
            done: (bool) True if the episode is finished (the recipe has reached its end).
            info:    
        """
        global frame, action_idx, annotations, inaction, memory_objects_in_table, correct_action, path_labels_pkl
        

        
        self.flags['decision'] = action_array[1]
        action = action_array[0]
        self.mode = action_array[2]
        assert self.action_space.contains(action)
        
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
        self.flags['threshold'] = " "

        len_prev = 1
        
        min_time = 0 #Optimal time for recipe
        max_time = 0 #Human time withour HRI
        hri_time = 0
        
        threshold, fr_execution, fr_end = self.time_course(action)
        
        #duration_action = cfg.ROBOT_ACTION_DURATIONS[action]
        duration_action = self.duration_action
        frames_waiting = 0

        
        #print("\nFlag decision: ", self.flags['decision'])
        """
        if self.flags['decision']:
            print("\nFrame. ", frame)
            print("Threshold: ", threshold)
            print("Fr execution: ", fr_execution)
            print("Fr end: ", fr_end)
        """
        
        # if self.flags['decision'] == True:
        # #     print('\nFrame prev: ', frame)
        #      print("Action: ", ROBOT_ACTIONS_MEANINGS[action])
        frame_prev = frame 
       
       
        frame_post = []

        
        if action !=5:
            self.update_objects_in_table(action)
            memory_objects_in_table.append(list(self.objects_in_table.values()))
            # print("Threshold: ",threshold)

        if frame >= annotations['frame_init'].iloc[-1]:
  
            while frame <= annotations['frame_end'].iloc[-1]:
               
                if annotations['frame_init'][action_idx] <= frame <= annotations['frame_end'][action_idx]:
                    self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]]
                #self.robot_state = "Predicting..."
                self.robot_state = "idle"
                
                self.transition() 
                 
            #execution_times.append(annotations['frame_end'].iloc[-1])
            hri_time = self.time_execution
            self.flags['pdb'] = True
            # pdb.set_trace()
                        
            #next_state = prev_state
            next_state = self.state
            done = True
            # print("Time execution: ", self.time_execution)
            self.save_history()
  
        else:               


            while frame <= threshold:
                # print(frame)
                # print(ATOMIC_ACTIONS_MEANINGS[undo_one_hot(self.state[0:33])] )
                current_state = self.state #Current state
    
                if self.flags['decision'] == False:   
                    # se transiciona de estado pero no se hace ninguna acción 
                    self.flags['freeze state'] = False
                    #self.robot_state = "Predicting..."
                    self.robot_state = "idle"
                    
                    if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                        self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                    else:
                        self.person_state = "other manipulation"

                else: 
                    
                    optim = True
                    # self.flags['freeze state']  = True
                    # if self.flags['threshold'] == "flipo":
                    #     pdb.set_trace()
                    reward, new_threshold, optim, frame_post, correct_action = self.evaluation(action, fr_execution, fr_end, frame_post)
                    
                    # if self.flags['threshold'] != "flipo":
                    if new_threshold != 0: 
                        threshold = new_threshold
                        fr_execution = new_threshold
                        # pdb.set_trace()
                        if self.flags['evaluation'] != 'Correct Inaction': 
                            action = correct_action
                            self.update_objects_in_table(action)
                            memory_objects_in_table.append(list(self.objects_in_table.values()))
                            len_prev = 2
                        
                    if action != 5:
                        self.robot_state = ROBOT_ACTIONS_MEANINGS[action]  
                    else: 
                        self.robot_state = "idle"
                    
                    if fr_execution <= fr_end: 
                        if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                            self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                        else:
                            self.person_state = "other manipulation"
                        if frame > fr_execution: 
                            self.robot_state = "Waiting for evaluation..."
                            #self.robot_state = "idle"
                            frames_waiting += 1
                    elif fr_execution > fr_end: 
                        if frame > fr_end: 
                            self.person_state = "Waiting for robot action..."

                            
                        elif frame < fr_end: 
                            if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                                self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                            else:
                                self.person_state = "other manipulation"
                                              
                self.rwd_history.append([reward]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                self.h_history.append([self.person_state])
                self.r_history.append([self.robot_state])
                self.rwd_time_h.append([self.reward_time])
                self.rwd_energy_h.append([self.reward_energy])
                
                print("\nIn STEP\nFrame: ", frame, "\nHuman: ", self.person_state, " |  Robot: ", self.robot_state)
                print("Frame_execution: ", fr_execution, " | Frame end: ", fr_end)
                

                #print("Frame: ", frame, "Human: ", self.person_state, " | Robot: ", self.robot_state)
                
                if frame == threshold :
                    self.flags['freeze state']  = False
                   
                self.transition() #Transition to a new state
                
                next_state = self.state

                
                #PRINT STATE-ACTION TRANSITION & REWARD
                if self.display: self.render(current_state, next_state, action, reward, self.total_reward)
                
                if frame >= annotations['frame_end'].iloc[-1]:
                    done = True
                    
                if self.flags['break'] == True: 
                    break
                    


            
        if self.flags['decision'] == True:
            self.state[110:133] = memory_objects_in_table[len(memory_objects_in_table)-1]
    
        # if self.flags['pdb']:
        #     self.prints_terminal(action, frame_prev, frame_post, reward)
        #     # self.prints_debug(action)
        #     print(frame)
        #     pdb.set_trace()
             
        if optim:
            #print("\nTime rwd: ", -self.reward_time)
            #print("duration: ", duration_action)
            
            if duration_action > 0:
                #print("Duration: ", duration_action)
                #print("Frames waiting: ", frames_waiting)
                #print("Reward: ", self.reward_time)
                #print("Flag: ", self.flags["action robot"])
                
                if self.flags["action robot"] and (abs(self.reward_time) < duration_action): self.anticipation.append(duration_action + frames_waiting + self.reward_time)
                #print(self.anticipation)
            #self.prints_terminal(action, frame_prev, frame_post, reward)
            #print("Frame post: ",frame)
            #self.prints_debug(action)
        #     print(frame)
            #print(self.objects_in_table)
            #pdb.set_trace()
            
        #if done: 
            #print("Frame post: ",frame)
        #     print('time executiom: ',self.time_execution)
        #     total_minimum_time_execution,_ = self.get_minimum_execution_times()
        #     print('minimun time: ',total_minimum_time_execution)
        #     print('time execution: ', self.time_execution)
            
        #     if total_minimum_time_execution > self.time_execution:
        #         pdb.set_trace()
            """
            print(annotations)
            print("Here, at the done: ", videos_realData[video_idx])
            print("Minimum time: !, ", total_minimum_time_execution)
            print("Execution timees: ", execution_times[0])
            
            path_to_save = videos_realData[video_idx] + '/human_times'
            
            print("Path: ", path_to_save)
            
            human_times = {'min': total_minimum_time_execution, 'max': execution_times[0]}
            
            with open(path_to_save, 'wb') as handle:
                pickle.dump(human_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
            """
            
            #path_to_times = videos_realData[video_idx] + '/human_times'
            #human_times = np.load(path_to_times, allow_pickle=True)            
            
            #min_time = human_times['min']
            #max_time = human_times['max']
            
            #print("\n\n")
            #print(annotations)
            
            #print("min: ",min_time)
            #print("max: ", max_time)            
            #print("hri: ", hri_time)
  
       
        self.total_reward += reward 
        
             

        return self.state, reward, done, optim,  self.flags['pdb'], self.reward_time, self.reward_energy, self.time_execution, action, self.flags['threshold'], self.prediction_error, self.total_prediction 
        
        
    def get_total_reward(self):
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
            rwd_energy_h = self.rwd_energy_h
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
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if video_idx+1 < total_videos:
            video_idx += 1

        else:
            video_idx = 0
            #print("EPOCH COMPLETED.")
      
        #print("Video idx in reset: ", video_idx)
        
        #annotations = np.load(videos[video_idx], allow_pickle=True)
        action_idx = 1 
        frame = 0
        
        # FOR REAL DATA --------------------------------------------------------------- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # 0) Identify the recipe 123456        
        recipe = videos_realData[video_idx].split("/")[-1][0]
        
        # 1) Read labels and store it in annotation_pkl
        labels_pkl = 'labels_updated.pkl'
        path_labels_pkl = os.path.join(videos_realData[video_idx], labels_pkl)
        
        #print("\n\nPath to annotation pkl: ", path_labels_pkl)
        
        annotations = np.load(path_labels_pkl, allow_pickle=True)
        
        #print("This is the annotation pkl: \n", annotations)   
        #print("Which is of length: ", len(annotation_pkl[0])) 

        
        # 2) Read initial state        
        frame_pkl = 'frame_0000' #Initial frame pickle        
        path_frame_pkl = os.path.join(videos_realData[video_idx], frame_pkl)
        
        read_state = np.load(path_frame_pkl, allow_pickle=True)
        
        data = read_state['data']
        z = read_state['z']
        pre_softmax = read_state['pre_softmax']
        
        data[0:33] = pre_softmax
        
        self.total_reward = 0   
            
        if Z_hidden_state:
            self.state = concat_3_vectors(data, list(OBJECTS_INIT_STATE.values()), z)
        else:
            self.state = concat_vectors(data, list(OBJECTS_INIT_STATE.values()))
        
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
        
        #12345
        self.UA_related = 0
        self.UA_unrelated = 0
        
        self.prediction_error = 0
        self.total_prediction = 0
        
        
        self.r_history = []
        self.h_history = []
        self.rwd_history = []
        self.rwd_time_h = []
        self.rwd_energy_h = []
        
        #0000
        self.anticipation = []
        self.duration_action = 0
        
        self.objects_in_table = OBJECTS_INIT_STATE.copy()
        memory_objects_in_table.append(list(self.objects_in_table.values()))

        
        return self.state


    def _take_action(self, action, state = []): 
        global annotations, action_idx
        """
        Version of the take action function that considers a unique correct robot action for each state, related to the required object and its position (fridge or table). 
                
        Input:
            action: (int) from the action repertoire taken by the agent.
        Output:
            reward: (int) received from the environment.
        
        """

        global memory_objects_in_table
        if state == []:
            state = undo_one_hot(self.state[0:33]) #Next action prediction
                            
        object_before_action = memory_objects_in_table[len(memory_objects_in_table)-1]
        reward = 0
        positive_reward = POSITIVE_REWARD
        
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
           
                # print("AQUI TB")
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
            if action == 0: #'bring butter'
                reward = positive_reward
                #pdb.set_trace()
            elif object_before_action[key] == 1:
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
            if action == 1: #'bring jam'
                reward = positive_reward
                
            elif object_before_action[key] == 1:
                # pdb.set_trace()
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
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'nutella'][0]
            if action == 3: #'bring nutella'
                reward = positive_reward
                
            elif object_before_action[key] == 1:
                # pdb.set_trace()
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

            
        frame += 1 #Update frame        
        
        if frame <= annotations['frame_end'].iloc[-1]:
            self.time_execution += 1
        
        length = len(annotations['label']) - 1
        
        
        #print("Flag decision: ", self.flags['decision'])
        
        if frame % cfg.DECISION_RATE != 0: return
        
           
        # 1)
        #GET TIME STEP () (Updates the action_idx)
        #We transition to a new action index when we surpass the init frame of an action (so that we point towards the next one).    
        if  self.flags['freeze state'] == False:        
            # frame >= annotations['frame_init'][action_idx]
            if  frame >= annotations['frame_init'][action_idx]:
                if action_idx <= length-1: 
                    action_idx += 1
                    inaction = []
            
        
        # 2) GET STATE FROM OBSERVED DATA            
        # 2.1 ) Read the pickle
        frame_to_read = int(np.floor(frame/6)) 
        frame_pkl = 'frame_' + str(frame_to_read).zfill(4) # Name of the pickle (without the .pkl extension)
        path_frame_pkl = os.path.join(videos_realData[video_idx], frame_pkl) # Path to pickle as ./root_realData/videos_realData[video_idx]/frame_pkl
            
        #print("This is the path to the frame picke: ", path_frame_pkl)
            
        read_state = np.load(path_frame_pkl, allow_pickle=True) # Contents of the pickle. In this case, a 110 dim vector with the Pred Ac., Reg. Ac and VWM          
        #print("Contenido del pickle en el frame", frame, "\n", read_state)
            
        
        # 2.2 ) Generate state
        data = read_state['data']
        z = read_state['z']
        pre_softmax = read_state['pre_softmax'] 
            
        data[0:33] = pre_softmax    
            
        
            
        # OBJECTS IN TABLE            
        variations_in_table = len(memory_objects_in_table)
        if variations_in_table < 2:

            oit = memory_objects_in_table[0]
        else:
            oit = memory_objects_in_table[variations_in_table-1]
                

        if Z_hidden_state:
            self.state = concat_3_vectors(data, oit, z)
        else:
            self.state = concat_vectors(data,oit)




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
        return video_idx
        
    def summary_of_actions(self):
        print("\nCORRECT ACTIONS (in time): ", self.CA_intime)
        print("CORRECT ACTIONS (late): ", self.CA_late)
        print("INCORRECT ACTIONS (in time): ", self.IA_intime)
        print("INCORRECT ACTIONS (late): ", self.IA_late)
        # print("UNNECESSARY ACTIONS (in time): ", self.UA_intime)
        # print("UNNECESSARY ACTIONS (late): ", self.UA_late)
        print("UNNECESSARY ACTIONS CORRECT (in time): ", self.UAC_intime)
        print("UNNECESSARY ACTIONS CORRECT (late): ", self.UAC_late)
        print("UNNECESSARY ACTIONS INCORRECT (in time): ", self.UAI_intime)
        print("UNNECESSARY ACTIONS INCORRECT (late): ", self.UAI_late)
        print("CORRECT INACTIONS: ", self.CI)
        print("INCORRECT INACTIONS: ", self.II)
        print("")
    
    def close(self):
        pass

