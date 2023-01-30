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


INTERACTIVE_OBJECTS_ROBOT = cfg.INTERACTIVE_OBJECTS_ROBOT

#ANNOTATION-RELATED VARIABLES
root = "./video_annotations/train/*"
videos = glob.glob(root)
random.shuffle(videos)
total_videos = len(videos)

video_idx = 0 #Index of current video
action_idx = 0 #Index of next_action
frame = 0 #Current frame

annotations = np.load(videos[video_idx], allow_pickle=True)


class BasicEnv(gym.Env):
    message = "Custom environment for recipe preparation scenario."
        

    
    def __init__(self, display=False, test=False):
        self.action_space = gym.spaces.Discrete(ACTION_SPACE) #[0, ACTION_SPACE-1]
        
        if VERSION == 1:
            self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS) #State as Next Action
        elif VERSION == 2:
            self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS+N_OBJECTS) #State as Next Action + VWM     
        elif VERSION == 3:
            self.observation_space = gym.spaces.Discrete(N_ATOMIC_ACTIONS+N_OBJECTS*2)
        self.state = [] #One hot encoded state        
        self.total_reward = 0
        self.prev_state = []
        
        self.action_repertoire = ROBOT_ACTIONS_MEANINGS
        self.next_atomic_action_repertoire = ATOMIC_ACTIONS_MEANINGS
        
        self.display = display
        
        self.test = test
        
        self.flags = {'freeze state': False, 'decision': False, 'threshold': " ",'evaluation': "Not evaluated", 'action robot': False,'break':False,'pdb': False}
        
        self.person_state = "Other manipulation"
        self.robot_state = "Predicting..."
        
        self.reward_energy = 0
        self.reward_time = 0
        
        self.time_execution = 0
        self.mode = 'train'
        
        self.objects_in_table = OBJECTS_INIT_STATE.copy()
        
        global root, videos, total_videos, annotations
        
        if self.test:
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

        self.UAC_intime = 0
        self.UAC_late = 0
        self.UAI_intime = 0
        self.UAI_late = 0
        self.CI = 0
        self.II = 0
        
    
    def get_action_meanings(self):
        return self.action_repertoire
    def get_state_meanings(self):
        return self.state_repertoire
    
   
    def energy_robot_reward (self, action):
         
        self.reward_energy = ROBOT_ACTION_DURATIONS[action]/100 
    
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
        # print(annotations)
        # print(person_states_print)
        # print(df_video)
       
        return df_video
       
        
    
    def get_energy_robot_reward(self,action,frame):
        global memory_objects_in_table
        
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
                        energy_reward = 1
                    else:
                        energy_reward = -1
                else:
                    energy_reward = -1
            else:
                 energy_reward = -1
            
        elif len(added) > 0:
            
            # print("BRING ")
            if (df_video['Object'] == added[0]).any():
                df = df_video.loc[df_video['Object'] == added[0]]
                if (df['In table'] == 1).any():
                    df_ = df.loc[df['In table'] == 1]
                    fr_init = int(df_['Frame init'])
                    fr_end = int(df_['Frame end'])
                    
                    if fr_init < frame < fr_end:
                        energy_reward = 1
                    else:
                        energy_reward = -1
                else:
                    energy_reward = -1
            else:
                 energy_reward = -1
            
        # print("Robot: ",ROBOT_ACTIONS_MEANINGS[action])
        # print("ENERGY REWARD: ", str(energy_reward))
        self.energy_robot_reward(action)
        self.reward_energy = energy_reward * self.reward_energy
        
        # pdb.set_trace()
     
        # aqui podemos ver a que objeto implica con el historico, 
        # si la accion favorece al futuro se le da positivo y sino se le da negativo
        
        
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
                        idx_put = dict_put.copy()[obj]
                        possible_actions[idx_put] = 1
           
                   
        # print(self.objects_in_table)
        # print(dict_bring)
        # print(dict_put)
        possible_actions[18] = 1     
        # pdb.set_trace()
        return possible_actions

    def select_inaction_sample (self, inaction):
        random_position = random.randint(0,len(inaction)-1)
        
        self.prev_state = inaction[random_position][1]
        self.state = inaction[random_position][1] # esto se hace para que el next_state sea el siguiente al guardado
        reward = inaction[random_position][2]
        
        return reward
        
    def select_correct_action (self, action): 
        
        global frame, annotations
        
        length = len(annotations['label']) -1 
        last_frame = int(annotations['frame_end'][length])
        
        for idx, val in cfg.ROBOT_ACTION_DURATIONS.items(): 
            reward = self._take_action(idx)
            if reward > -1 and idx!=18: 
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
        
        if update_type == "action":
             
            frame = int(annotations['frame_end'][action_idx]) 
            if action_idx + 1 <= length:
                action_idx = action_idx + 1
          
                
            inaction = []
                
        if update_type == "unnecesary":
            
            if self.flags['threshold'] == 'second':
                frame = fr_init_next 
                if action_idx + 1 <= length:
                    action_idx = action_idx + 1
            elif self.flags['threshold'] == 'first':
                if frame > fr_end:
                    frame = fr_end 
            
            inaction = []
           
        # print(self.flags['threshold'])
        #   QUE SE HACE SI ESTAMOS EN EL ESTADO TERMINAL? 
        
        
    def time_course (self, action):
        global frame, action_idx, inaction
               
        fr_execution = cfg.ROBOT_ACTION_DURATIONS[int(action)] + frame
        fr_end = int(annotations['frame_end'][action_idx-1])
        fr_init_next = int(annotations['frame_init'][action_idx]) 
        
        last_frame = int(annotations['frame_end'].iloc[-1])
        
        # pdb.set_trace()
        if action != 18: 
            if fr_execution > last_frame: 
                threshold = last_frame
                fr_execution = last_frame
            elif frame < fr_end: 
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
                    
                    
        return threshold, fr_execution, fr_end 
    
    def evaluation (self, action, fr_execution, fr_end, frame_post):
        global frame, action_idx, inaction, new_energy
        
        optim = True
        simple_reward = self._take_action(action)
        new_threshold = 0 
        reward = 0
        correct_action = 0

        if self.flags['evaluation'] == 'Incorrect action' or self.flags['evaluation'] == "Incorrect inaction":   
            
            if frame == fr_execution: 
               
                self.flags['evaluation'] = 'Not evaluated'
                frame_post.append(frame)
                self.update("action")
                reward = self.reward_energy + self.reward_time
                self.flags['break'] = True
             
            else: 
                self.reward_time += -1
                    
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
                            self.energy_robot_reward(action)
                            reward =  self.reward_energy + simple_reward
                            
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
                            self.energy_robot_reward(action)
                            reward = self.reward_energy + self.reward_time + simple_reward
                            
                            # self.flags['pdb'] = True

                        if frame >=  fr_end: 
                        
                            self.reward_time += -1
                    
                # # INCORRECT
                else: 
              
                    # INCORRECT ACTION
                    if self.flags["action robot"] == True: 
                        
                        if fr_execution <= fr_end: 
                            if frame == fr_execution: 
                                frame_post.append(frame)
                                
                            if frame == fr_end: 
                                self.IA_intime += 1
                                # print("Action idx: ", action_idx)
                                # print("*************** INCORRECT ACTION (in time) ***************")
                                inaction.append("action")
                                new_threshold, correct_action = self.select_correct_action(action)
                                self.flags['evaluation'] = 'Incorrect action'
                                self.energy_robot_reward(action)
                                self.get_energy_robot_reward(action,frame)
                                prev_energy = self.reward_energy
                                self.energy_robot_reward(correct_action)
                                self.reward_energy = self.reward_energy + prev_energy
                                
                                
                        else: 
                            if frame > fr_end:
                                self.reward_time += -1
                            if frame == fr_execution: 
                                self.IA_late += 1
                                # print("Action idx: ", action_idx)
                                # print("*************** INCORRECT ACTION (late) ***************")
                            
                                inaction.append("action")
                                frame_post.append(frame)
                                new_threshold, correct_action = self.select_correct_action(action)
                                self.flags['evaluation'] = 'Incorrect action'
                          
                                self.get_energy_robot_reward(action,frame)
                                prev_energy = self.reward_energy
                                self.energy_robot_reward(correct_action)
                                self.reward_energy = self.reward_energy + prev_energy
                                
                     
                        # aumentar el tiempo, seleccionando la accion correcta + unos frmaes mas para devolver objeto
                        
                    # UNNECESARY ACTION 
                    else: 

                        if fr_execution <= fr_end: 
                            
                            if frame == fr_execution:
                                frame_post.append(frame)

                            if frame == fr_end: 
                                
                                inaction.append("action")
                                self.energy_robot_reward(action)
                                self.get_energy_robot_reward(action,frame)
                                reward = self.reward_energy  
                                if reward > 0:
                                    self.UAC_intime += 1
                                else:
                                    self.UAI_intime += 1
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
                                self.get_energy_robot_reward(action,frame)
                                reward =  self.reward_energy + self.reward_time
                                if self.reward_energy > 0:
                                    self.UAC_late += 1
                                else:
                                    self.UAI_late += 1
                                self.update("unnecesary")
                                
                                self.flags['break'] = True
                                # flag_break = True
                        # flag_pdb = True
                               
                        
            # if action == 18 (no se hace nada)
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
                            self.II += 1
                            new_threshold, correct_action = self.select_correct_action(action)
                            self.energy_robot_reward(correct_action)
                            reward = (-self.reward_energy)
                            self.flags['evaluation'] = 'Incorrect inaction'
                           
                            # print("Action idx: ", action_idx)
                            # print("*************** INCORRECT INACTION ***************")
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
        
        data_robot = {"Robot action":accion_robot, "Frame init": int(frame_prev), "Frame end": str(frame_post), "Reward": reward, "Reward time": self.reward_time}
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
            if obj == change and action != 18:
                print('----> ' + obj + ', ')
            else:
                print(obj + ', ')
        # print(*objects_prev_print, sep = ", ")
        print("\n ROBOT ACTION: \n", action_robot)
        print("-------- State Video --------")
        print("     ",state)
        print("OBJECTS IN TABLE:")
        for obj in objects_print: 
            if obj == change and action != 18:
            
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
        global frame, action_idx, annotations, inaction, memory_objects_in_table
        
        """
        print("\nann: ", annotations)
        print("F: ", frame)
        print("Previous end frame. ", annotations['frame_end'][action_idx-1])
        
        print("ACTION: ", action)
        
        print("Execution frame: ", self.perform_action_time(action))
        print("idx: ", action_idx)
        print("ANNOTATION RIGHT NOW: ", annotations['label'][action_idx])
        """
        
        self.flags['decision'] = action_array[1]
        action = action_array[0]
        self.mode = action_array[2]
        assert self.action_space.contains(action)
        
        reward = 0
        self.reward_energy = 0
        self.reward_time = 0
       
        done = False
        optim = False
        
        self.flags['freeze state'] = False
        self.flags['pdb'] = False 
        self.flags['break'] = False
        self.flags['evaluation'] = 'Not evaluated' 
        self.flags['threshold'] = " "

        len_prev = 2
        
        threshold, fr_execution, fr_end = self.time_course(action)
        
        # print('\nFrame prev: ', frame)

        frame_prev = frame 
       
        frame_post = []
        execution_times = []
        
        if action != 18:
            self.update_objects_in_table(action)
            memory_objects_in_table.append(list(self.objects_in_table.values()))
            # print("Threshold: ",threshold)
        prev_state = self.prev_state
        if frame >= annotations['frame_init'].iloc[-1]:
            while frame <= annotations['frame_end'].iloc[-1]:
               
                if annotations['frame_init'][action_idx] <= frame <= annotations['frame_end'][action_idx]:
                    self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]]
                self.robot_state = "Predicting..."
                
                self.transition() 
                 
            execution_times.append(annotations['frame_end'].iloc[-1])
            execution_times.append(self.time_execution)
            self.flags['pdb'] = True
            # pdb.set_trace()
                        
            next_state = prev_state
            done = True
  
        else:
                
            while frame <= threshold:
                
                current_state = self.state #Current state
    
                if self.flags['decision'] == False:   
                    # se transiciona de estado pero no se hace ninguna acción 
                    self.flags['freeze state'] = False
                    self.robot_state = "Predicting..."
                    
                    if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                        self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                    else:
                        self.person_state = "Other manipulation"

                else: 
                    
                    optim = True
                    self.flags['freeze state']  = True
                    
                    reward, new_threshold, optim, frame_post, correct_action = self.evaluation(action, fr_execution, fr_end, frame_post)
                    
                    if new_threshold != 0: 
                        threshold = new_threshold
                        fr_execution = new_threshold
                        action = correct_action
                        self.update_objects_in_table(action)
                        memory_objects_in_table.append(list(self.objects_in_table.values()))
                        len_prev = 3
                        
                    self.robot_state = ROBOT_ACTIONS_MEANINGS[action]  
                    
                    if fr_execution <= fr_end: 
                        if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                            self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                        else:
                            self.person_state = "Other manipulation"
                        if frame > fr_execution: 
                            self.robot_state = "Waiting for evaluation..."
                    elif fr_execution > fr_end: 
                        if frame > fr_end: 
                            self.person_state = "Waiting for robot action..."
                            
                        elif frame < fr_end: 
                            if annotations['frame_init'][action_idx-1] <= frame <= annotations['frame_end'][action_idx-1]:
                                self.person_state = ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]
                            else:
                                self.person_state = "Other manipulation"
                                
        
                if frame == threshold:
                    self.flags['freeze state']  = False
    
                self.transition() #Transition to a new state
                
                
                next_state = self.state
                
                #PRINT STATE-ACTION TRANSITION & REWARD
                if self.display: self.render(current_state, next_state, action, reward, self.total_reward)
                
                if self.flags['break'] == True: 
                    break
        if self.flags['decision'] == True:
            # self.prints_debug(action)

            prev_state[56:] = memory_objects_in_table[len(memory_objects_in_table)-len_prev]
            next_state[56:] = memory_objects_in_table[len(memory_objects_in_table)-1]
            # if len_prev == 3:
            #     pdb.set_trace()
        # # pdb.set_trace()
        # # if optim == True:   
           
        #     self.prints_terminal(action, frame_prev, frame_post, reward)
        # # # pdb.set_trace()
        # print(str(self.flags))
        self.total_reward += reward 

        # print("Execution times: ",self.time_execution)

        return prev_state, next_state, reward, done, optim,  self.flags['pdb'], self.reward_time, self.reward_energy, execution_times       
        
        
    def get_total_reward(self):
        return self.total_reward
    
    
    def reset(self):
        """
        Resets the environment to an initial state.
        """
        super().reset()
        
        global video_idx, action_idx, annotations, frame, inaction, memory_objects_in_table
        
        inaction = []
        memory_objects_in_table = []
        annotations = np.load(videos[video_idx], allow_pickle=True)
        self.time_execution = 0
        
        if video_idx+1 < total_videos:
            video_idx += 1
        else:
            video_idx = 0
            #print("EPOCH COMPLETED.")
            # pdb.set_trace()
        
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
        elif VERSION == 3:
            self.state = concat_3_vectors(na, ao,list(OBJECTS_INIT_STATE.copy().values()))
            self.prev_state = concat_3_vectors(one_hot(annotations['label'][0], N_ATOMIC_ACTIONS), ao,list(OBJECTS_INIT_STATE.values()))
            
        self.CA_intime = 0
        self.CA_late = 0
        self.IA_intime = 0
        self.IA_late = 0    
        # self.UA_intime = 0
        # self.UA_late = 0
        self.UAC_intime = 0
        self.UAC_late = 0 
        self.UAI_intime = 0
        self.UAI_late = 0
        self.CI = 0
        self.II = 0
        
        self.objects_in_table = OBJECTS_INIT_STATE.copy()
        memory_objects_in_table.append(list(self.objects_in_table.values()))
        
        
        # print("RESET OBJECTS TABLE ... ")
        # print(self.objects_in_table)
        # pdb.set_trace()
        return self.state


    def _take_action(self, action, state = []): 
        """
        Version of the take action function that considers a unique correct robot action for each state, related to the required object and its position (fridge or table). 
                
        Input:
            action: (int) from the action repertoire taken by the agent.
        Output:
            reward: (int) received from the environment.
        
        """

        global memory_objects_in_table
        if state == []:
            if VERSION == 1:
                state = undo_one_hot(self.state) #If the state is the Next Action vector, undo the O-H to obtain the integer value.
            elif VERSION == 2: #If the state is NA + VWM, first separate the two variables and then obtain the value of the state from the Next Action.
                na, ao = undo_concat_state(self.state)
                state = undo_one_hot(na) #Only consider the Next Action as the state.    
            elif VERSION == 3:
                na, ao, oit =  undo_concat_state(self.state)
                state = undo_one_hot(na) 
                
        object_before_action = memory_objects_in_table[len(memory_objects_in_table)-2]
        reward = 0
        positive_reward = POSITIVE_REWARD
        
        self.flags["action robot"] = False

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
            
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'milk'][0]
            if action == 8:
                # print("ENTRA")
                reward = positive_reward
           
            elif object_before_action[key] == 1:
           
                # print("AQUI TB")
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
            else: reward = -1                    
        
        elif state == 9: #'extract water fridge'
            self.flags["action robot"] = True
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
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'butter'][0]
            if action == 1: #'bring butter'
                reward = positive_reward
                
            elif object_before_action[key] == 1:
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
            else: reward = -1
        
        elif state == 13: #'extract jam fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'jam'][0]
            if action == 6: #'bring jam'
                reward = positive_reward
                
            elif object_before_action[key] == 1:
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
            else: reward = -1                    
        
        elif state == 14: #'extract tomato sauce fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'tomato sauce'][0]
            if action == 16: #'bring tomato sauce'
                reward = positive_reward
                
            elif object_before_action[key] == 1:
           
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
            else: reward = -1
        
        elif state == 15: #'extract nutella fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'nutella'][0]
            if action == 10: #'bring nutella'
                reward = positive_reward
                
            elif object_before_action[key] == 1:
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
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
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'jam'][0]
            if action == 19:
                reward = positive_reward
                
            elif object_before_action[key] == 0:
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
            else: reward = -1    
        
        elif state == 22: #'put butter fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'butter'][0]
            if action == 20:
                reward = positive_reward
               
            elif object_before_action[key] == 0:
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
            else: reward = -1
        
        elif state == 23: #'put tomato sauce fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'tomato sauce'][0]
            if action == 21:
                reward = positive_reward
                
            elif object_before_action[key] == 0:
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
            else: reward = -1                    
        
        elif state == 24: #'put nutella fridge'
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'nutella'][0]
            if action == 22:
                reward = positive_reward
               
            elif object_before_action[key] == 0:
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
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
            self.flags["action robot"] = True
            key = [k for k, v in OBJECTS_MEANINGS.items() if v == 'milk'][0]
            if action == 23:
                reward = positive_reward
                
            elif object_before_action[key] == 0:
                self.flags["action robot"] = False
                if action == 18: 
                    reward = positive_reward
                else:
                    reward = -1
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
        
        # if reward == positive_reward:
        #     if self.flags['action robot']==True:
        #         reward = POSITIVE_REWARD    
                
               
    
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
        
        if self.mode == 'val':
            rng = np.random.RandomState(2021)
            
        else: 
            rng = np.random
            
        
            
        frame += 1 #Update frame
        if frame <= annotations['frame_end'].iloc[-1]:
            self.time_execution += 1
        length = len(annotations['label']) - 1 #Length from 0 to L-1 (same as action_idx). "Duration of the recipe, in number of actions"
        
        #print("Frmae. ", frame, end='\r')
        
        
        # 1)
        #GET TIME STEP () (Updates the action_idx)
        #We transition to a new action index when we surpass the init frame of an action (so that we point towards the next one).    
        if  self.flags['freeze state'] == False: 
            # frame >= annotations['frame_init'][action_idx]
            if  frame >= annotations['frame_init'][action_idx]:
                if action_idx <= length-1: 
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
            var = 0.5*diff**4 #Noise variance
            
            # 5% chance of erroneously coding another action that does not correspond to the annotations.
            if p>1:
                # na = random.randint(0, N_ATOMIC_ACTIONS-2) #Random label with 5% prob (from action 0 to N_AA-2, to avoid coding the TERMINAL STATE)
                na = random.randint(0, N_ATOMIC_ACTIONS-2)
                na_prev = na
            # 95 % chance of coding the proper action
            else:
                na = annotations['label'][action_idx] #Correct label
                na_prev = annotations['label'][action_idx-1]
                
            na = one_hot(na, N_ATOMIC_ACTIONS) #From int to one-hot vector.
            na_prev = one_hot(na_prev, N_ATOMIC_ACTIONS)

            
            #Para eliminar ruido
            #var = 0
            
            
            # Generate gaussian noise with 0 mean and 1 variance.
            # noise = np.random.normal(0, 1, N_ATOMIC_ACTIONS)
            noise = rng.normal(0, 1, N_ATOMIC_ACTIONS)
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
            
            variations_in_table = len(memory_objects_in_table)
            if variations_in_table < 2:
                oit_prev = memory_objects_in_table[0]
                oit = memory_objects_in_table[0]
            else:
                oit_prev = memory_objects_in_table[variations_in_table-2]
                oit = memory_objects_in_table[variations_in_table-1]

            # pdb.set_trace()
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
        elif VERSION == 3:
            state = concat_3_vectors(na, ao, oit)
            prev_state = concat_3_vectors(na_prev, ao_prev,oit_prev)
            
        self.state = state
        self.prev_state = prev_state
        
        # if self.mode=='val':
        #     if video_idx <2:
        #         if frame <2: 
        #             print('Val: ')
        #             print('noise: ',noise)
                    # pdb.set_trace()
                
        # if self.mode=='train':
        #     if video_idx <2:
        #         if frame <2: 
        #             print('Train: ')
        #             print('noise: ',noise)
                    # pdb.set_trace()

    
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
        elif VERSION == 3:
            na, vwm, oit = undo_concat_state(state)
            next_na, next_vwm, next_oit = undo_concat_state(next_state)
            state = undo_one_hot(na)
            next_state = undo_one_hot(next_na)
        #Numerical version
        #print('| State: {:>3g}'.format(state), ' | Action: {:>3g}'.format(action), ' | New state: {:>3g}'.format(next_state), ' | Reward {:>3g}'.format(reward), ' | Total reward {:>3g}'.format(total_reward), ' |')
        #print('='*82)
        
        #Version with dictionary meanings (provided that the state/action spaces are shorter than or equal to the dictionary.
        print('| STATE: {0:>29s}'.format(self.next_atomic_action_repertoire[state]), ' | ACTION: {0:>20s}'.format(self.action_repertoire[action]), ' | NEW STATE: {0:>29s}'.format(self.next_atomic_action_repertoire[next_state]), ' | REWARD {:>3g}'.format(reward), ' | TOTAL REWARD {:>3g}'.format(total_reward), ' |')
        #print('='*151)


    def transition_without_noise(self):
        """
        Gets a new observation of the environment based on the current frame and updates the state.
        
        Global variables:
            frame: current time step.
            action_idx: index of the NEXT ACTION (state as the predicted action). *The action_idx points towards the next atomic action at the current frame.
            annotations: pickle with the annotations, in the form of a table. 
        
        """
        
        global action_idx, frame, annotations, inaction
        

            
        frame += 1 #Update frame
        if frame <= annotations['frame_end'].iloc[-1]:
            self.time_execution += 1
        length = len(annotations['label']) - 1 #Length from 0 to L-1 (same as action_idx). "Duration of the recipe, in number of actions"
        
        #print("Frmae. ", frame, end='\r')
        
        
        # 1)
        #GET TIME STEP () (Updates the action_idx)
        #We transition to a new action index when we surpass the init frame of an action (so that we point towards the next one).    
        # if  self.flags['freeze state'] == False: 
            # frame >= annotations['frame_init'][action_idx]
        if  frame >= annotations['frame_init'][action_idx]:
            if action_idx <= length: 
                action_idx += 1
               
        
        
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

            na = annotations['label'][action_idx] #Correct label
            na_prev = annotations['label'][action_idx-1]
            
            na = one_hot(na, N_ATOMIC_ACTIONS) #From int to one-hot vector.
            na_prev = one_hot(na_prev, N_ATOMIC_ACTIONS)
            
            
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

    def CreationDataset(self):
        #ver porque los states aveces tienen un elemento, de resto creo que esta todo ok
        global frame, action_idx, annotations
        
        guarda = 50 
        done = False
        state = []
        action = []
        no_action_state = []
        no_actions = []
        
        
        self.transition()
        fr_end = int(annotations['frame_end'][action_idx-1])
        length = len(annotations)
        print("          Frame: ", frame)
        print("Action idx: ",action_idx)
        # print(annotations['frame_end'].iloc[-2])
        
        if frame >= annotations['frame_end'].iloc[-2]:
            frame = annotations['frame_end'].iloc[-1]
            done = True
        _,real_correct_action = self.select_correct_action_video (action_idx)
        cont = 0
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
            
        prev_state = self.prev_state
        while frame < fr_end:
            self._take_action(15)
            if self.flags["action robot"] == True: 
                _, correct_action = self.select_correct_action(15)
                
                
                print("Frame: ",frame)
                print('State: ',ATOMIC_ACTIONS_MEANINGS[undo_one_hot(self.state[:32])])
                print("Correct action: ",ROBOT_ACTIONS_MEANINGS[correct_action])
                print("real_correct_action: ",ROBOT_ACTIONS_MEANINGS[real_correct_action])
                # pdb.set_trace()
                # if real_correct_action == 21:
                # pdb.set_trace()
                if ROBOT_ACTIONS_MEANINGS[correct_action] in df_video['Actions'].values:
                     df_action = df_video.loc[df_video['Actions']==ROBOT_ACTIONS_MEANINGS[correct_action]]
                     fr_init = int(df_action['Frame init'])
                     fr_end = int(df_action['Frame end'])
                     duration = ROBOT_ACTION_DURATIONS[correct_action]
                     if fr_init < frame < fr_end-duration:
                         if video_dict[ROBOT_ACTIONS_MEANINGS[correct_action]] == 0:
                             self.update_objects_in_table(correct_action)
                             memory_objects_in_table.append(list(self.objects_in_table.values()))
                             video_dict[correct_action] = 0
                         if ATOMIC_ACTIONS_MEANINGS[undo_one_hot(self.state[:32])] == ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]]:
                        
                             if len(memory_objects_in_table) > 1:
                                 state_append = self.state
                                 state_append[56:] = memory_objects_in_table[len(memory_objects_in_table)-2]
                                 state.append(state_append)
                            
                             else: 
                                 state_append = self.state
                                 state_append[56:] = memory_objects_in_table[len(memory_objects_in_table)-1]
                                 state.append(state_append)
                                 
                             action.append(correct_action)
                         elif ATOMIC_ACTIONS_MEANINGS[undo_one_hot(self.prev_state[:32])] == ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]:
                           
                                if len(memory_objects_in_table) > 1:
                                    state_append = self.prev_state
                                    state_append[56:] = memory_objects_in_table[len(memory_objects_in_table)-2]
                                    state.append(state_append)
                               
                                else: 
                                    state_append = self.prev_state
                                    state_append[56:] = memory_objects_in_table[len(memory_objects_in_table)-1]
                                    state.append(state_append)
                                    
                                action.append(correct_action)
                         
                # if correct_action == real_correct_action and correct_action != 18:
                #     # pdb.set_trace()
                #     if cont == 0:
                #         prev_correct = correct_action
                #         self.update_objects_in_table(correct_action)
                #         memory_objects_in_table.append(list(self.objects_in_table.values()))
                #     cont = 1
                #     duration = ROBOT_ACTION_DURATIONS[correct_action]


                # pdb.set_trace()
                # if  ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]] == ('extract jam fridge' or 'put jam fridge'):
                #     print('Frame: ',frame)
                #     # pdb.set_trace()
                #     print('State: ',ATOMIC_ACTIONS_MEANINGS[undo_one_hot(self.state[:32])])
                #     print('Correct action: ', ROBOT_ACTIONS_MEANINGS[correct_action])
                #     # pdb.set_trace()
                    # if frame == fr_end -1:
                        # pdb.set_trace()
                    # if (fr_end-duration-guarda) <= frame < (fr_end-duration):
                    #     # print("Frame: ", frame)
                    #     state.append(self.state)
                    #     action.append(correct_action)
                        # print(cfg.ROBOT_ACTIONS_MEANINGS[correct_action])
            elif real_correct_action == 18:
                if ATOMIC_ACTIONS_MEANINGS[undo_one_hot(self.state[:32])] == ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx]]:
                    
                    no_actions.append(18)
                    no_action_state.append(self.state)
                elif ATOMIC_ACTIONS_MEANINGS[undo_one_hot(self.prev_state[:32])] == ATOMIC_ACTIONS_MEANINGS[annotations['label'][action_idx-1]]:
                        
                        no_actions.append(18)
                        no_action_state.append(self.prev_state)
                    
            self.transition()
           
        

        else:

            frame = int(annotations['frame_init'][action_idx]) - 1
            
            # print("Action idx: ",action_idx)
    
        
        if len(action)>0:
            new_no_actions = []
            new_no_actions_state = []
        else: 
            number_of_no_actions = round(len(no_actions)*0.01)
            random_positions = random.sample(range(0,len(no_actions)),number_of_no_actions)        
            new_no_actions = ([no_actions[i] for i in random_positions])
            new_no_actions_state = ([no_action_state[i] for i in random_positions])
            
              
        # # FILTRAR LO QUE PUEDE ESTAR MAL
        # action_count = Counter(action)
        # if len(action_count.keys()) > 1:
            
        #     print("Pre: ",action)

        #     key_max = max(zip(action_count.values(), action_count.keys()))[1]
        #     print(key_max)
        
        #     action_old = action
        #     state_old = state
            
        #     state = []
        #     action = []
        #     for idx,val in enumerate(action_old): 
        #         if val == key_max:
        #             action.append(val)
        #             state.append(state_old[idx])
                
            
        # if len(state) < guarda/4: 
        #     state = []
        #     action = []
            
        print("Post: ",action)
  
        print("")
        self.prints_terminal(18, 0, frame, 1)
        print("State: ",len(state))
        print("No actions: ",len(no_actions))
        print("New no actions: ", len(new_no_actions))
        
       
        

        #     pdb.set_trace()
        
        # if (len(new_no_actions)==0 and len(state)==0):
        #     pdb.set_trace()
        self.prints_terminal(18, 0, frame, 1)
        
        if len(action)>0:
            state_env = state
            action_env = action
            
            # for mem in memory_objects_in_table:
              
            #     for idx,obj in OBJECTS_MEANINGS.items():
            #         if mem[idx] == 1:
            #             print(obj)
            #     print("--------------------------")
                
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
            print("Frame: ",frame)
            if len(missing) > 0:
                print("------> MISSING: ", missing[0])
            if len(added) > 0:
                print("------> ADDED: ", added[0])
            # pdb.set_trace()
            
            for idx, state_ in enumerate(state_env): 
                print("Estado: ",ATOMIC_ACTIONS_MEANINGS[undo_one_hot(state_[:32])])
                objects_print = []
                for key,value in OBJECTS_MEANINGS.items():
                    if state_[56:][key] == 1: 
                        objects_print.append(value)
                print("Objects in table: ",*objects_print)
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
