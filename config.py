#DQN PARAMETERS ------
#---------------------------------------------------------------------------------

TEST_FOLD = "fold5"

REPLAY_MEMORY = 2048 #Size of replay memory (deque object)

NUM_EPOCH = 500
NUM_EPISODES = 63 #"Number of training episodes per epoch"
BATCH_SIZE = 1024
GAMMA = 0.1 #Discount rate for future rewards
EPS_START = 0.99 #Initial exporation rate
EPS_END = 0.01 #Final exploration rate
EPS_DECAY = NUM_EPOCH #Exploration rate decay factor
TARGET_UPDATE = 60 #Episodes between target network update (policy net parameters -> target net)
LR = 1e-3 #Learning rate
POSITIVE_REWARD = 0
NO_ACTION_PROBABILITY = 90


ROOT = './Checkpoints/'
EXPERIMENT_NAME = "DQN"
SAVE_MODEL = False
SAVE_EPISODE = 100
LOAD_MODEL = False
LOAD_EPISODE = 0

Z_hidden_state = True
TEMPORAL_CONTEXT = False

 
LATE_FUSION = True #Only supported for Z_hidden_state = True

#--------------------------------------------

#DECISION RATE [frames]
DECISION_RATE = 30

#ROBOT SPEED FACTOR: if BETA < 1, the robot becomes faster; if BETA > 1, the robot becomes slower. BETA = 1 is a robot that is as fast as the average person.
BETA = 1.5

#ENERGY PENALTY FACTOR
FACTOR_ENERGY_PENALTY = 1

#CLUMSINESS: probability of extending the duration of an action by a 50%
ERROR_PROB = 0.05

# REACTIVENESS (for setting a baseline)
REACTIVE = False

#ENVIRONMENT PARAMETERS ------
#---------------------------------------------------------------------------------

VERSION = 4 

INTERACTIVE_OBJECTS_ROBOT = ['butter','jam','milk','nutella','tomato sauce']

ATOMIC_ACTIONS_MEANINGS = {
	0: 'other manipulation',
	1: 'pour milk',
	2: 'pour water',
	3: 'pour coffee',
	4: 'pour Nesquik',
	5: 'pour sugar',
	6: 'put microwave',
	7: 'stir spoon',
	8: 'extract milk fridge',
	9: 'extract water fridge',
	10: 'extract sliced bread',
	11: 'put toaster',
	12: 'extract butter fridge',
	13: 'extract jam fridge',
	14: 'extract tomato sauce fridge',
	15: 'extract nutella fridge',
	16: 'spread butter',
	17: 'spread jam',
	18: 'spread tomato sauce',
	19: 'spread nutella',
	20: 'pour olive oil',
	21: 'put jam fridge',
	22: 'put butter fridge',
	23: 'put tomato sauce fridge',
	24: 'put nutella fridge',
	25: 'pour milk bowl',
	26: 'pour cereals bowl',
	27: 'pour nesquik bowl',
	28: 'put bowl microwave',
	29: 'stir spoon bowl',
	30: 'put milk fridge',
	31: 'put sliced bread plate',
	32: 'TERMINAL STATE',

}

"""
Objects that are in the fridge:
	butter
	jam
	milk
	nutella
	tomato sauce
	water
Objects that are at human's reach:
	everything else
"""

OBJECTS_MEANINGS = {
	0: 'background',
	1: 'bowl',
	2: 'butter',
	3: 'cereals',
	4: 'coffee',
	5: 'cup',
	6: 'cutting board',
	7: 'fork',
	8: 'fridge',
	9: 'jam',
	10: 'knife',
	11: 'microwave',
	12: 'milk',
	13: 'nesquik',
	14: 'nutella',
	15: 'olive oil',
	16: 'plate',
	17: 'sliced bread',
	18: 'spoon',
	19: 'sugar',
	20: 'toaster',
	21: 'tomato sauce',
	22: 'water'
}


OBJECTS_INIT_STATE = {
    'background': 0,
	'bowl': 1,
	'butter': 0,
	'cereals': 1,
	'coffee': 1,
	'cup': 1,
	'cutting board': 1,
	'fork': 1,
	'fridge': 1,
	'jam': 0,
	'knife': 1,
	'microwave': 1,
	'milk': 0,
	'nesquik': 1,
	'nutella': 0,
	'olive oil': 1,
	'plate': 1,
	'sliced bread': 1,
	'spoon': 1,
	'sugar': 1,
	'toaster': 1,
	'tomato sauce': 0,
	'water': 1
    } 



ROBOT_ACTIONS_MEANINGS = {
	0: 'bring butter',
	1: 'bring jam',
	2: 'bring milk',
	3: 'bring nutella',
	4: 'bring tomato sauce',
	5: 'do nothing',

}


ROBOT_AVERAGE_DURATIONS = {
 	0: 174,  # bring butter
 	1: 198,  # bring jam
 	2: 186,  # bring milk
 	3: 234,  # bring nutella
 	4: 270,  # bring tomato sauce
 	5: 0,    # do nothing

 }

ROBOT_POSSIBLE_INIT_ACTIONS = {
	0: 1,
	1: 1,
	2: 1,
	3: 1,
	4: 1,
	5: 1,
}


ROBOT_ACTION_DURATIONS = ROBOT_AVERAGE_DURATIONS
ROBOT_ACTION_DURATIONS.update((x, y*BETA) for x, y in ROBOT_ACTION_DURATIONS.items())


N_OBJECTS = len(OBJECTS_MEANINGS)
ACTION_SPACE = len(ROBOT_ACTIONS_MEANINGS)
N_ATOMIC_ACTIONS = len(ATOMIC_ACTIONS_MEANINGS)

if Z_hidden_state:
    Z_HIDDEN = 1024
else:
    Z_HIDDEN = 0
    

#MEANS and STDs
MEAN_ACTION_PREDICTION = 0.030303027
STD_ACTION_PREDICTION = 0.15637583

MEAN_ACTION_RECOGNITION = 0.030303024
STD_ACTION_RECOGNITION = 0.1510676

MEAN_Z = -0.007947011
STD_Z = 1.1464134

MEAN_VWM = [0.18949991, -0.1344934,   0.18949991, -0.1344934,   0.18949991, -0.1344934, 
  0.18949991, -0.1344934,   0.18949991, -0.1344934,   0.18949991, -0.1344934,
  0.18949991, -0.1344934,   0.18949991, -0.1344934,   0.18949991, -0.1344934,
  0.18949991, -0.1344934,   0.18949991, -0.1344934,   0.18949991, -0.1344934,
  0.18949991, -0.1344934,   0.18949991, -0.1344934,   0.18949991, -0.1344934,
  0.18949991, -0.1344934,   0.18949991, -0.1344934,   0.18949991, -0.1344934,
  0.18949991, -0.1344934,   0.18949991, -0.1344934,   0.18949991, -0.1344934,
  0.18949991, -0.1344934, ]

STD_VWM = [0.652777,   0.68666816, 0.652777,   0.68666816, 0.652777,   0.68666816,
 0.652777,   0.68666816, 0.652777,   0.68666816, 0.652777,   0.68666816,
 0.652777,   0.68666816, 0.652777,   0.68666816, 0.652777,   0.68666816,
 0.652777,   0.68666816, 0.652777,   0.68666816, 0.652777,   0.68666816,
 0.652777,   0.68666816, 0.652777,   0.68666816, 0.652777,   0.68666816,
 0.652777,   0.68666816, 0.652777,   0.68666816, 0.652777,   0.68666816,
 0.652777,   0.68666816, 0.652777,   0.68666816, 0.652777,   0.68666816,
 0.652777,   0.68666816,]


MEAN_VWM_EVEN = -0.1344934
STD_VWM_EVEN = 0.68666816

MEAN_VWM_ODD = 0.18949991
STD_VWM_ODD = 0.652777    

MEAN_TEMP_CTX = 177.0
STD_TEMP_CTX = 85.39906322671227