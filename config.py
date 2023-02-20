

#DQN PARAMETERS ------
#---------------------------------------------------------------------------------


REPLAY_MEMORY = 2048 #Size of replay memory (deque object)

NUM_EPOCH = 20000

BATCH_SIZE = 512
GAMMA = 0.0 #Discount rate for future rewards
EPS_START = 0.99 #Initial exporation rate
EPS_END = 0.01 #Final exploration rate
EPS_DECAY = NUM_EPOCH #Exploration rate decay factor
TARGET_UPDATE = 10 #Episodes between target network update (policy net parameters -> target net)
LR = 1e-5 #Learning rate
POSITIVE_REWARD = 0
NO_ACTION_PROBABILITY = 70

ROOT = './Checkpoints/'
EXPERIMENT_NAME = "DQN"
SAVE_MODEL = False
SAVE_EPISODE = 100
LOAD_MODEL = False
LOAD_EPISODE = 0

DECISION_RATE = 60 
#ENVIRONMENT PARAMETERS ------
#---------------------------------------------------------------------------------

VERSION = 4 

N_ATOMIC_ACTIONS = 33 #Number of total atomic actions. 33 = 31 actions + 1 'other manipulation' + 1 Terminal state
N_OBJECTS = 23 #Number of objects. Input variables: "Active Object" and "VWM" (Visual Working Memory)


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


#Reduced action repertoire
ROBOT_ACTIONS_MEANINGS = {
	0: 'bring butter',
	1: 'bring jam',
	2: 'bring milk',
	3: 'bring nutella',
	4: 'bring sliced bread',
	5: 'bring tomato sauce',
	6: 'do nothing',
	7: 'put jam fridge',
	8: 'put butter fridge',
	9: 'put tomato sauce fridge',
	10: 'put nutella fridge',
	11: 'put milk fridge'

}

#Reduced action repertoire durations (from get_human_estimates)

# VERSION 1) AVERAGE OF HUMAN ACTION DURATIONS
ROBOT_ACTION_DURATIONS = {
	0: 174,  # bring butter
	1: 198,  # bring jam
	2: 186,  # bring milk
	3: 234,  # bring nutella
	4: 342,  # bring sliced bread
	5: 270,  # bring tomato sauce
	6: 0,    # do nothing
	7: 90,   # put jam fridge
	8: 114,  # put butter fridge
	9: 120,  # put tomato sauce fridge
	10: 168, # put nutella fridge
	11: 150  # put milk fridge
}

"""
# VERSION 2) 0.5*HUMAN ---> FAST ROBOT
ROBOT_ACTION_DURATIONS = {
	0: 87,  # bring butter
	1: 99,  # bring jam
	2: 93,  # bring milk
	3: 117,  # bring nutella
	4: 171,  # bring sliced bread
	5: 135,  # bring tomato sauce
	6: 0,    # do nothing
	7: 45,   # put jam fridge
	8: 57,  # put butter fridge
	9: 60,  # put tomato sauce fridge
	10: 84, # put nutella fridge
	11: 75  # put milk fridge
}


# VERSION 3) 2*HUMAN ---> SLOW (MORE REALISTIC) ROBOT
ROBOT_ACTION_DURATIONS = {
	0: 348,  # bring butter
	1: 396,  # bring jam
	2: 372,  # bring milk
	3: 468,  # bring nutella
	4: 684,  # bring sliced bread
	5: 540,  # bring tomato sauce
	6: 0,    # do nothing
	7: 180,   # put jam fridge
	8: 228,  # put butter fridge
	9: 249,  # put tomato sauce fridge
	10: 336, # put nutella fridge
	11: 300  # put milk fridge
}

"""

ROBOT_POSSIBLE_INIT_ACTIONS = {
	0: 1,
	1: 1,
	2: 1,
	3: 1,
	4: 1,
	5: 1,
	6: 1,
	7: 0,
	8: 0,
	9: 0,
	10: 0,
	11: 0	
}




N_OBJECTS = len(OBJECTS_MEANINGS)
ACTION_SPACE = len(ROBOT_ACTIONS_MEANINGS)
N_ATOMIC_ACTIONS = len(ATOMIC_ACTIONS_MEANINGS)


def print_setup(args):
	"""
	Prints a table with the arguments of the training script.
	Input:
		args: arguments during execution time. 
	
	"""
	print("")
	print(" Experiment ", args.experiment_name)
	print("="*39)
	print("  DQN parameters")
	print("="*39)
	print("| SIZE OF REPLAY MEMORY     | {0:<6g}".format(args.replay_memory), " |")
	print("| INITIAL EXPLORATION RATE  | {0:<6g}".format(args.eps_start), " |")
	print("| TERMINAL EXPLORATION RATE | {0:<6g}".format(args.eps_end), " |")
	print("| GAMMA DISCOUNT FACTOR     | {0:<6g}".format(args.gamma), " |")
	print("| FREQ. TARGET UPDATE       | {0:<6g}".format(args.target_update), " |")
	print("="*39)
	print("  Training parameters")
	print("="*39)
	print("| BATCH SIZE                | {0:<6g}".format(args.batch_size), " |")
	print("| LEARNING RATE             | {0:<6g}".format(args.lr), " |")
	print("| LOAD MODEL                | {0:<6g}".format(args.load_model), " |")
	print("| LOAD EPISODE              | {0:<6g}".format(args.load_episode), " |")
	
	print("="*39)	
