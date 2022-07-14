

#DQN PARAMETERS ------
#---------------------------------------------------------------------------------

REPLAY_MEMORY = 1000 #Size of replay memory (deque object)

NUM_EPISODES = 2000 #"Number of training epochs"
BATCH_SIZE = 64
GAMMA = 0.6 #Discount rate for future rewards
EPS_START = 0.99 #Initial exporation rate
EPS_END = 0.01 #Final exploration rate
EPS_DECAY = NUM_EPISODES*12 #Exploration rate decay factor
TARGET_UPDATE = 10 #Episodes between target network update (policy net parameters -> target net)
LR = 1e-4 #Learning rate

ROOT = './Checkpoints/'
EXPERIMENT_NAME = "first_DQN"
SAVE_MODEL = False
SAVE_EPISODE = 100
LOAD_MODEL = False
LOAD_EPISODE = 0


#ENVIRONMENT PARAMETERS ------
#---------------------------------------------------------------------------------

VERSION = 2 #If VERSION == 1, the STATE is the NEXT ATOMIC ACTION. If VERSION == 2, the STATE is the concatenation of the NEXT ATOMIC ACTION and the VISUAL WORKING MEMORY. So far, in both cases the reward only depends on the action taken considering only the NEXT ACTION. VERSION == 3, STATE = NA + VWM + AO

N_ATOMIC_ACTIONS = 33 #Number of total atomic actions. 33 = 31 actions + 1 'other manipulation' + 1 Terminal state
N_OBJECTS = 23 #Number of objects. Input variables: "Active Object" and "VWM" (Visual Working Memory)
ACTION_SPACE = 24 #Number of robot actions. Output variable


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

"""
Objects that are at human's reach:
	everything else

Objects that are in the fridge:
	butter
	jam
	milk
	nutella
	tomato sauce
	water

"""


ROBOT_ACTIONS_MEANINGS = {	
	0: 'bring bowl',
	1: 'bring butter',
	2: 'bring cereals',
	3: 'bring coffee',
	4: 'bring cup',
	5: 'bring fork',
	6: 'bring jam',
	7: 'bring knife',
	8: 'bring milk',
	9: 'bring nesquik',
	10: 'bring nutella',
	11: 'bring olive oil',
	12: 'bring plate',
	13: 'bring sliced bread',
	14: 'bring spoon',
	15: 'bring sugar',
	16: 'bring tomato sauce',
	17: 'bring water',
	18: 'do nothing',
	19: 'return jam',
	20: 'return butter',
	21: 'return tomato sauce',
	22: 'return nutella',
	23: 'return milk'

}

N_OBJECTS = len(OBJECTS_MEANINGS)
ACTION_SPACE = len(ROBOT_ACTIONS_MEANINGS)
N_ATOMIC_ACTIONS = len(ATOMIC_ACTIONS_MEANINGS)



