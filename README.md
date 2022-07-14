###PROJECT TITLE: "A Deep Q Network implementation for a recipe preparation task"

###PROJECT DESCRIPTION_________________________________________________________________________________

A Pytorch implementation of a Deep Q Learning scenario with a custom gym environment. The environment simulates a recipe-preparation problem, where a passive agent interacts with the objects in a kitchen and the active agent aims to provide useful help by performing an action related to the task.
The passive agent provides a reward signal via the available interfaces in order to modify and fine-tune the action decision system. 

In our particular Reinforcement Learning scenario:
- The states are continuous and dynamic. We do not have a model of the environment (state transitions), so the state is updated by observations.
- The discount factor of future rewards is close to 0 -we do not care that much about cumulative rewards; we want the agent to be optimal on a moment-to-moment basis-.
- Rewards signals are bounded between -1 and 1. 


###FILES DESCRIPTION__________________________________________________________________________________

./train_dqn.py
Training script of the DQN.

./test_dqn.py
Testing script of the DQN.

./DQN.py
Implementation of the fully-connected neural network and the Replay Memory module of the DQN.

./config.py
Default configuration of the training script and the environment. 

./gym-basic/gym_basic/envs/main.py
Implementation of the custom environment.

./aux.py
Auxiliary functions of the environment (to get state variables and reward signals from user input).


./video_annotations/*.pkl
Video annotations from the breakfast dataset, as pickle files.
Each pickle has the following fields:
	'idx_action': Index/Order of the atomic actions in the recipe.
	'label': Atomic Action.
	'verb_label': Verb of the atomic action. 
	'object_label': Objects used in the atomic action. 
	'frame_init': Initial frame of the atomic action in the video.
	'frame_end': Last frame of the atomic action in the video. 



###HOW TO INSTALL_____________________________________________________________________________________
To install required libraries, run the following command:

pip install -r requirements.txt




###HOW TO USE__________________________________________________________________________________________
Train the model.

usage: train_dqn.py [-h] [--experiment_name EXPERIMENT_NAME] [--save_model] [--load_model]
                    [--load_episode LOAD_EPISODE] [--batch_size BATCH_SIZE]
                    [--num_episodes NUM_EPISODES] [--lr LR] [--replay_memory REPLAY_MEMORY]
                    [--gamma GAMMA] [--eps_start EPS_START] [--eps_end EPS_END]
                    [--eps_decay EPS_DECAY] [--target_update TARGET_UPDATE] [--root ROOT]
                    [--display] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --experiment_name EXPERIMENT_NAME
                        (str) Name of the experiment. Used to name the folder where the model is
                        saved. For example: my_first_DQN.
  --save_model          Save a checkpoint in the EXPERIMENT_NAME folder.
  --load_model          Load a checkpoint from the EXPERIMENT_NAME folder. If no episode is
                        specified (LOAD_EPISODE), it loads the latest created file.
  --load_episode LOAD_EPISODE
                        (int) Number of episode to load from the EXPERIMENT_NAME folder, as the
                        sufix added to the checkpoints when the save files are created. For
                        example: 500, which will load 'model_500.pt'.
  --batch_size BATCH_SIZE
                        (int) Batch size for the training of the network. For example: 64.
  --num_episodes NUM_EPISODES
                        (int) Number of episodes or training epochs. For example: 2000.
  --lr LR               (float) Learning rate. For example: 1e-3.
  --replay_memory REPLAY_MEMORY
                        (int) Size of the Experience Replay memory. For example: 1000.
  --gamma GAMMA         (float) Discount rate of future rewards. For example: 0.99.
  --eps_start EPS_START
                        (float) Initial exploration rate. For example: 0.99.
  --eps_end EPS_END     (float) Terminal exploration rate. For example: 0.05.
  --eps_decay EPS_DECAY
                        (int) Decay factor of the exploration rate, in proportion to the number
                        of of steps. Step where the epsilon has decay to 0.367. For example: 3000.
  --target_update TARGET_UPDATE
                        (int) Frequency of the update of the Target Network, in number of
                        episodes. For example: 10.
  --root ROOT           (str) Name of the root folder for the saving of checkpoints. Parent
                        folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/
  --display             Display environment info as [Current state, action taken, transitioned
                        state, immediate reward, total reward].
  --cuda                Use GPU if available.


------------------------------------------------
Test the model.

usage: test_dqn.py [-h] [--experiment_name EXPERIMENT_NAME] [--load_episode LOAD_EPISODE]
                   [--root ROOT] [--num_episodes NUM_EPISODES] [--eps_test EPS_TEST] [--display] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --experiment_name EXPERIMENT_NAME
                        (str) Name of the experiment. Used to name the folder where the model was
                        saved during training. For example: my_first_DQN.
  --load_episode LOAD_EPISODE
                        (int) Number of episode to load from the EXPERIMENT_NAME folder, as the
                        sufix added to the checkpoints when the save files were created. For
                        example: 500, which will load 'model_500.pt'.
  --root ROOT           (str) Name of the root folder for the saving of checkpoints. Parent
                        folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/
  --num_episodes NUM_EPISODES
                        (int) Number of episodes.
  --eps_test EPS_TEST   (float) Exploration rate for the action-selection during test. For
                        example: 0.05
  --display             Display environment info as [Current state, action taken, transitioned
                        state, immediate reward, total reward].                        
  --cuda                Use GPU if available.


