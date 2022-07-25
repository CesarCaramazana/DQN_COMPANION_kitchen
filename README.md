# PROJECT TITLE 
### A Deep Q Network implementation for a recipe preparation task

# PROJECT DESCRIPTION

A Pytorch implementation of a Deep Q Learning scenario with a custom gym environment. The environment simulates a recipe-preparation problem, where a passive agent interacts with the objects in a kitchen and the active agent aims to provide useful help by performing an action related to the task.
The passive agent provides a reward signal via the available interfaces in order to modify and fine-tune the action decision system. 

![DQN Training loop](https://github.com/CesarCaramazana/DQN_COMPANION_kitchen/blob/main/images/DQN_loop.PNG?raw=True)

In our particular Reinforcement Learning scenario:
- The states are continuous and dynamic. We do not have a model of the environment (state transitions), so the state is updated by observations.
- The discount factor of future rewards is close to 0 -we do not care that much about cumulative rewards; we want the agent to be optimal on a moment-to-moment basis-.
- Rewards signals are bounded between -1 and 1. 
- The output of the network should not be softmaxed (it approximates the Q value, not the probability of each action, which depends on the reward).


**Some problems that have been identified**:
- In order to train the model without the need of real interaction, a table of deterministic rewards has been stablished in the environment setup. This shifts the problem closer to a supervised learning scenario, since there is only one optimal action for every posible state.
- The states are currently defined as the "Next atomic action", even though the Active Object variable is concatenated. To make the Active Object useful, one could try to implement a take_action function that takes into account which object is needed. 
- The robot action repertoire is not the actual repertoire. Right now, actions have been hard-coded as "grab object", where "object" is every possibility of the environment. 
- The discount gamma factor plays a critical role in the loss function curve. When gamma is close to 0.99, the loss is noisier; when gamma is close to 0, the loss has a very steep slope during the first episodes and then gets horizontal. 


# FILES DESCRIPTION

Training script of the DQN.
------------------------------------------------
```
./train_dqn.py
```
Trains the DQN. The arguments are specified below.

Testing script of the DQN.
------------------------------------------------
```
./test_dqn.py
```
Tests the DQN. Essentially executes the same operations as the training module, but the exploration rate is set to 0 and gradients are not backpropagated. 

Implementation of the fully-connected neural network and the Replay Memory module of the DQN.
------------------------------------------------
```
./DQN.py
```
Basic implementation of a FCNN, with only one hidden layer, as well as the Replay Memory.

The number of input neurons depends on the size of the STATE SPACE. For example, if we consider the state to be defined as the probabilities vector of Next Atomic Action, we would have as many neurons in the input layer as the number of atomic actions (given that they are one-hot encoded). 
The number of output neurons is the same as the number of actions from the robot action repertoire (or ACTION SPACE). The network approximates que Q value for each possible action. 

![DQN FCNN](https://github.com/CesarCaramazana/DQN_COMPANION_kitchen/blob/main/images/DQN_FCNN.PNG?raw=True)




Default configuration of the training script and the environment. 
------------------------------------------------
```
./config.py
```
Default variables for the training (such as the exploration rate, learning rate and number of episodes) and for the environment (action and state spaces, with their meanings).

Implementation of the custom environment.
------------------------------------------------
```
./gym-basic/gym_basic/envs/main.py
```
Creation and definition of the kitchen environment. This is where the take_action, transition and get_reward functions are implemented.  

Auxiliary functions of the environment .
------------------------------------------------
```
./aux.py
```
In this script, some auxiliary functions that are used in the environment setup (./gym-basic/gym_basic/envs/main.py) are implemented.
There are three types of functions:
1. **General purpose**: regarding the management of array variables.
2. **Get state**: as interface functions between the input systems and the environment. Right now using the video annotations. In the future, these functions will be used to retrieve the outputs of the Action Prediction system (among others) and generate the state of the environment.
3. **Rewards**: user interfaces to get the reward value. 

Video annotations from the breakfast dataset, as pickle files.
------------------------------------------------
```
./video_annotations/*.pkl
```
Each pickle has the following fields:
- 'idx_action': Index/Order of the atomic actions in the recipe.
- 'label': Atomic action.
- 'verb_label': Verb of the atomic action. 
- 'object_label': Objects used in the atomic action. 
- 'frame_init': Initial frame of the atomic action in the video.
- 'frame_end': Last frame of the atomic action in the video. 



# HOW TO INSTALL
To install required libraries, run the following command:
```
pip install -r requirements.txt
```
In order to generate the requirements file, you need to install the library ```pipreqs``` and run the following command in the root folder:
```
pipreqs ./ --force
```


# HOW TO USE
Train the model.
```
usage: train_dqn.py [-h] [--experiment_name EXPERIMENT_NAME] [--save_model] [--load_model] [--load_episode LOAD_EPISODE] [--batch_size BATCH_SIZE]
                    [--num_episodes NUM_EPISODES] [--lr LR] [--replay_memory REPLAY_MEMORY] [--gamma GAMMA] [--eps_start EPS_START] [--eps_end EPS_END]
                    [--eps_decay EPS_DECAY] [--target_update TARGET_UPDATE] [--root ROOT] [--display] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --experiment_name EXPERIMENT_NAME
                        (str) Name of the experiment. Used to name the folder where the model is saved. For example: my_first_DQN.
  --save_model          Save a checkpoint in the EXPERIMENT_NAME folder.
  --load_model          Load a checkpoint from the EXPERIMENT_NAME folder. If no episode is specified (LOAD_EPISODE), it loads the latest created file.
  --load_episode LOAD_EPISODE
                        (int) Number of episode to load from the EXPERIMENT_NAME folder, as the sufix added to the checkpoints when the save files are
                        created. For example: 500, which will load 'model_500.pt'.
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
                        (int) Decay factor of the exploration rate. Episode where the epsilon has decay to 0.367 of the initial value. For example:
                        num_episodes/2.
  --target_update TARGET_UPDATE
                        (int) Frequency of the update of the Target Network, in number of episodes. For example: 10.
  --root ROOT           (str) Name of the root folder for the saving of checkpoints. Parent folder of EXPERIMENT_NAME folders. For example: ./Checkpoints/
  --display             Display environment info as [Current state, action taken, transitioned state, immediate reward, total reward].
  --cuda                Use GPU if available.


```
------------------------------------------------
Test the model.
```
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

```

# FOLDER STRUCTURE

```
./
  /gym-basic
      /gym_basic
          __init__.py
          envs/
              basic_env.py
              __init__.py
              main.py
      setup.py
  /images
      some_image.png
  /video_annotations
      recipe_date.pkl
  aux.py
  DQN.py
  config.py
  train_dqn.py
  test_dqn.py
  
  /Checkpoints
      /experiment_name
          model_#episode.pt
  
  README.md
  requirements.txt

```

