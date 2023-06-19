# A Deep Q-Network Decision Making System for assistive robots in a simulation environment of human-robot interaction

Abstract -- The development of intelligent robots has become a trend to assist people with cognitive or motor disabilities and reduce the dependency in their own homes. Personalizing the human-robot interaction (HRI) is a key component to ease the acceptance of the artificial agent into hospital or domestic environments. Latest efforts on social robotics aim to equip the robot with more human-like capabilities, such as Natural Language skills, emotional responses or proactiveness.
This latter characteristic shifts the interactive paradigm from the traditional master-slave, in which the robot simply obeys commands, to the anticipative one, in which the robot predicts the patient’s needs and acts in advance without explicit orders.

Our proposal contributes in the direction of anticipative robots by designing and implementing a Decision Making System for a robot that observes both its own
capabilities and a patient’s behavior to assist in a collaborative cooking task. In our experiments we demonstrate that the robot is able to adapt its behavior based on the self-perception of its own physical skills and other parameters of interest, improving the efficiency of the HRI with respect to a reactive robot.

# PROJECT DESCRIPTION

A Pytorch implementation of a Deep Q Learning scenario with a custom gym environment. 

Our application scenario focuses on collaborative cooking tasks, in which a patient with reduced mobility tries to prepare a breakfast recipe in a kitchen. The robot aims to assist the person by bringing in advance the ingredients and utensils that are out of the immediate reach, acting proactively and avoiding explicit commands.

![DQN Training loop](https://github.com/CesarCaramazana/DQN_COMPANION_kitchen/blob/main/images/system_overview.png)



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
Basic implementation of the DQN with a Mid-level fusion strategy, as well as the Replay Memory.


![DQN Midfusion](https://github.com/CesarCaramazana/DQN_COMPANION_kitchen/blob/main/images/DQN_Late3.PNG)


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
Creation and definition of the kitchen environment.




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


