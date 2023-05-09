import torch
import torch.nn as nn
import random
from collections import namedtuple, deque
import config as cfg


class DQN(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        self.feature_dim = input_size - cfg.Z_HIDDEN #First features and Z variable separated
        
        # print("DQN input size: ", input_size)
        # print("DQN feature: ", self.feature_dim)
                
        #self.input_layer1 = nn.Linear(self.feature_dim, 256)
        self.input_layer1 = nn.Linear(input_size - cfg.Z_HIDDEN, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.hidden1 = nn.Linear(256,512)
        self.bn5 = nn.BatchNorm1d(512)
        
        if cfg.Z_hidden_state:
              #self.input_layer2 = nn.Linear(input_size-self.feature_dim, 256)
              self.input_layer2 = nn.Linear(cfg.Z_HIDDEN, 256)
              self.bn2 = nn.BatchNorm1d(256)
              
              #Here we concat (x1,x2)
                  
              self.hidden_layer = nn.Linear(512, 512)
              self.bn3 = nn.BatchNorm1d(512)
              
        self.hidden2 = nn.Linear(512,256) 
        self.bn4 = nn.BatchNorm1d(256)
           
        self.output_layer = nn.Linear(256, output_size)
        
        self.relu = nn.ReLU()
        
        
            
    def forward(self, x):
        # print("INPUT SHAPE IN THE DQN ", x.shape)
        #print("\nInput states\n", x)
        #print("")
        
        # Separate input tensor
        input1 = x[:, 0:self.feature_dim]
        input2 = x[:, self.feature_dim:]
        
        #1) Forward first set of features
        x1 = self.relu(self.bn1(self.input_layer1(input1)))        
        # print("INPUT 1: ", input1.shape)
        
        #2.1) Forward Z hidden state
        if cfg.Z_hidden_state:
            x2 = self.relu(self.bn2(self.input_layer2(input2)))
            
            # 2.1.2) Concatenate features    
            x = torch.cat((x1, x2), 1) #Concat after 1st layer pass
        
            # 3) Forward concat feats in hidden
            x = self.relu(self.bn3(self.hidden_layer(x)))
        
        #2.2) Ignore Z hidden state
        else:
            x = x1
            x = self.relu(self.bn5(self.hidden1(x)))

        #3) Second hidden layer and output layer
        x = self.relu(self.bn4(self.hidden2(x)))
        x = self.output_layer(x)
        
        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
        

#----------------------------------        
        
        
        
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))        
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def show_batch(self, batch_size):
        """
        Prints 'batch_size' random elements stored in the memory.
        Input:
            batch_size: number of entries to be displayed.
        """
        for i in range(batch_size):
            print(random.sample(self.memory, 1), "\n")
        return 0
    
    def __len__(self):
        return len(self.memory)            

