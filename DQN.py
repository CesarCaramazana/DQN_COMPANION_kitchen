import torch
import torch.nn as nn
import random
from collections import namedtuple, deque
import config as cfg


class DQN(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        print("DQN without Z variable")
        
                        
        self.input_layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size)            
        )       
        
            
    def forward(self, x):
        # print("INPUT SHAPE IN THE DQN ", x.shape)
        #print("\nInput states\n", x)
        #print("")


        x = self.input_layer1(x)  
        x = self.hidden1(x)
        x = self.output_layer(x)
        
        return x






class DQN_Z(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(DQN_Z, self).__init__()
               
        print("DQN with Z variable")
        
        self.feature_dim = input_size - cfg.Z_HIDDEN
        
        # print("DQN input size: ", input_size)
        # print("DQN feature: ", self.feature_dim)
        # print("DQN Zhidden: ", cfg.Z_HIDDEN)
        
                        
        self.input_layer1 = nn.Sequential(
            nn.Linear(input_size - cfg.Z_HIDDEN, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.input_layer2 = nn.Sequential(
            nn.Linear(cfg.Z_HIDDEN, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()            
        )
           
        self.hidden1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size)            
        )       
        
            
    def forward(self, x):
        # print("INPUT SHAPE IN THE DQN ", x.shape)
        #print("\nInput states\n", x)
        #print("")
        
        input1 = x[:, 0:self.feature_dim]
        input2 = x[:, self.feature_dim:]
                
        x1 = self.input_layer1(input1)
        x2 = self.input_layer2(input2)
        
        x = torch.cat((x1, x2), 1)         
        
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output_layer(x)

        
        return x



class DQN_LateFusion(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(DQN_LateFusion, self).__init__()
               
        print("DQN Late fusion")
                
        self.embedding_size = 32
        
                        
        self.input_AcPred = nn.Sequential(
            nn.Linear(33, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()
        )
        
        self.input_AcRec = nn.Sequential(
            nn.Linear(33, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()            
        )
        
        self.input_VWM = nn.Sequential(
            nn.Linear(44, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()            
        )
        
        self.input_OiT = nn.Sequential(
            nn.Linear(23, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()            
        )
        
        self.input_TempCtx = nn.Sequential(
            nn.Linear(7, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()            
        )
        
        self.input_Z = nn.Sequential(
            nn.Linear(1024, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()            
        )
           
        self.hidden1 = nn.Sequential(
            nn.Linear(self.embedding_size*6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size)            
        )       
        
            
    def forward(self, x):
        # print("INPUT SHAPE IN THE DQN ", x.shape)
        #print("\nInput states\n", x)
        #print("")
        
        ac_pred = x[:, 0:33]
        ac_rec = x[:, 33:66]
        vwm = x[:, 66:110]
        oit = x[:, 110:133]
        temp_ctx = x[:, 133:140]
        z = x[:, 140:]
        
        x1 = self.input_AcPred(ac_pred)
        x2 = self.input_AcRec(ac_rec)
        x3 = self.input_VWM(vwm)
        x4 = self.input_OiT(oit)
        x5 = self.input_TempCtx(temp_ctx)
        x6 = self.input_Z(z)
              
        x = torch.cat((x1, x2, x3, x4, x5, x6), 1)         
        
        x = self.hidden1(x)
        x = self.hidden2(x)
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

