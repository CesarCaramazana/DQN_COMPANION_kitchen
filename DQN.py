import torch
import torch.nn as nn
import random
from collections import namedtuple, deque


class DQN(nn.Module):
	
	def __init__(self, input_size, output_size):
		super(DQN, self).__init__()
		
		self.feature_dim = 133 #First features and Z variable separated
		
		self.input_layer1 = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.ReLU())
		self.input_layer2 = nn.Sequential(nn.Linear(input_size-self.feature_dim, 256), nn.ReLU())
		
		
		self.hidden_layer = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
				
		self.output_layer = nn.Linear(256, output_size)
		
		
		
			
	def forward(self, x):
		#print("INPUT SHAPE IN THE DQN ", x.shape)
		#print("\nInput states\n", x)
		#print("")
		
		# Separate input tensor
		input1 = x[:, 0:self.feature_dim]
		input2 = x[:, self.feature_dim:]

		x1 = self.input_layer1(input1)
		x2 = self.input_layer2(input2)

		x = torch.cat((x1, x2), 1) #Concat after 1st layer pass
		
		x = self.hidden_layer(x)
		x = self.output_layer(x)
		
		
		#print("output X device: ", x.device)
		
		#print("OUTPUT SHAPE: ", x.shape)
		#print("\nOutput Q values\n", x)
		#print("OUTPUT TYPE: ", x.dtype)
		
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
