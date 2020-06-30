import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import gym
import random
from collections import namedtuple
from atari_wrappers import wrap_deepmind,make_atari
Experience = namedtuple('Experience',
                        ('s', 'a', 'r', 's_','d'))
BATCH_SIZE = 32
MEMORY_SIZE = 1000000
LEARN_START = 50000
TARGET_NET_UPDATE_FREQUENCY = 10000
LR = 0.0000625
GAMMA = 0.99
EPSILON_START  = 1
EPSILON_END = 0.1
seed = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, num_actions,seed):
        super(Net,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8 , stride=4)  # (84-8)/4 +1 = 20  16*20*20
        #self.conv1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4 , stride=2)   # (20-4)/2 +1 = 9 32*9*9
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 , stride=1)  
        #self.conv2.weight.data.normal_(0, 0.1)
        self.fc1 = nn.Linear(3136 ,512)
        #self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 = nn.Linear(512 ,num_actions)
        #self.fc2.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)


    def forward(self,x):
        x = x/255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,3136)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            #m.bias.data.fill_(0.1)

class DQN:
    def __init__(self,env):
        self.env_raw = make_atari(env)
        self.env =  wrap_deepmind(self.env_raw, frame_stack=False, episode_life=True, clip_rewards=True)
        self.num_actions = self.env.action_space.n
        self.target_net = Net(self.num_actions ,seed).to(device)
        self.eva_net = Net(self.num_actions ,seed).to(device)
        self.eva_net.apply(self.eva_net.init_weights)
        self.target_net.load_state_dict(self.eva_net.state_dict())
        self.target_net.eval()
        self.memory_counter = 0
        self.state_counter = 0
        self.memory = []
        self.optimizer = optim.Adam(self.eva_net.parameters(),lr=LR,eps = 1.5e-4)
        self.epsilon = EPSILON_START
        self.state_size = 5   #4+1




    def state_initialize(self):
        self.state_buffer = []
        img = self.env.reset()

        for i in range(self.state_size):
            self.state_buffer.append(img)

        return self.state_buffer[1:5]

    def choose_action(self,x,train = True):
        if train == True:          
            if len(self.memory)>=LEARN_START:
                self.epsilon -= (EPSILON_START-EPSILON_END)/MEMORY_SIZE
                self.epsilon = max(self.epsilon,EPSILON_END)
            epsilon = self.epsilon
        else:
            epsilon = 0.05
        if np.random.uniform() > epsilon :
            x = torch.unsqueeze(torch.tensor(np.array(x,dtype=np.float32),device=device,dtype=torch.float32),0).to(device)      
            q_value = self.eva_net(x).detach()           
            action = torch.argmax(q_value).item()
        else:
            action = self.env.action_space.sample()

        return action

    def store_transition(self,s,a,r,s_,d):
        self.state_counter += 1
        exp = [s,a,r,s_,d]
        if len(self.memory) >= MEMORY_SIZE:
            self.memory.pop(0)
        self.memory.append(exp)


    def learn(self):



        sample = random.sample(self.memory , BATCH_SIZE)
        batch = Experience(*zip(*sample))

        b_s = torch.tensor(np.array(batch.s,dtype=np.float32),device=device,dtype=torch.float32).to(device)
        b_a = torch.tensor(batch.a,device=device).unsqueeze(1).to(device)
        b_r = torch.tensor(np.array(batch.r,dtype=np.float32),device=device,dtype=torch.float32).unsqueeze(1).to(device)
        b_s_ = torch.tensor(np.array(batch.s_,dtype=np.float32),device=device,dtype=torch.float32).to(device)
        b_d = torch.tensor(np.array(batch.d,dtype=np.float32),device = device,dtype=torch.float32).unsqueeze(1).to(device)

        
        q_eval = torch.gather(self.eva_net(b_s),1,b_a)
        avg_q = torch.sum(q_eval.detach())/BATCH_SIZE
        q_eval = q_eval.to(device)
       
        #ddqn
        #argmax = self.eva_net(b_s_).detach().max(1)[1].long()

        #q_next = self.target_net(b_s_).detach().gather(1,torch.unsqueeze(argmax,1))
        q_next = self.target_net(b_s_).detach()   #target网络不更新
        q_next = q_next.to(device)
       
        q_target = b_r + GAMMA*q_next.max(1)[0].unsqueeze(1)*(-b_d+1)

   
        #q_target = b_r + GAMMA*q_next*(-b_d+1)
   
        loss = F.mse_loss(q_eval,q_target)
        
        self.optimizer.zero_grad()

        loss.backward()
        #for param in self.eva_net.parameters():
            #param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        return loss.item(),avg_q.item()
