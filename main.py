from dqn import DQN
import numpy as np
import gym
from datetime import datetime
import torch
import utilities as u
from atari_wrappers import wrap_deepmind, make_atari
import argparse
LEARN_START = 50000

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="BreakoutNoFrameskip-v4",
                    help='name of Mujoco environement')
args = parser.parse_args()
myDQN = DQN(args.env_name)
NUM_STEPS =50000000
max_r = 0
mean_r= 0
loss=0
avg_q =0
running_loss = 0
max_mean_r = 0
sum_r =0
done =True
i = 0

def evaluate(step, eva_net, env, num_episode=15):
    env = wrap_deepmind(env)
    e_rewards = []
    for i in range(num_episode):
        img = env.reset()
        sum_r = 0
        done = False
        state_buffer=[]
        for i in range(5):
            state_buffer.append(img)
        s = state_buffer[1:5]
        while not done:
            a = myDQN.choose_action(s,train= False)
        
            img,r,done,info = env.step(a)
            sum_r += r
            state_buffer.pop(0)
            state_buffer.append(img)
            s_ = state_buffer[1:5]
            s = s_

        e_rewards.append(sum_r)

    f = open("file.txt",'a') 
    f.write("%f, %d, %d\n" % (float(sum(e_rewards))/float(num_episode), step, num_episode))
    f.close()

progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=50, leave=False, unit='b')
for step in progressive:

    if done and myDQN.env.was_real_done == True:
        mean_r += sum_r
        if sum_r >max_r :
            max_r = sum_r
        #计算分数均值和最大值
        if (i+1)%20 == 0:
            u.save_log_score(i,mean_r/20, max_r)
            print('    ',i,mean_r/20,max_r)
            if mean_r>max_mean_r:
                max_mean_r = mean_r
                u.save_model_params(myDQN.eva_net)
            max_r,mean_r=0,0

        sum_r = 0
        i +=1

    if done :
        s = myDQN.state_initialize()
        img,_,_,_ = myDQN.env.step(1)
 
    
    a = myDQN.choose_action(s)
        
    img,r,done,info = myDQN.env.step(a)
    sum_r += r
    myDQN.state_buffer.pop(0)
    myDQN.state_buffer.append(img)


    s_ = myDQN.state_buffer[1:5]
    myDQN.store_transition(s,a,r,s_,done)
    s = s_

    if len(myDQN.memory)>LEARN_START and myDQN.state_counter%4 :
        loss,avg_q = myDQN.learn()
    running_loss += loss
    if myDQN.state_counter%500 ==0:
        running_loss /= 250
    if myDQN.state_counter%10000 ==0:
        u.save_log_loss(step,loss,avg_q)
        myDQN.target_net.load_state_dict(myDQN.eva_net.state_dict())
    if myDQN.state_counter%500 ==0:
        running_loss =0
    if (myDQN.state_counter+1) %50000 ==0 :
        evaluate(step, myDQN.eva_net, myDQN.env_raw,  num_episode=10)    




  


