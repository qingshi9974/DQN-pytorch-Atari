from datetime import datetime
import re
import os
import torch


def date_in_string():
    time = str(datetime.now())
    time = re.sub(' ', '_', time)
    time = re.sub(':', '', time)
    time = re.sub('-', '_', time)
    time = time[0:15]
    return time



def save_log_loss(i_steps, loss, avg_qscore):
    path = os.getcwd()+ '/log'
    with open(path + '/log_loss_6_10.txt', 'a') as outfile:
        outfile.write(date_in_string() + '\t' + str(i_steps) + '\t' + str(loss) + '\t' + str(avg_qscore)  + '\n')
    return

def save_log_score(i_episodes,mean_r,max_r):
    path = os.getcwd()+'/log'
    with open(path + '/log_score_6_10.txt', 'a') as outfile:
        outfile.write(date_in_string() + '\t' + str(i_episodes) + '\t' + str(mean_r) + '\t' + str(max_r) +'\n')
    return


def save_model_params(model):
    path = './model'
    torch.save(model.state_dict(), path + '/DQN_breakout'  + '.pkl')
    return
