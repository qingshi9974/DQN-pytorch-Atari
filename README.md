# DQN-pytorch-Atari
Implement DQN and DDQN algorithm on Atari games，such as BreakoutNoFrameskip-v4, PongNoFrameskip-v4,BoxingNoFrameskip-v4.


## Requirements
- cuda 10.1
- python 3.7.6
- gym 0.17.6
- pytorch 1.5.0
- tdqm



## Usage Example
```
$ python main.py --env_name BreakoutNoFrameskip-v4-v2
```
## Results without tricks in [OpenAI Baselines](https://github.com/openai/baselines)
### BreakoutNoFrameskip-v4  
![image](https://github.com/qingshi9974/DQN-pytorch-Atari/blob/master/images/3.png)![image](https://github.com/qingshi9974/DQN-pytorch-Atari/blob/master/images/4.png)![image](https://github.com/qingshi9974/DQN-pytorch-Atari/blob/master/images/5.png)

## Results with tricks in [OpenAI Baselines](https://github.com/openai/baselines)
### BreakoutNoFrameskip-v4  
![image](https://github.com/qingshi9974/DQN-pytorch-Atari/blob/master/images/10.png)
### PongNoFrameskip-v4  
![image](https://github.com/qingshi9974/DQN-pytorch-Atari/blob/master/images/15.png)
### BoxingNoFrameskip-v4  
![image](https://github.com/qingshi9974/DQN-pytorch-Atari/blob/master/images/16.png)
