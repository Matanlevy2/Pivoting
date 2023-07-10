#!/usr/bin/env python3
from gc import callbacks
from re import A
from gym.envs.registration import register
import gym   
from train_manipulator.Kinova_Env import ManipulatorEnv 
import rospy
import math
import pickle
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback



if __name__ == '__main__':
	

	register(
            id='Manipulator-v0',
            entry_point='train_manipulator.Kinova_Env_pos_ddpg:ManipulatorEnv',
            max_episode_steps = 130,
        )

	tmp_path = "/home/matan/catkin_ws/src/train_manipulator/src/train_manipulator/PPO_out"
	# set up logger
	new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

	checkpoint_callback = CheckpointCallback(save_freq=20000,save_path=tmp_path,name_prefix='PPO')
	env = gym.make('Manipulator-v0')
	
	n_actions = env.action_space.shape[-1]

	policy_kwargs = dict(activation_fn = th.nn.ReLU, net_arch=([256,256]))
	action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma= 0.1 * np.ones(n_actions))
	
	#model = DDPG("MlpPolicy", env,verbose=1,learning_rate = 0.001,gamma = 0.99,policy_kwargs = policy_kwargs)
	
	#model = DDPG("MlpPolicy",env, verbose=1,learning_rate = 0.001, gamma = 0.99,device='cpu',policy_kwargs=policy_kwargs,batch_size=256)
	
	model = PPO("MlpPolicy", env,verbose=1,learning_rate = 0.0001,gamma = 0.99,policy_kwargs = policy_kwargs)
	
	model.set_logger(new_logger)

	model.learn(total_timesteps= 1000000,callback = checkpoint_callback)

	model.save("PPO_agent")
	
	print ("done")

	rospy.spin()

