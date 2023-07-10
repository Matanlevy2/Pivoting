#!/usr/bin/env python3
import os
from gym.envs.registration import register
import gym   
from train_manipulator.Kinova_Env import ManipulatorEnv 
import rospy
import math
import time
import torch as T
import csv
import torch
from BC import NeuralNetwork
from BC_mixed import NeuralNetworkMixed,GaussianPolicy
from Test_data_loader_seperate import Discriminator
import numpy as np
from std_msgs.msg import Float64,Bool

class Controller():
	def __init__(self,kp = 0.6,kd =0.1,tau = 1):
		self.Kd = kd
		self.Kp = kp
		self.action = [0,0,0]
		self.control_state = 0
		self.counter = 0

	def reset(self):
		self.control_state = 0

	def ControlSM(self,pos_error):
		if abs(pos_error) < 0.03:
			self.control_state = 3

		if self.control_state == 0:
			self.counter += 1
			if self.counter > 5:
				self.counter = 0
				self.control_state = 1
		
		elif self.control_state == 1:
			if abs(pos_error) < 0.02:
				self.counter += 1
				if self.counter > 5:
					self.control_state = 2
					self.counter = 0
		
		elif self.control_state == 2:
			self.counter += 1
			if self.counter > 5:
				self.counter = 0
				self.control_state = 3

	def calc_action(self,obs):
		action = [0,0,0]
		if self.control_state == 0:
			action1 = 1.5 * (0.2 - obs[2])
			action = [action1,0,0]
		
		elif self.control_state == 1:
			action1 = 1.5 * (0.2 - obs[2])
			action2 = self.Kp * obs[0] + self.Kd * obs[1]
			action = [action1,action2,0]
		
		elif self.control_state == 2:
			action2 = -self.Kp * obs[0]
			action1 = 1.5 * (0.2 - obs[2])
			action = [action1,action2,1]
		
		elif self.control_state == 3:
			action[0] = 1.5 * (0 - obs[2])
			action[1] = 1.5 * (0 - obs[3])
			action[2] = 1
		
		self.action = action
		
def limiter(xin,uplim,lowlim):
	return min(max(xin, lowlim), uplim)

if __name__ == '__main__':
	#open file for logging
	pub_reset = rospy.Publisher('init_rand', Float64,queue_size=5,latch=True)
	log_files 	 	= True
	folder_name     = 'size_64_50_fixed_5'
	run_PID 	 	= False
	ActionNoiseFlag = False
	numOfRuns 		= range(100)

	register(
            id='Manipulator-v1',
            entry_point='train_manipulator.Kinova_Env_pos:ManipulatorEnv'
        )

	env = gym.make('Manipulator-v1')
	
	episode_reward = 0

	model = NeuralNetwork(num_in=3,num_out=3,size_one=64,size_two=64)

	if not(run_PID):
		model.load_state_dict(torch.load('size_64_50_noise_optimal_model.pt',map_location=torch.device('cpu')))
		print("Model Loaded")
		
	pub_reset.publish(1)
	pub_reset.publish(0)
	C = Controller()

	
	for	n in numOfRuns:
		action = [0,0,0]

		name = './' + folder_name + '/ExpertNoise_%s.csv' % (n)
		
		os.makedirs(os.path.dirname(name), exist_ok=True)
		
		if log_files:
			f = open(name,'w')
			writer = csv.writer(f)
			Header = ['error','error_d','theta1','theta2','prev_theta1','prev_theta2','goal','angle','prev_angle','action1','action2','action3']
			writer.writerow(Header)
		
		obs = env.reset()
		prev_action = []
		start_fix = False
		for i in range(160): 
			row = []
			
			C.ControlSM(obs[0])
			C.calc_action(obs)
			
			if (ActionNoiseFlag and np.mod(n,1)==0):
				action_noise =  0.2 * (np.random.rand(1,2) - 0.5)
			else:
				action_noise =  0.00 * (np.random.rand(1,2) - 0.5)

			action_noise = (action_noise.tolist())
			
			#Run learned model
			if not(run_PID):
				obs = torch.tensor(obs, dtype = torch.float32)
				obs = obs[[0,2,3]]
				
				ModelAction = model(obs)

				ModelActionArray = ModelAction.detach().numpy()

				action[0] += (ModelActionArray[0] + action_noise[0][0]) * 0.05
				action[1] += (ModelActionArray[1] + action_noise[0][1]) * 0.05
				action[2] =  ModelActionArray[2]
				
				
			#Run PID controller
			else:
				obs = torch.tensor(obs, dtype = torch.float32)
				obs = obs[[2,3,7]]
				action_tensor = torch.tensor(C.action)
				state_action = torch.concat((obs,action_tensor))
		
	
				action[0] += (C.action[0] + action_noise[0][0]) * 0.05 
				action[1] += (C.action[1] + action_noise[0][1]) * 0.05
				action[2] =  C.action[2]
			

	
				
		
			
			action[0] = limiter(action[0],0.3,-0.3)
			action[1] = limiter(action[1],1.5,-1.5)

			obs, reward, done, info = env.step(action)
			
			for iobs in range(len(obs)):
				row.append(obs[iobs])
			
			if (run_PID):
				row.append(C.action[0])
				row.append(C.action[1])
				row.append(C.action[2])
				row.append(0)
				row.append(0)
				row.append(0)
			else:
				row.append(ModelActionArray[0])
				row.append(ModelActionArray[1])
				row.append(ModelActionArray[2])



			if (run_PID):
				prev_action = C.action
			else:
				prev_action = ModelActionArray

			if (log_files):
				writer.writerow(row)
		
		env.reset()

		C.reset()

	rospy.spin()

