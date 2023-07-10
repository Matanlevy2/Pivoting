#! /usr/bin/env python3
import rospy

import numpy as np
import math
import time
import csv
import argparse
import random
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from stable_baselines3 import TD3
from kortex_driver.srv import *
from kortex_driver.msg import *
from control_msgs.msg  import GripperCommandActionGoal, GripperCommandGoal,GripperCommand
from std_msgs.msg import Header
from actionlib_msgs.msg import GoalID
from BC import NeuralNetwork 
import torch
class Pivoting():
    def __init__(self,goal):
        msg_joints     = rospy.wait_for_message('/my_gen3/joint_states',JointState,timeout=5)
        msg_angle      = rospy.wait_for_message('/angle',Float64,timeout=5)  
        self.sub_manipulator  = rospy.Subscriber('/my_gen3/joint_states',JointState,self.joint_callback)
        self.sub_object       = rospy.Subscriber('/angle',Float64,self.angle_callbcak)
        self.goal = goal
        

        self.joint5 = msg_joints.position[5]
        self.joint6 = msg_joints.position[6]
        self.joint5_rate = msg_joints.velocity[5]
        self.joint6_rate = msg_joints.velocity[6]

        self.prev_joint5 = self.joint5
        self.prev_joint6 = self.joint6
        self.prev_joint5_rate = self.joint5_rate
        self.prev_joint6_rate = self.joint6_rate

        self.action = [0.0, 0.0, 0.0]
        self.angle = msg_angle.data
        self.prev_error = goal - self.angle
        self.prev_angle = self.angle

        self.i = 0
        self.obs = []
     
        print("init")

    def joint_callback(self,data):
        self.joint5 = data.position[5]
        self.joint6 = data.position[6]
        self.joint5_rate = data.velocity[5]
        self.joint6_rate = data.velocity[6]
       
        pass

    def angle_callbcak(self,data):
        self.angle = data.data 
        pass

    def angle_rate_callback(self,data):
        self.angle_rate = data.data
        pass
    
    def set_observations(self):
        obs = []
        error   = self.goal - self.angle
        error_d = (error - self.prev_error)/0.05

        obs.append(error)
        obs.append(error_d)
        obs.append(self.joint5)
        obs.append(self.joint6)
        obs.append(self.prev_joint5)
        obs.append(self.prev_joint6)
 
        self.obs = obs

        self.prev_joint5       = self.joint5 
        self.prev_joint6       = self.joint6

    def calc_actions(self):
        obs = []
        error   = self.goal - self.angle
        error_d = (error - self.prev_error)/0.05

        obs.append(error)
        obs.append(error_d)
        obs.append(self.joint5)
        obs.append(self.joint6)
        obs.append(self.prev_joint5)
        obs.append(self.prev_joint6)
 
        self.obs = obs
        action = [0,0,0]

        self.prev_joint5       = self.joint5 
        self.prev_joint6       = self.joint6

        self.action = action
        pass

 

    def run(self):
        self.calc_actions()
        

    def limiter(self,input,max,min):
        if input >= max:
            output = max
        elif input <= min:
            output = min
        else:
            output = input
        return output
        
class PosAction():  
    def __init__(self):
        self.PubV      = rospy.Publisher('/my_gen3/in/joint_velocity',Base_JointSpeeds,queue_size=10)
        self.position_limiter_high = [0.5,
                                      1.7]

        self.position_limiter_low  = [-0.5,
                                     -1.7]

        msg = rospy.wait_for_message('/my_gen3/joint_states',JointState,timeout=5)
        self.sub = rospy.Subscriber('/my_gen3/joint_states',JointState,self.joint_callback)
        self.goal = [0, 0]
        self.joint_pos  = self.goal
        self.joint_vel = [msg.velocity[5],msg.velocity[6]]
        self.prev_pos =  self.goal
        self.prev_vel = self.joint_vel
        self.rate = rospy.Rate(20)
        self.vel_com = []
        print("init control")
    
    def joint_callback(self,data):
        self.joint_pos = []
        self.joint_vel = []

        self.joint_pos.append(data.position[5])
        self.joint_pos.append(data.position[6])

        self.joint_vel.append(data.velocity[5])
        self.joint_vel.append(data.velocity[6])
        
        
		    
    def calc_vel_com(self,zero_vel):
        gain = [4.0, 4.0]
        if (zero_vel):
            vel_com = np.array([0,0])
        else:
            vel_com = np.multiply(gain , (np.asarray(self.goal) - np.asarray(self.joint_pos)))
            
            
        self.vel_com = vel_com
        pass
	
    def set_goal(self,input):
        self.goal = input
        pass
    
        
  
    def CreateJointVelMsg(self):
        Joint_speeds = []
        BaseMsg = Base_JointSpeeds()
        for i in range(2):
            msg = JointSpeed()
            msg.joint_identifier = i + 5
            msg.value = self.vel_com[i]
            msg.duration = 2
            Joint_speeds.append(msg)
        
        BaseMsg.joint_speeds = Joint_speeds
        
        self.pub_msg = BaseMsg
        
        
    def publish_position(self):
        self.CreateJointVelMsg()
        self.PubV.publish(self.pub_msg)
    
    def on_shutdown(self):
        self.calc_vel_com(zero_vel=True)
        self.CreateJointVelMsg()
        self.PubV.publish(self.pub_msg)
        pass


class GripController_new():
    def __init__(self):
        rospy.wait_for_message('/my_gen3/joint_states',JointState,timeout=5)
        self.msg_pub = rospy.Publisher('/my_gen3/robotiq_2f_85_gripper_controller/gripper_cmd/goal',GripperCommandActionGoal,queue_size=5,latch=True)
        self.grip_msg = []
        self.position = 0
        print("init")
        pass

    def run(self):
        self.calc_actions()
        
        
    def CreateGripMsg(self):
        grip_msg = GripperCommandActionGoal()
        goal_id  = GoalID()
        msg_head = Header()
        goal     = GripperCommandGoal()
        command  = GripperCommand()

        command.position  = self.position
        goal.command      = command
        grip_msg.header   = msg_head
        grip_msg.goal_id  = goal_id
        grip_msg.goal     = goal
        self.grip_msg     = grip_msg
        
        
    def send_grip_comm(self):
        self.CreateGripMsg()
        self.msg_pub.publish(self.grip_msg)

class Controller():
    def __init__(self,kp = 0.5,kd =0.2):
        self.Kd = kd
        self.Kp = kp
        self.action = [0,0,0]
        self.control_state = 0
        self.counter = 0
        
    def reset(self):
        self.control_state = 0

    def ControlSM(self,pos_error):
        if self.control_state == 0:
            if abs(pos_error) < 0.01:
                self.control_state = 3
            else:
                self.counter += 1
                if self.counter > 5:
                    self.counter = 0
                    self.control_state = 1
		
        elif self.control_state == 1:
            if abs(pos_error) < 0.01:
                self.control_state = 2
                pass
		
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
            print(obs[0])
            action2 = self.Kp * obs[0] + self.Kd * obs[1]
            action = [action1,action2,0]
		
        elif self.control_state == 2:
            action2 = self.Kp * obs[0] + self.Kd * obs[1]
            action1 = 1.5 * (0.2 - obs[2])
            action = [action1,action2,1]
		
        elif self.control_state == 3:
            action[0] = 1.5 * (0 - obs[2])
            action[1] = 1.5 * (0 - obs[3])
            action[2] = 1
		
        self.action = action
    
    def run(self,obs):
        self.ControlSM(obs[0])
        self.calc_action(obs)
        
    def add_action_noise(self):
        action_noise =  0.4*(np.random.rand(1,2) - 0.5)
        action_noise = (action_noise.tolist())
        self.action[0] += float(action_noise[0][0])
        self.action[1] += float(action_noise[0][1])

def limiter(xin,uplim,lowlim):
	return min(max(xin, lowlim), uplim)

if __name__ == "__main__":
    rospy.init_node("Pivot_node")
    times = 1
    #model = NeuralNetwork(num_in=3,num_out=3,size_one=32,size_two=32)
    #model.load_state_dict(torch.load('Lab_32_20_optimal_model.pt',map_location=torch.device('cpu')))
    while not rospy.is_shutdown():
        filename = '/home/matan/recorded_lab_exp_4/' + 'experiment_PD_mass' + str(times) + '.csv'
        print(times)
        f = open(filename, 'w')
        writer = csv.writer(f)
        goal = 0.65*2*(random.random() - 0.5)
        p    = Pivoting(goal)
        v    = PosAction()
        gr   = GripController_new()
        ctrl = Controller(kp = -1.1, kd = 0.0)
        gr.position = 0.44
        gr.send_grip_comm()
        rospy.on_shutdown(v.on_shutdown)
        omega  = 2 * math.pi * 2.5
        action = [0,0,0]

        sign = 1.0
        prev_action_3 = 1
        count = 0 
        action = [0,0,0]
        for i in range(150):
            row = []
        
            p.goal = goal
        
            p.set_observations()
            
            ctrl.run(p.obs)
            
            #obs_model = torch.tensor(p.obs, dtype = torch.float32)
            #obs_model = obs_model[[0,2,3]]
            #model_action = model(obs_model)
            #print(model_action)
            #ModelActionArray = model_action.detach().numpy()

            #row.append(ModelActionArray[0])
            #row.append(ModelActionArray[1])
            
            #ctrl.add_action_noise()
        
            action[0] += ctrl.action[0] * 0.05
            action[1] += ctrl.action[1] * 0.05
            action[2] =  ctrl.action[2]
            
            row.append(action[0])
            row.append(action[1])
            row.append(action[2])
            
            #action[0] += ModelActionArray[0] * 0.05
            #action[1] += ModelActionArray[1] * 0.05
            #action[2] =  ModelActionArray[2]


            action[0] = limiter(action[0],0.3,-0.3)
            action[1] = limiter(action[1],1.5,-1.5)
        
            #set gripper acion
            if (prev_action_3 <0.5) and (action[2] >= 0.5):
                gr.position = 0.44
                gr.send_grip_comm()
                print("close")
            elif (prev_action_3 >= 0.5) and (action[2] < 0.5):
                print("open")
                gr.position = 0.36
                gr.send_grip_comm()

        
            v.set_goal(action[0:2])
        
            v.calc_vel_com(zero_vel=False)

            prev_action_3 = action[2]

            v.publish_position()
       
            row.append((rospy.Time.now().to_sec()))

            row.append(goal)
        
            for id in range(len(p.obs)):
                row.append(p.obs[id])

     
            writer.writerow(row)
            v.rate.sleep()
        
        f.close()
        times = times + 1

    