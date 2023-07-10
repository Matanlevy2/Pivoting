#!/usr/bin/env python3

import rospy
import gym
import random
import time
import numpy as np
from sensor_msgs.msg import JointState
from control_msgs.msg import JointControllerState
from std_srvs.srv import Empty
from std_msgs.msg import Float64,Bool
from geometry_msgs.msg import WrenchStamped
from gazebo_msgs.srv import SetJointProperties,GetJointProperties,SetPhysicsProperties,GetLinkProperties,SetLinkProperties,ApplyJointEffort
from gazebo_msgs.msg import ODEJointProperties
from geometry_msgs.msg import Pose
from gym.utils import seeding
from gym import spaces

                    
class ControllersHandle():
	def __init__(self):
		self.pub1 = rospy.Publisher('/command_topic_joint_6', Float64, queue_size=10,latch=True)
		self.pub2 = rospy.Publisher('/command_topic_joint_7', Float64, queue_size=10,latch=True)
		self.pub_reset = rospy.Publisher('reset',			  Bool,	   queue_size=10,latch=True)
		self.damp_pub = rospy.Publisher('/set_cf',			  Float64, queue_size=10,latch=True)
		self.grip_pub = rospy.Publisher('/friction',		  Float64, queue_size=10,latch=True)
		
		self.publisher_array =  []

		self.publisher_array.append(self.pub1)
		self.publisher_array.append(self.pub2)
		
		self.rate = rospy.Rate(20)
	

	def set_init_position(self):
		com3 = 0.0
		com4 = 0.0

		
		self.pub1.publish(com3)
		self.pub2.publish(com4)


class GazeboManager():
	def __init__(self):
		self.unpause =  rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause   = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

	def pause_sim(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		self.pause()

	def unpause_sim(self):
		rospy.wait_for_service('/gazebo/unpause_physics')
		self.unpause()
		

class ManipulatorEnv(gym.Env):
	def __init__(self):
		random.seed(0)
		low_action = np.array([-0.87,
							   -0.87,
							   0])

		high_action = np.array([0.87,
								0.87,
								1])

		
		low_obsv = np.ones((3)) * (-5) 
	
		rospy.init_node('manipulator', anonymous=True)
		
		self.controller_handle = ControllersHandle()
		
		self.gazebo_manager    = GazeboManager()
		
		self.joint7_prop_msg = ODEJointProperties()

		self.action_space      = spaces.Box(low_action,high_action)

		self.observation_space = spaces.Box(low_obsv,-low_obsv)

		self.goal           = 0.25
		self.curr_action    = [0.0 ,0.0, 0.0]
		self.goal_counter   = 0
		self.counter = 0
		msg = rospy.wait_for_message("/gen3/joint_states",JointState,timeout=10)
		self.last_action = [0.0, 0.0, 0.0]
		self.joint_state = []
		self.prev_joint_state = msg 
		self.input_delay = np.zeros([20,3])

		self.prev_time_stamp = rospy.get_time()
		self.error      = 0
		self.prev_error = 0
		self.error_f    = 0
		self.error_int  = 0
		self.position_com = [0,0]
		self.srv_get = rospy.ServiceProxy('/gazebo/get_link_properties',GetLinkProperties)
		self.srv_set = rospy.ServiceProxy('/gazebo/set_link_properties',SetLinkProperties)
		rospy.Subscriber("/gen3/joint_states", JointState, self._joints_state_callback)


	def joint6_vel_clbk(self,data):
		self.joint6_vel = data.data
	
	def joint7_vel_clbk(self,data):
		self.joint7_vel = data.data


	def step(self,action):
		self.gazebo_manager.unpause_sim()		
		self._set_action(action)
		

		self.gazebo_manager.pause_sim()
		self.counter += 1
		obs    = self._get_obs()
		reward = self._compute_reward(obs)
		done   = 0
		info   = {}

		return obs, reward, done, info


	def set_sim_parametrs(self):
		self.mass_update()
		self.damp_update()
		if (self.goal_counter > 0):
			self.goal = round(1.8 * ((random.random() - 0.5)),3)
			self.goal_counter = 0
			print(self.goal)
			pass		

		self.goal_counter += 1
		self.delay = 3 + (random.randint(0,2) - 1)

	
	def damp_update(self):
		damp = 0.003 * (random.random() - 0.5) + 0.01
		self.controller_handle.damp_pub.publish(damp)
	

	def mass_update(self):
		msg = self.srv_get('point_mass')
		pose = Pose()
		position = msg.com.position
		orientation = msg.com.orientation
		pose.position = position
		pose.orientation = orientation
		mass =  0.03 * np.random.random() + 0.01
	
		ixx = msg.ixx
		ixy = msg.ixy
		ixz = msg.ixz
		iyy = msg.iyy
		iyz = msg.iyz
		izz = msg.izz
		gravity_mode = True
		ans = self.srv_set(link_name = 'point_mass',mass = mass, com = pose, gravity_mode = gravity_mode, ixx = ixx, ixy = ixy, ixz = ixz, iyy = iyy, iyz = iyz, izz = izz)
	

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	

	def reset(self):
		self.gazebo_manager.unpause_sim()
		self.counter = 0
		self.error_int = 0
		self.action_int = 0
		time.sleep(0.2)
		self.controller_handle.pub_reset.publish(True)
		self.controller_handle.grip_pub.publish(0)
		self.position_com = [0,0]
		for i in range(200):
			self.controller_handle.set_init_position()
			rospy.sleep(0.01)
		
		self.controller_handle.pub_reset.publish(False)

		self.set_sim_parametrs()

		self.gazebo_manager.pause_sim()
		
		self.error_f = self.goal - self.joint_states.position[7]
		
		obs = self._get_obs()

		return obs

	def close(self):
		pass


	def _get_obs(self):
		observation  = []
		
		self.error = self.goal - self.joint_states.position[7]

		self.error_d = (self.error - self.error_f)

		self.error_f += self.error_d * 0.05

		
		#joint positions
		observation.append(round(self.joint_states.position[5],3))
		observation.append(round(self.joint_states.position[6],3))
		
		observation.append(round(self.error,3))
		observation.append(round(self.joint_states.position[7],3))
		
		self.prev_joint_state = self.joint_states
		
		self.prev_error = self.error

		return observation
		

	def _set_action(self,action):
		self.curr_action = action
		self.move_joints(self.curr_action)

		self.controller_handle.rate.sleep()
		

	def move_joints(self,action):
		i = 0
		self.input_delay[0,0] = action[0]
		self.input_delay[0,1] = action[1]
		self.input_delay[0,2] = action[2]
		
		self.position_com[0] += action[0] * 0.05
		self.position_com[1] += action[1] * 0.05

		self.position_com[0] = max(min(self.position_com[0],0.3),-0.3)
		self.position_com[1] = max(min(self.position_com[1],2.0),-2.0)		
		self.input_delay = np.roll(self.input_delay,1,axis = 0)
		
		for publisher_object in self.controller_handle.publisher_array:
			joint_value = Float64()
			joint_value.data = self.position_com[i]
			publisher_object.publish(joint_value)
			i += 1

		d = random.randint(1,3)
		self.controller_handle.grip_pub.publish(self.input_delay[d,2])
	
	def limiter(self,input,max,min):
		if input >= max:
			out = max
		elif input<= min:
			out = min
		else:
			out = input
		return out
			
		

	def _compute_reward(self,obs):
		reward  = 1    - 3 * abs(obs[2])
		reward += 0.5*(1  - 2*abs((1/0.3)*(obs[0])))
		reward += 0.5*(1  - 2*abs((1/2.0)*(obs[1])))
		reward -= 0.6*abs(self.curr_action[0])
		reward -= 0.6*abs(self.curr_action[1])
		return reward

	
	def _joints_state_callback(self,data):
		self.joint_states = data
		
		
