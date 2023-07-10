#! /usr/bin/env python3

from re import A
from xmlrpc.client import Boolean
import rospy
import actionlib
import numpy as np
from std_msgs.msg import Float64,Bool
from sensor_msgs.msg import JointState
import random

class PosActionServer():
	def __init__(self,Ts,wn,zeta,vel_limit):
		self.pub1 = rospy.Publisher('/gen3/joint4_position_controller/command', Float64,queue_size=10,latch=True)
		self.pub2 = rospy.Publisher('/gen3/joint5_position_controller/command', Float64,queue_size=10,latch=True)
		self.pub3 = rospy.Publisher('/gen3/joint6_position_controller/command', Float64,queue_size=10,latch=True)
		self.pub4 = rospy.Publisher('/gen3/joint7_position_controller/command', Float64,queue_size=10,latch=True)
		self.pub3_vel = rospy.Publisher('/gen3/joint6_vel',Float64,queue_size=10,latch=True)
		self.pub4_vel = rospy.Publisher('/gen3/joint7_vel',Float64,queue_size=10,latch=True)
		self.pub_friction = rospy.Publisher('/gen3/joint8_position_controller/command', Float64,queue_size=10,latch=True)

		self.sub_goal4  = rospy.Subscriber('/command_topic_joint_4', Float64, self.get_goal, (0))
		self.sub_goal5  = rospy.Subscriber('/command_topic_joint_5', Float64, self.get_goal, (1))
		self.sub_goal6  = rospy.Subscriber('/command_topic_joint_6', Float64, self.get_goal, (2))
		self.sub_goal7  = rospy.Subscriber('/command_topic_joint_7', Float64, self.get_goal, (3))
		self.sub_friction = rospy.Subscriber('friction',Float64,self.get_friction_command)
		self.sub_reset    = rospy.Subscriber('reset',Bool,self.get_reset )
		self.sub_cf 	  = rospy.Subscriber('set_cf',Float64,self.set_cf)
		self.sub_rand    = rospy.Subscriber('init_rand',Float64,self.init_rand)
		self.close_joint = False

		self.A   = []
		self.B   = []
		self.Ts  = Ts
		self.wn  = wn
		self.zeta = zeta
		self.vel_limit = vel_limit
		self.rate = rospy.Rate(200)
		self.goal = [0,0,0,0]
		msg = rospy.wait_for_message('/gen3/joint_states',JointState,timeout=5)
		self.joint_7_sub = rospy.Subscriber('/gen3/joint_states',JointState,self.get_joint_pos)
		self.goal_delay = np.zeros([4,200])
		self.state4 = np.array([msg.velocity[3],0])
		self.state5 = np.array([msg.velocity[4],0])
		self.state6 = np.array([msg.velocity[5],0])
		self.state7 = np.array([msg.velocity[6],0])
		self.goal   = [msg.velocity[3], msg.velocity[4], msg.velocity[5],msg.velocity[6]]
		self.counter = 0
		self.joint_pos_7 = msg.position[6]
		self.joint_pos_6 = msg.position[5]
		self.joint_pos_5 = msg.position[4]
		self.joint_pos_4 = msg.position[3]
		self.tool_vel = 0
		self.tool_effort_int = 0
		self.tool_pos = msg.position[7]
		self.tool_pos_com = msg.position[7]
		self.cf = 0
		self.reset = False
		self.prev_reset = False
		self.init_pos_tool = 0
		self.vel_com_delay = np.zeros([4,300])
		print("init")
		self.define_dynamics()
		random.seed(0)

	def get_goal(self,data,arg):
		self.goal[arg] = data.data
		self.counter = 0
	
	def init_rand(self,data):
		if data.data == 1:
			random.seed(0)
			print("init rand")
		
	def get_reset(self,data):
		if (data.data):
			self.reset = True
		else:
			self.reset = False

	def get_friction_command(self,data):
		if data.data > 0.5:
			self.close_joint = True
			self.tool_pos_com = self.tool_pos
		else:
			self.close_joint = False
			self.tool_effort_int = 0
		

		
	def get_joint_pos(self,data):
		self.joint_pos_7 = data.position[6]
		self.joint_pos_6 = data.position[5]
		self.joint_pos_5 = data.position[4]
		self.joint_pos_4 = data.position[3]
		
		self.tool_vel = data.velocity[7]
		self.tool_pos = data.position[7]


	def set_cf(self,data):
		self.cf = data.data
		pass
	
	def joint_limiters(self):
		pass

	def publish_position(self):
		vel_com = np.asarray(self.goal)
		
		#damping
		cf = self.cf
		damp = -cf * self.tool_vel
		
		if (self.close_joint):
			damp = -0.05 * self.tool_vel
			pos_error = self.tool_pos_com - self.tool_pos
			self.tool_effort_int  +=  0.05 * pos_error
			damp += 0.1 * pos_error + self.tool_effort_int
		else:
			self.tool_effort_int = 0
		
		
		if (self.reset) and not(self.prev_reset):
			self.init_pos_tool = (random.random() - 0.5)
			print(self.init_pos_tool)

		if (self.reset):
			damp = -0.05 * self.tool_vel
			pos_error = self.init_pos_tool -self.tool_pos
			damp += 0.45 * pos_error
			
		self.prev_reset = self.reset
		self.pub_friction.publish(damp)
		

		self.vel_com_delay[:,0] = vel_com
		
		self.vel_com_delay = np.roll(self.vel_com_delay,1,axis = 1)
		self.step(self.vel_com_delay[:,18])
		
		self.pub3.publish(self.state6[0])
		self.pub4.publish(self.state7[0])

		self.pub3_vel.publish(self.state6[1])
		self.pub4_vel.publish(self.state7[1])
		




	def limiter(self,input,max,min):
		if input >= max:
			out = max
		elif input<= min:
			out = min
		else:
			out = input
		return out
			

	def define_dynamics(self):
		A = np.array([[0, 1],[-self.wn**2, -2 * self.zeta * self.wn]])
		B = np.array([0, self.wn**2])
		self.A = np.identity(2) + self.Ts * A
		self.B = B * self.Ts
		pass
	
	def step(self,input):
		self.state4 = np.matmul(self.A ,self.state4) + self.B * input[0]
		self.state5 = np.matmul(self.A ,self.state5) + self.B * input[1]
		self.state6 = np.matmul(self.A ,self.state6) + self.B * input[2]
		self.state7 = np.matmul(self.A ,self.state7) + self.B * input[3]
		
		self.state4[1] = self.limiter(self.state4[1],self.vel_limit,-self.vel_limit)
		self.state5[1] = self.limiter(self.state5[1],self.vel_limit,-self.vel_limit)
		self.state6[1] = self.limiter(self.state6[1],self.vel_limit,-self.vel_limit)
		self.state7[1] = self.limiter(self.state7[1],self.vel_limit,-self.vel_limit)
		
		
	def run(self):
		self.set_delayed_goal()	
		
if __name__ == "__main__":
	rospy.init_node("pose_action_sub_arm")
	Ts   = 0.005
	wn   = 8.0
	zeta = 0.7
	vel_limit = 0.87

	s = PosActionServer(Ts,wn,zeta,vel_limit)
	
	while not rospy.is_shutdown():
		s.publish_position()
		s.rate.sleep()
		
