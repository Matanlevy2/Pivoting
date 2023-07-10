#! /usr/bin/env python3
from turtle import pos
import rospy
from gazebo_msgs.srv import GetLinkProperties, SetLinkProperties
from geometry_msgs.msg import Pose, Point, Quaternion

rospy.init_node('test')
srv_get = rospy.ServiceProxy('/gazebo/get_link_properties',GetLinkProperties)
srv_set = rospy.ServiceProxy('/gazebo/set_link_properties',SetLinkProperties)
ans1 = srv_get('tool')

pose = Pose()
#position = Point()
position = ans1.com.position
#orientation = Quaternion()
orientation = ans1.com.orientation
mass = ans1.mass * 1.1

pose.position = position
pose.orientation = orientation
ixx = ans1.ixx
ixy = ans1.ixy
ixz = ans1.ixz
iyy = ans1.iyy
iyz = ans1.iyz
izz = ans1.izz
gravity_mode = True



ans2 = srv_set(link_name = 'tool',mass = mass, com = pose, gravity_mode = gravity_mode, ixx = ixx, ixy = ixy, ixz = ixz, iyy = iyy, iyz = iyz, izz = izz)
ans3 = srv_get('tool')


print(ans1)
print(ans3)
rospy.spin()