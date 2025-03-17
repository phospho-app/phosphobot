#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import JointState
from piper_msgs.msg import PosCmd
from std_msgs.msg import Header

# float64 x
# float64 y
# float64 z
# float64 roll
# float64 pitch
# float64 yaw
# float64 gripper    # 单位：米    范围：0 ~ 0.08米
# int32 mode1
# int32 mode2

class PIPER:
    def __init__(self):
        
        # 发布控制piper机械臂话题
        self.pub_descartes = rospy.Publisher('pos_cmd', PosCmd, queue_size=10)
        self.pub_joint = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.left_pub_joint = rospy.Publisher('/left_joint_states', JointState, queue_size=100)
        self.right_pub_joint = rospy.Publisher('/right_joint_states', JointState, queue_size=100)
        self.descartes_msgs = PosCmd()
        
        # self.rate = rospy.Rate(80) # 10hz

    def init_pose(self):
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = rospy.Time.now()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_joint.publish(joint_states_msgs)
        # self.rate.sleep()
        print("send joint control piper command")
        
    def left_init_pose(self):
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = rospy.Time.now()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_pub_joint.publish(joint_states_msgs)
        # self.rate.sleep()
        print("send joint control piper command")
        
    def right_init_pose(self):
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = rospy.Time.now()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.right_pub_joint.publish(joint_states_msgs)
        # self.rate.sleep()
        print("send joint control piper command")
        
    def descartes_control_piper(self,x,y,z,roll,pitch,yaw,gripper):
        self.descartes_msgs.x = x
        self.descartes_msgs.y = y
        self.descartes_msgs.z = z
        self.descartes_msgs.roll = roll
        self.descartes_msgs.pitch = pitch
        self.descartes_msgs.yaw = yaw
        self.descartes_msgs.gripper = gripper
        self.pub_descartes.publish(self.descartes_msgs)
        # print("send descartes control piper command")
    
    def joint_control_piper(self,j1,j2,j3,j4,j5,j6,gripper):
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = rospy.Time.now()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position.append(j1)
        joint_states_msgs.position.append(j2)
        joint_states_msgs.position.append(j3)
        joint_states_msgs.position.append(j4)
        joint_states_msgs.position.append(j5)
        joint_states_msgs.position.append(j6)
        joint_states_msgs.position.append(gripper)
        self.pub_joint.publish(joint_states_msgs)
        # self.rate.sleep()
        print("send joint control piper command")
    
    def left_joint_control_piper(self,j1,j2,j3,j4,j5,j6,gripper):
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = rospy.Time.now()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position.append(j1)
        joint_states_msgs.position.append(j2)
        joint_states_msgs.position.append(j3)
        joint_states_msgs.position.append(j4)
        joint_states_msgs.position.append(j5)
        joint_states_msgs.position.append(j6)
        joint_states_msgs.position.append(gripper)
        self.left_pub_joint.publish(joint_states_msgs)
        # self.rate.sleep()
        print("send joint control piper command")
        
    
    def right_joint_control_piper(self,j1,j2,j3,j4,j5,j6,gripper):
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = rospy.Time.now()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position.append(j1)
        joint_states_msgs.position.append(j2)
        joint_states_msgs.position.append(j3)
        joint_states_msgs.position.append(j4)
        joint_states_msgs.position.append(j5)
        joint_states_msgs.position.append(j6)
        joint_states_msgs.position.append(gripper)
        self.right_pub_joint.publish(joint_states_msgs)
        # self.rate.sleep()
        print("send joint control piper command")
    
    
     
# test code
if __name__ == '__main__':
    # piper = PIPER() 
    rospy.init_node('control_piper_node', anonymous=True)
    # piper.control_piper(0.0,0.0,0.0,0.0,0.0,0.0,0.05)
    # 保持节点运行并监听外部程序的调用
    rospy.spin()
