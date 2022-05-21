#!/usr/bin/env python3

from re import I
import rospy 
from laser_assembler.srv import *
from sensor_msgs.msg import PointCloud
import time
import numpy as np

i = 0
def callback(msg):
    x_obs = np.ones((200,1))
    y_obs = np.ones((200,1))
    start_time = time.time()
    msg_len = len(msg.points)
    for i in range(200):
        if (i< msg_len):
            x_obs[i] = msg.points[i].x
            y_obs[i] = msg.points[i].y
            print (i)
    print ("time:", time.time() - start_time)
rospy.init_node("get_points")
rospy.Subscriber("/pointcloud", PointCloud, callback)
rospy.spin()
