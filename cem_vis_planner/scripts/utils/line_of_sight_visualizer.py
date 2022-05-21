#!/usr/bin/env python

import rospy
# import math
import tf
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from numpy import *
import scipy.special
from scipy import interpolate
from scipy.io import loadmat
import random
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose
import tf
import message_filters
from nav_msgs.msg import Odometry


line_of_sight_marker = Marker() 
line_of_sight_marker.id = 25
line_of_sight_marker.header.frame_id = 'los'
line_of_sight_marker.type = Marker.LINE_STRIP
line_of_sight_marker.ns = ''
line_of_sight_marker.action = Marker.ADD
line_of_sight_marker.scale.x = 0.1
line_of_sight_marker.pose.orientation.w = 1.0
line_of_sight_marker.color.a= 1.0
line_of_sight_marker.color.r = 1
line_of_sight_marker.color.g = 0
line_of_sight_marker.color.b = 1

def odom_broadcaster(drone_odom):
  br = tf.TransformBroadcaster()
  br.sendTransform((drone_odom.pose.pose.position.x, drone_odom.pose.pose.position.y, drone_odom.pose.pose.position.z)
  ,(drone_odom.pose.pose.orientation.x, drone_odom.pose.pose.orientation.y,
   drone_odom.pose.pose.orientation.z, drone_odom.pose.pose.orientation.w),
                   rospy.Time.now(),"base_link","world")

def odomCallback(drone_odom, target_odom):
  line_sight_temp_point_1 = Point()
  line_sight_temp_point_2 = Point()
  obs_1_traj_point = Point()
  obs_2_traj_point = Point()
  obs_3_traj_point = Point()
  line_of_sight_marker.points=[]

  line_sight_temp_point_1.x = drone_odom.pose.pose.position.x
  line_sight_temp_point_1.y = drone_odom.pose.pose.position.y
  line_sight_temp_point_1.z = 1

  line_sight_temp_point_2.x = target_odom.pose.pose.position.x
  line_sight_temp_point_2.y = target_odom.pose.pose.position.y
  line_sight_temp_point_2.z = 1

  line_of_sight_marker.points.append(line_sight_temp_point_1)
  line_of_sight_marker.points.append(line_sight_temp_point_2)
  line_of_sight_marker.header.stamp = rospy.Time.now()
  line_of_sight_pub.publish(line_of_sight_marker)

if __name__ == '__main__':

  rospy.init_node('los_visualizer')
  traj_number = 8
  drone_odom_sub = message_filters.Subscriber('/bebop/odom', Odometry)
  target_odom_sub = message_filters.Subscriber('/target/odom', Odometry)
  
  ts = message_filters.ApproximateTimeSynchronizer([drone_odom_sub, target_odom_sub], 1,1)
  ts.registerCallback(odomCallback)
  line_of_sight_pub = rospy.Publisher( 'line_of_sight', Marker, queue_size=100)

  rospy.spin()

  




