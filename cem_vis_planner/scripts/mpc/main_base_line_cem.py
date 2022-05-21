#!/usr/bin/env python
import rospy
import message_filters
from nav_msgs.msg import Odometry
import rospkg
import sys
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import rospkg
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import threading

import queue
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

import bernstein_coeff_order10_arbitinterval
import mpc_module_base_line_cem as mpc_module
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import time
from jax import vmap, random
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud
import rospkg

robot_cmd_publisher = None

robot_pose_vel = []
target_pose_vel = []
obs_1_pose_vel  = []
obs_2_pose_vel  = []
obs_3_pose_vel  = []

is_received = False
robot_cmd_publisher = None
robot_traj_marker_publisher = None
robot_traj_publisher = None
robot_traj_publisher = None
pointcloud_publisher = None
obstacle_points = None

x_obs_pointcloud = np.ones((200,1)) * 100
y_obs_pointcloud = np.ones((200,1)) * 100

pointcloud_mutex = threading.Lock()
odom_mutex = threading.Lock()

def pointcloudCallback(msg):
    global x_obs_pointcloud, x_obs_pointcloud, obstacle_points, is_received, pointcloud_mutex
    msg_len = len(msg.points)
    pointcloud_mutex.acquire()
    for nn in range(200):
        if (nn< msg_len):
            x_obs_pointcloud[nn] = msg.points[nn].x
            y_obs_pointcloud[nn] = msg.points[nn].y
    obstacle_points = np.hstack((x_obs_pointcloud, y_obs_pointcloud))
    pointcloud_mutex.release()
    is_received = True
    

def odomCallback(robot_odom, target_odom, obs_1_odom, obs_2_odom, obs_3_odom):

    global is_received, robot_pose_vel, target_pose_vel, obs_1_pose_vel, obs_2_pose_vel, obs_3_pose_vel, odom_mutex
    odom_mutex.acquire()
    robot_orientation_q = robot_odom.pose.pose.orientation
    robot_orientation_list = [robot_orientation_q.x, robot_orientation_q.y, robot_orientation_q.z, robot_orientation_q.w]
    target_orientation_q = target_odom.pose.pose.orientation
    target_orientation_list = [target_orientation_q.x, target_orientation_q.y, target_orientation_q.z, target_orientation_q.w]

    obs_1_orientation_q = obs_1_odom.pose.pose.orientation
    obs_1_orientation_list = [obs_1_orientation_q.x, obs_1_orientation_q.y, obs_1_orientation_q.z, obs_1_orientation_q.w]

    obs_2_orientation_q = obs_2_odom.pose.pose.orientation
    obs_2_orientation_list = [obs_2_orientation_q.x, obs_2_orientation_q.y, obs_2_orientation_q.z, obs_2_orientation_q.w]

    obs_3_orientation_q = obs_3_odom.pose.pose.orientation
    obs_3_orientation_list = [obs_3_orientation_q.x, obs_3_orientation_q.y, obs_3_orientation_q.z, obs_3_orientation_q.w]

    target_orientation_q = target_odom.pose.pose.orientation
    target_orientation_list = [target_orientation_q.x, target_orientation_q.y, target_orientation_q.z, target_orientation_q.w]

    (robot_roll, robot_pitch, robot_yaw) = euler_from_quaternion (robot_orientation_list)
    (target_roll, target_pitch, target_yaw) = euler_from_quaternion (target_orientation_list)
    (obs_1_roll, obs_1_pitch, obs_1_yaw) = euler_from_quaternion (obs_1_orientation_list)
    (obs_2_roll, obs_2_pitch, obs_2_yaw) = euler_from_quaternion (obs_2_orientation_list)
    (obs_3_roll, obs_3_pitch, obs_3_yaw) = euler_from_quaternion (obs_3_orientation_list)

    robot_pose_vel = [robot_odom.pose.pose.position.x, robot_odom.pose.pose.position.y, robot_yaw, 
                    robot_odom.twist.twist.linear.x, robot_odom.twist.twist.linear.y, robot_odom.twist.twist.angular.z]
    target_pose_vel = [target_odom.pose.pose.position.x, target_odom.pose.pose.position.y, target_yaw, 
                    target_odom.twist.twist.linear.x, target_odom.twist.twist.linear.y, target_odom.twist.twist.angular.z]
    obs_1_pose_vel = [obs_1_odom.pose.pose.position.x, obs_1_odom.pose.pose.position.y, obs_1_yaw,
                    obs_1_odom.twist.twist.linear.x, obs_1_odom.twist.twist.linear.y, obs_1_odom.twist.twist.angular.z]

    obs_2_pose_vel = [obs_2_odom.pose.pose.position.x, obs_2_odom.pose.pose.position.y, obs_2_yaw,
                    obs_2_odom.twist.twist.linear.x, obs_2_odom.twist.twist.linear.y, obs_2_odom.twist.twist.angular.z]

    obs_3_pose_vel = [obs_3_odom.pose.pose.position.x, obs_3_odom.pose.pose.position.y, obs_3_yaw,
                    obs_3_odom.twist.twist.linear.x, obs_3_odom.twist.twist.linear.y, obs_3_odom.twist.twist.angular.z]
    odom_mutex.release()

def updatePointcloud():
    global obstacle_points, obs_1_pose_vel, obs_2_pose_vel, obs_3_pose_vel, pointcloud_publisher, is_received
    while(True):
        if (is_received):
            pointcloud_marker = Marker() 
            pointcloud_marker.id = 25
            pointcloud_marker.header.frame_id = 'base'
            pointcloud_marker.type = Marker.POINTS
            pointcloud_marker.ns = ''
            pointcloud_marker.action = Marker.ADD
            pointcloud_marker.scale.x = 0.05
            pointcloud_marker.scale.y = 0.05
            pointcloud_marker.scale.z = 10
            pointcloud_marker.pose.orientation.w = 1.0
            pointcloud_marker.color.a= 1.0
            pointcloud_marker.color.r = 1
            pointcloud_marker.color.g = 0
            pointcloud_marker.color.b = 0
            pointcloud_marker.points = []
            for kk in range (obstacle_points.shape[0]):
                temp_point = Point()
                temp_point.x = obstacle_points[kk,0]
                temp_point.y = obstacle_points[kk,1]
                temp_point.z = 1
                pointcloud_marker.points.append(temp_point)
            pointcloud_publisher.publish(pointcloud_marker)

def mpc():

    global is_received, robot_pose_vel, target_pose_vel, obs_1_pose_vel, obs_2_pose_vel, obs_3_pose_vel, robot_cmd_publisher, \
    robot_traj_publisher, robot_traj_marker_publisher, pointcloud_publisher, obstacle_points
    rospy.loginfo("MPC thread started sucessfully!")

    rospack = rospkg.RosPack()
    package_path = rospack.get_path("cem_vis_planner")

    v_max = 1.5 
    a_max = 1.5
    
    num_batch_projection = 500
    num_batch_cem = 500
    num_target = 1
    num_workspace = 1
    ellite_num_shift = int(num_batch_projection/3)

    rho_ineq = 1.0 
    rho_projection =  1.0
    rho_target = 1.0
    rho_workspace = 1.0
    maxiter_projection = 1
    maxiter_cem = 2
    maxiter_mpc = 3000

    a_target = 0.3
    b_target = 0.3
    d_min_target = 1
    d_max_target = 3
    d_avg_target = (d_min_target+d_max_target)/2.0

    ############# parameters

    t_fin = 5.0
    num = 70
    tot_time = np.linspace(0, t_fin, num)
    tot_time_copy = tot_time.reshape(num, 1)
            
    P, Pdot, Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
    nvar = np.shape(P)[1]

    tot_time_jax = jnp.asarray(tot_time)

    ###################################
    t_update = 0.05
    num_up = 200
    dt_up = t_fin/num_up
    tot_time_up = np.linspace(0, t_fin, num_up)
    tot_time_copy_up = tot_time_up.reshape(num_up, 1)

    P_up, Pdot_up, Pddot_up = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy_up[0], tot_time_copy_up[-1], tot_time_copy_up)

    P_up_jax = jnp.asarray(P_up)
    Pdot_up_jax = jnp.asarray(Pdot_up)
    Pddot_up_jax = jnp.asarray(Pddot_up)
    
    ########################################

    x_init =  -2.0
    vx_init = 0.0
    ax_init = 0.0

    y_init =  -2.0
    vy_init = 0.0
    ay_init = 0.0


    x_target_init = -0.5
    y_target_init = -2.0

    vx_target = 0.0
    vy_target = 0.4

    x_target = x_target_init+vx_target*tot_time
    y_target = y_target_init+vy_target*tot_time

    x_target = x_target.reshape(1, num)
    y_target = y_target.reshape(1, num)


    x_target_fin = x_target[0, -1]
    y_target_fin = y_target[0, -1]



    x_workspace = 0*jnp.ones((num_workspace, num))
    y_workspace = 0*jnp.ones((num_workspace, num ))

    a_workspace = 1300.0
    b_workspace = 1300.0

    weight_biases_mat_file = loadmat(package_path + "/nn_weight_biases/nn_weight_biases.mat")

    ###############################################################3
    A = np.diff(np.diff(np.identity(num), axis = 0), axis = 0)

    temp_1 = np.zeros(num)
    temp_2 = np.zeros(num)
    temp_3 = np.zeros(num)
    temp_4 = np.zeros(num)

    temp_1[0] = 1.0
    temp_2[0] = -2
    temp_2[1] = 1
    temp_3[-1] = -2
    temp_3[-2] = 1

    temp_4[-1] = 1.0

    A_mat = -np.vstack(( temp_1, temp_2, A, temp_3, temp_4   ))
    
    R = np.dot(A_mat.T, A_mat)
    mu = np.zeros(num)
    cov = np.linalg.pinv(R)

    ################# Gaussian Trajectory Sampling
    eps_k = np.random.multivariate_normal(mu, 0.001*cov, (num_batch_projection, ))
    
    goal_rot = -np.arctan2(y_target_fin-y_target_init, x_target_fin-x_target_init)
    
    x_init_temp = x_target_init*np.cos(goal_rot)-y_target_init*np.sin(goal_rot)
    y_init_temp = x_target_init*np.sin(goal_rot)+y_target_init*np.cos(goal_rot)


    x_fin_temp = x_target_fin*np.cos(goal_rot)-y_target_fin*np.sin(goal_rot)
    y_fin_temp = x_target_fin*np.sin(goal_rot)+y_target_fin*np.cos(goal_rot)


    x_interp = jnp.linspace(x_init_temp, x_fin_temp, num)
    y_interp = jnp.linspace(y_init_temp, y_fin_temp, num)

    x_guess_temp = jnp.asarray(x_interp+0.0*eps_k) 
    y_guess_temp = jnp.asarray(y_interp+eps_k)

    x_samples_init = x_guess_temp*jnp.cos(goal_rot)+y_guess_temp*jnp.sin(goal_rot)
    y_samples_init = -x_guess_temp*jnp.sin(goal_rot)+y_guess_temp*jnp.cos(goal_rot)

    x_samples_shift = x_samples_init[0:ellite_num_shift, :]
    y_samples_shift = y_samples_init[0:ellite_num_shift, :]
    
    ##############################################################

    occlusion_weight = 10000

    prob = mpc_module.batch_occ_tracking(P, Pdot, Pddot, v_max, a_max, t_fin, num, num_batch_projection, 
                                                        num_batch_cem, tot_time, rho_ineq, maxiter_projection, rho_projection, rho_target, num_target, 
                                                        a_workspace, b_workspace, num_workspace, rho_workspace, maxiter_cem, d_min_target, d_max_target, 
                                                        P_up_jax, Pdot_up_jax, Pddot_up_jax, occlusion_weight)
    

    prob.W0, prob.b0, prob.W1, \
    prob.b1, prob.W2, prob.b2, \
    prob.W3, prob.b3 = mpc_module.get_weights_biases(weight_biases_mat_file)

    lamda_x = jnp.zeros((num_batch_projection, nvar))
    lamda_y = jnp.zeros((num_batch_projection, nvar))

    d_a = a_max*jnp.ones((num_batch_projection, num))
    alpha_a = jnp.zeros((num_batch_projection, num))		

    d_v = v_max*jnp.ones((num_batch_projection, num))
    alpha_v = jnp.zeros((num_batch_projection, num))		

    alpha_workspace = jnp.zeros(( num_batch_projection, num_workspace*num))
    d_workspace = jnp.ones(( num_batch_projection, num_workspace*num))

    key = random.PRNGKey(0)
    cost_track = np.zeros(maxiter_mpc)
    x_drone = np.ones(maxiter_mpc)
    y_drone = np.ones(maxiter_mpc)
    vx_drone = np.ones(maxiter_mpc)
    vy_drone = np.ones(maxiter_mpc)
    x_target_vec = np.ones(maxiter_mpc)
    y_target_vec = np.ones(maxiter_mpc)
    ax_drone = np.ones(maxiter_mpc)
    ay_drone = np.ones(maxiter_mpc)

    robot_traj_marker = Marker() 
    robot_traj_marker.id = 25
    robot_traj_marker.header.frame_id = 'base'
    robot_traj_marker.type = Marker.LINE_STRIP
    robot_traj_marker.ns = ''
    robot_traj_marker.action = Marker.ADD
    robot_traj_marker.scale.x = 0.1
    robot_traj_marker.pose.orientation.w = 1.0
    robot_traj_marker.color.a= 1.0
    robot_traj_marker.color.r = 1
    robot_traj_marker.color.g = 0
    robot_traj_marker.color.b = 1

    alpha_init = np.arctan2(y_target_init - y_init, x_target_init - x_init)

    rospy.loginfo("Waiting for initial JAX compilation!")
    for i in range(0, maxiter_mpc):
        start_time = time.time()
        pointcloud_mutex.acquire()
        odom_mutex.acquire()
        jax_obstacle_points = jnp.asarray(obstacle_points)
        vx_target = target_pose_vel[3] * np.cos(target_pose_vel[2])
        vy_target = target_pose_vel[3] * np.sin(target_pose_vel[2])
        alpha_init = robot_pose_vel[2]
        pointcloud_mutex.release()
        odom_mutex.release()

        x_samples_init, y_samples_init = prob.compute_initial_samples(jnp.asarray(eps_k), x_target_init, y_target_init, x_target_fin, y_target_fin, x_samples_shift, y_samples_shift, ellite_num_shift, x_init, y_init)

        c_x_samples_init, c_y_samples_init, x_samples_init, y_samples_init = prob.compute_inital_guess( x_samples_init, y_samples_init)


        c_x_best, c_y_best, cost_track[i], x_best, y_best, alpha_v, d_v, alpha_a, d_a, \
             alpha_target, d_target, lamda_x, lamda_y, alpha_workspace, d_workspace, key, x_samples_shift, y_samples_shift = prob.compute_cem(key, x_init, vx_init, ax_init, y_init, vy_init, ay_init, alpha_a, d_a, alpha_v, d_v,
					                                x_target, y_target, lamda_x, lamda_y, x_samples_init, y_samples_init, x_workspace, y_workspace,
					                                alpha_workspace, d_workspace, c_x_samples_init, c_y_samples_init, vx_target, vy_target,
					                                d_avg_target, ellite_num_shift, jax_obstacle_points)

        vx_control_local, vy_control_local, ax_control, \
        ay_control, vangular_control, robot_traj_x, robot_traj_y, vx_control, vy_control= prob.compute_controls(c_x_best, c_y_best, dt_up, vx_target, vy_target, 
							                                                    t_update, tot_time_copy_up, x_init, y_init, alpha_init,
                                                                                 x_target_init, y_target_init)


        if (i!=0):
            cmd = Twist()
            cmd.linear.x= vx_control_local
            cmd.linear.y= vy_control_local
            cmd.angular.z = vangular_control
            robot_cmd_publisher.publish(cmd)
        time_taken = time.time() - start_time
        rospy.loginfo ("Time taken: %s", str(time_taken))
        
        odom_mutex.acquire()
        
        x_init = robot_pose_vel[0]
        y_init = robot_pose_vel[1]

        vx_init = vx_control
        vy_init = vy_control

        ax_init = ax_control
        ay_init = ay_control

        x_target_init = target_pose_vel[0]
        y_target_init = target_pose_vel[1]

        x_target = x_target_init + vx_target * tot_time_jax
        y_target = y_target_init + vy_target * tot_time_jax

        x_target_fin = x_target[-1]
        y_target_fin = y_target[-1]

        x_target = x_target.reshape(1, num)
        y_target = y_target.reshape(1, num)
        
        x_drone[i] = x_init
        y_drone[i] = y_init
        
        vx_drone[i] = vx_init
        vy_drone[i] = vy_init
        ax_drone[i] = ax_control
        ay_drone[i] = ay_control  
        
        x_target_vec[i] = x_target_init
        y_target_vec[i] = y_target_init

        odom_mutex.release()


if __name__ == "__main__":

	
    rospy.init_node('nn_mpc_node')
    rospack = rospkg.RosPack()

    robot_cmd_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    robot_traj_marker_publisher = rospy.Publisher('/robot_traj', Marker, queue_size=10)
    pointcloud_publisher = rospy.Publisher('/generated_pointcloud', Marker, queue_size=10)
    robot_traj_publisher = rospy.Publisher('command/pose', PoseStamped, queue_size=10)
    rospy.Subscriber("/pointcloud", PointCloud, pointcloudCallback)

    robot_odom_sub = message_filters.Subscriber('bebop/odom', Odometry)
    target_odom_sub = message_filters.Subscriber('/target/odom', Odometry)
    obs_1_odom_sub = message_filters.Subscriber('/obs_1/odom', Odometry)
    obs_2_odom_sub = message_filters.Subscriber('/obs_2/odom', Odometry)
    obs_3_odom_sub = message_filters.Subscriber('/obs_3/odom', Odometry)
    ts = message_filters.ApproximateTimeSynchronizer([robot_odom_sub, target_odom_sub, obs_1_odom_sub, obs_2_odom_sub, obs_3_odom_sub], 1,1, allow_headerless=True)
    ts.registerCallback(odomCallback)
    mpc_thread = threading.Thread(target=mpc)
    pointcloud_update_thread = threading.Thread(target=updatePointcloud)
    mpc_thread.start()
    rospy.spin()


