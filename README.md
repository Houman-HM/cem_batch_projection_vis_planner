# cem_vis_planner
The repository is associated with our RAL + IROS 2022 submission Visibility-Aware Navigation with Batch Projection Augmented Cross-Entropy Method over a Learned Occlusion Cost.

# Dependecies:

* [JAX](https://github.com/google/jax)
* [bebop_simulator](https://github.com/Houman-HM/bebop_simulator/tree/bebop_hokuyo)
* [odom_visualizer](https://github.com/HKUST-Aerial-Robotics/plan_utils/tree/master/odom_visualization) (If you need the Rviz visualization)

## Installation procedure
After installing the dependencies, you can build our propsed MPC package as follows:
``` 
cd your_catkin_ws/src
git clone https://github.com/Houman-HM/cem_batch_projection_vis_planner
cd .. && catkin build
source your_catkin_ws/devel/setup.bash
```
## Running the algorithm

In order to run the MPC for tracking a target in a world with 6 obstacles, follow the procedure below:

### In the first terminal:
```
roslaunch cem_vis_planner 6_wall_world.launch
```

This launches a Gazebo environment with 6 walls spawned.
### In the second terminal:

#### For Algorithm 2:

```
rosrun cim_vis_planner main_projection.py
```
#### For Algorithm 1:
```
rosrun cim_vis_planner main_base_line.py
```

### In the third terminal:

There are two bag files available for moving the target with speeds of 0.5 and 1 m/s in the 6_wall environment. You can play either of them to test the algorithms.

```
roscd cem_vis_planner && cd target_trajectory_bag_files

rosbag play 6_wall_target_vel_05.bag // or

rosbag play 6_wall_target_vel_1.bag
```
You can also start teleoperating the target by publishing velocities on ``` /target/cmd_vel ``` topic. The drone should start following it as you are teleoperating the target.

## Running Nageli implmentation using ACADO
#### Running generated ACADO code
* Edit global variables in ```test.c```: maker position, obstacles initial position and velocity, quadrotor's initial position, weights
* Run ```make clean all``` only for the first time otherwise just ```make``` followed by ```./test```.
* To visualize, run ```python quad_plot.py```

_Look into code_gen.cpp file for acado settings, cost terms, constraints, etc._