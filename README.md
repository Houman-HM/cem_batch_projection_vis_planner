# cem_vis_planner
The repository is associated with our RAL + IROS 2022 submission Visibility-Aware Navigation with Batch Projection Augmented Cross-Entropy Method over a Learned Occlusion Cost.

# Dependecies:

* [JAX](https://github.com/google/jax)
* [bebop_simulator](https://github.com/Houman-HM/bebop_simulator/tree/bebop_hokuyo)
* [robotont_gazebo](https://github.com/robotont/robotont_gazebo)
* [gazebo2rviz](https://github.com/andreasBihlmaier/gazebo2rviz) (If you need the RViz visualization)

## Demo Video


## Installation procedure
After installing the dependencies, you can build our propsed MPC package as follows:
``` 
cd your_catkin_ws/src
git clone https://github.com/Houman-HM/vacna
cd .. && catkin build
source your_catkin_ws/devel/setup.bash
```
## Running the algorithm

In order to run the MPC for the warehouse inventory, follow the procedure below:

### In the first terminal:
```
roslaunch vacna warehouse.launch
```

This launches a warehouse environment in Gazebo.
### In the second terminal:

#### For the MPC with Gaussian Initialization:

```
rosrun vacna planner_gaussian.py
```
#### For the MPC with CVAE Initialization:
```
rosrun vacna planner_cvae.py
```

## Hyper parameters used in the optimization

**Algorithm 2:** 

| **Parameter** | Occlusion weight | CEM batch size| projection batch size | &rho; | Smoohtness weight|
| :----: | :----: | :----:  | :----:  | :----:  | :----:|
| **Value**| 10000 | 500 | 100 | 1 | 10 |
