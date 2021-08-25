# LFD_Canal_Surface_Pkg
A ROS package that implements a learning from demonstrations algorithm using canal surfaces.

Maintained by Nick Pulsone, University of Massachusetts Lowell (Nicholas_Pulsone@student.uml.edu)
Based on "Encoding Demonstrations and Learning New Trajectories using Canal Surfaces"
by S. Reza Ahmadzadeh and Sonia Chernova (2016), Georgia Tech

The package currently operates as a demo for a simulated UR-5e robot. 

# Launching the Demo
1) Make sure you are running ROS Kinetic and have the TRAC-IK solver installed:
```
sudo apt-get update
```
```
sudo apt-get install ros-kinetic-trac-ik-kinematics-plugin
```

2) Navigate to your catkin workspace, and build the package:
```
roscd; cd ..
```
```
catkin_make
```

3) Source two different terminals by typing the following command in each:
```
source devel/setup.bash
```

4) Launch the simulated UR-5e robot in the first sourced terminal with the following:
```
roslaunch ur5_e_moveit_config demo.launch limited:=true
```

5) Ensure RVIZ has opened, and the motion planner is working. Then, in the second sourced terminal, launch the algorithm:
```
roslaunch lfd_canal_surface_pkg run_lfd.launch
```

The second terminal will then offer an interactive interface that allows you to record or load data, run the algorithm, and execute the results on the simulated robot. To quit the program at any time, press CTRL-D and then CTRL-C.

# Recording Demonstrations on the Simulated Robot

To record a demonstration on the simulated robot, use the motion planner in RVIZ to navigate your robot to the desired starting position. Then, use the planner to plan a movement to a desired end position. Follow the instruction on the terminal, and immedietly following the countdown, press "Execute" to execute the planned path and your demonstration will be recorded.

To see the path that was recorded in RVIZ, add a "PoseArray" and set the topic to "/p_arr."

<gif>

Alternaltively, you can input your own recorded demonstrations from an h5 file. Files must have the structure as follows:
https://raw.githubusercontent.com/brenhertel/Pearl-ur5e/master/brendan_ur5e/pictures/hdf5%20demo%20recorder%20flowchart.png

# Using the Algorithm

Recording demonstrations on the simulated robot yeilds fairly smooth data, thus the default parameters for smoothing are sufficient. When loading demonstrations from files, though, it is necessary to tune the smoothing parameters with the provided interface to ensure accurate results.
  
The algorithm will use the smoothed data to generate a canal surface, a gemetric figure that encapsulates the demonstrations with circular cross sections about the mean curve of all of the demonstrations.
  
Canal Surface for a reaching skill:
  
<img>
  
The canal surface will reproduce a trajectory for the simulated robot to execute. By default, the robot will reproduce a trajectory from a random starting point on the first cross section of the canal surface. Alternatively, it can reproduce from the robots current position in simulation, by changing the boolean "random" parameter to "False" in data_collection.py.
  
# Executing Results on the Simulated Robot

Depending on the smoothness your data or the complexity of the demonstrations, the robot may not be able to execute the reproduced trajectory, and it may require multiple tries or adjustments in the data to get the execution to work. 
  
To view the reproduced trajectory in RVIZ add a pose array and set the topic to "/p_arr."

<gif>


