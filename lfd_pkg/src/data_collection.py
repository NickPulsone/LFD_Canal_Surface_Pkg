#!/usr/bin/env python

# Design pattern derived from program by Brendan Hertel, UML: https://github.com/brenhertel/Pearl-ur5e/blob/master/brendan_ur5e/src/scripts/demo_xyz_playback.py 

"""
   Will offer the user an interface to record demonstrations and process 
   them so that they can to be sent to an algorithm to construct a canal surface
   and perform trajectory reproduction
"""

import sys
import copy
import h5py
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi, ceil
from std_msgs.msg import String, Float64MultiArray, MultiArrayDimension
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from canal_surface_algorithm import *
from lfd_canal_surface_pkg.srv import CanalSrv, CanalSrvRequest

"""DEFAULT PARAMTERS"""

# Number of points that smooth demonstrations will consist of
num_points_to_sample = 300

# Degree of spline interpolation
spline_degree = 3

# Smoothing factor of demonstrations (closer to 0 means less smooth, closer to 1 means more smooth)
smoothing_factor = 0.00001

# Number of points to cut off the start and end of each demonstration
start_cut = 0
end_cut = 2

""" Helper functions """

def get_y_n(prompt):
    """Gets a yes or no response from the user"""
    answer = raw_input(prompt)
    while len(answer) == 0 or answer[0] not in ["Y", "y", "N", "n"]:
        answer = raw_input("\nPlease enter a valid (Y/N) response: ")
    if answer[0] == "Y" or answer[0] == "y":
        return True
    else:
        return False


def pose_to_xyz(poses):
    """Convert array of pose objects (position and orientation) to array of cartesian positions"""
    x = []
    y = []
    z = []
    for pose in poses:
        x.append(pose.position.x)
        y.append(pose.position.y)
        z.append(pose.position.z)
    return np.array([x, y, z])


def xyz_to_pose(cartesian_points, orientation):
    """Convert array of cartesian positions to array of pose objects (position and orientation)"""
    poses = []
    for i in range(len(cartesian_points[0])):
        pose = geometry_msgs.msg.Pose()
        pose.position.x = cartesian_points[0][i]
        pose.position.y = cartesian_points[1][i]
        pose.position.z = cartesian_points[2][i]
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]
        poses.append(copy.deepcopy(pose))
    return poses


def make_plot_even(ax, xl, xr, yl, yr, zb, zt):
    """ Given axes boundries, make a 3d plot even """
    axis_limits = [abs(xr - xl), abs(yr - yl), abs(zt - zb)]
    if axis_limits.index(max(axis_limits)) == 0:
        ax.set_ylim(ax.get_ylim()[0], yr + (axis_limits[0] - axis_limits[1]))
        ax.set_zlim(ax.get_zlim()[0], zt + (axis_limits[0] - axis_limits[2]))
    elif axis_limits.index(max(axis_limits)) == 1:
        ax.set_xlim(ax.get_xlim()[0], xr + (axis_limits[1] - axis_limits[0]))
        ax.set_zlim(ax.get_zlim()[0], zt + (axis_limits[1] - axis_limits[2]))
    else:
        ax.set_xlim(ax.get_xlim()[0], xr + (axis_limits[2] - axis_limits[0]))
        ax.set_ylim(yl - (axis_limits[2] - axis_limits[1]), ax.get_ylim()[1])
    return

""" How the robot is understood and controlled """

class MoveGroupPythonInterface(object):
    def __init__(self, do_reaching):
        super(MoveGroupPythonInterface, self).__init__()
        # The moveit_commander is what is responsible for sending info the moveit controllers
        moveit_commander.roscpp_initialize(sys.argv)
        # Initialize node
        rospy.init_node('data_collect', anonymous=True)
        # Instantiate a `RobotCommander`_ object. Provides information such as the robot's kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()
        # Instantiate a `PlanningSceneInterface`_ object. This provides a remote interface for getting, setting, and updating the robot's internal understanding of the surrounding world:
        scene = moveit_commander.PlanningSceneInterface()
        # Instantiate a `MoveGroupCommander`_ object.  This object is an interface to a planning group (group of joints), which in our moveit setup is named 'manipulator'
        group_name = "manipulator" 
        move_group = moveit_commander.MoveGroupCommander(group_name)
        # Create a `DisplayTrajectory`_ ROS publisher which is used to display trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        # Create a `PoseArrayy`_ ROS publisher which is used to display demonstrations and planned paths in Rviz:
        pose_array_publiser = rospy.Publisher("/p_arr", geometry_msgs.msg.PoseArray, queue_size=1)

        """Get all the info which is carried with the interface object"""
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        #print("Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        #print  End effector link: %s" % eef_link

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        #print("Available Planning Groups:", robot.get_group_names())

        # Misc variables
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.pose_array_publiser = pose_array_publiser
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        # Raw pose objects (position and orientation) from a recorded demonstration
        self.raw_pose_demos = []
        # List of Raw cartesian position of each demonstration
        # Each demonstration of the form: [[x0, x1, ... , xn], [y0, y1, ... , yn], [z0, z1, ... , zn]]
        self.raw_demonstrations = []
        # Smoothed demonstrations with the same structure as raw demonstration with a seperate, defined n
        self.smooth_demonstrations = []
        # The mean/spine curve of the demonstration data, same exact structure as a given smooth demonstration
        self.directrix = []
        # The TNB frames of the directrix
        self.T = []
        self.N = []
        self.B = []
        # The radii of the canal surface
        self.radii = []
        # Keep track of the origin given a set of demonstrations (starts @ (0,0,0))
        self.origin = np.array([0.0, 0.0, 0.0])
        # Keep track of which demonstration is the best, particularly for reaching tasks
        self.reaching = do_reaching
        self.best_demo_index = 0
        # Keep track of an initial position for performing trajectory reproduction
        self.p0_for_reproduction = []
        # Trajectory reproduced from the canal surface and a given initial position
        self.reproduced_curve = []
        # Colors used for plotting demonstrations
        self.plot_line_colors = np.array(["red", "green", "blue", "yellow", "pink", "black",
                                          "orange", "purple", "beige", "brown", "gray", "cyan", 
                                          "magenta"])
        # Ranges for plotting the curves and identifying them. Of the form [[xleft, xright], [yleft, yright], [zleft, zright]]
        self.plot_curve_ranges = []
        

    def display_trajectory(self, plan):
        # Ask RVIZ to display the trajectory
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory)


    def display_pose_array(self, curve, orientation):
        # Ask RVIZ to display the pose array
        parray = geometry_msgs.msg.PoseArray() 
        parray.header.frame_id = "world"
        pose_trajectory = xyz_to_pose(curve, orientation)
        parray.poses = pose_trajectory
        # Publish
        self.pose_array_publiser.publish(parray)


    def load_demonstration_from_file(self):
        """Loads a demonstration from an h5 file based on user input
           Returns T/F based on whether the demonstration is one to keep"""
        # Open file that contains pose data
        filename = raw_input('Enter the filename (including file path) of the .h5 demo: ')
        try:
            hf = h5py.File(filename, 'r')
        except:
            print("Incorrect file name, failed to open file.\n")
            return self.load_demonstration_from_file()
        demo = hf.get('demo1')
        tf_info = demo.get('tf_info')
        pos_rot_data = np.array(tf_info.get('pos_rot_data'))
        hf.close()

        # Extract cartestian data from pose data (Rviz x and y is negative x and y in TF)
        self.raw_demonstrations.append(np.array([-1.0*pos_rot_data[0], -1.0*pos_rot_data[1], pos_rot_data[2]]))

        # Extra raw pose data for publishing pose arrays
        poses = []
        for i in range(len(pos_rot_data[0])):
            ps = geometry_msgs.msg.Pose()
            ps.position.x = -pos_rot_data[0][i]
            ps.position.y = -pos_rot_data[1][i]
            ps.position.z = pos_rot_data[2][i]
            ps.orientation.x = -pos_rot_data[4][i]
            ps.orientation.y = pos_rot_data[3][i]
            ps.orientation.z = pos_rot_data[6][i]
            ps.orientation.w = -pos_rot_data[5][i]
            poses.append(ps)
        self.raw_pose_demos.append(poses)


        # Show user the demonstration
        self.plot_curve_single(-1, raw=True)

        # Save demonstration if the user is satisfied, close plot
        save_demo = get_y_n("Would you like to save this demonstration (Y/N)?: ")
        plt.close()

        # Determine if the demonstration needs to be rerecorded. If not, add to data.
        if not save_demo:
            self.raw_demonstrations.pop()
            self.raw_pose_demos.pop()            
        return save_demo


    def record_a_demonstration(self):
        """Records a demonstration from the simulated robot based on movegroup data
           Returns T/F based on whether the demonstration is one to keep"""
        # Set rate of collecting poses
        rt = rospy.Rate(200)
        
        # Give user the option to have the robot move back to where it is now for speed
        return_to_start = get_y_n("Would you like to have the robot return to the current position after the recording (if so move it there now, Y/N)?: ")
        if return_to_start:
            start_joint_angles = self.move_group.get_current_joint_values()
        rospy.sleep(1.5)

        # Give the user the option to abandon the demonstration
        print("\nPress enter to begin recording (or 0 to quit)")
        begin_rec = raw_input() 
        if len(begin_rec) > 0 and begin_rec[0] == '0':
            return False

        # Give user a countdown
        print("\nGet ready to start moving the robot...\n")
        for num in [5, 4, 3, 2, 1]:
            print(str(num) + "...\n")
            rospy.sleep(1)

        # Define starting pose
        start_pose = self.move_group.get_current_pose(self.eef_link).pose

        # Record demonstration for the next 5 seconds
        wp = [start_pose]
        print("Start moving the robot!!!!!\n")
        now = rospy.get_time()
        while rospy.get_time() <= (now + 5.0):
            wp.append(copy.deepcopy(self.move_group.get_current_pose(self.eef_link).pose))
            # Requires root user priveleges to use (do not have on current linux installation)
            # if keyboard.is_pressed("enter"):
            #     break
            rt.sleep()

        # Test the length
        print("Done! Recorded " + str(len(wp)) + "poses!\n")
        rospy.sleep(1)

        # Update class variables
        self.raw_pose_demos.append(wp)
        cartesian_points = pose_to_xyz(wp)
        self.raw_demonstrations.append(cartesian_points)

        # Show user the demonstration
        self.plot_curve_single(-1, raw=True)

        # Save demonstration if the user is satisfied, close plot
        save_demo = get_y_n("Would you like to save this demonstration (Y/N)?: ")
        plt.close()

        # Determine if the demonstration needs to be rerecorded. If not, add to data.
        if not save_demo:
            self.raw_pose_demos.pop()
            self.raw_demonstrations.pop()

        # Move robot back to start if user had wished to do so
        if return_to_start:
            self.move_group.go(start_joint_angles, wait=True)
            self.move_group.stop()
    
        return save_demo


    def remove_demonstration(self, index):
        """ Removes a processed demonstration from the list 
            and recalculates the new directrix and coordinate reference """
        # Remove demonstration
        self.smooth_demonstrations = np.delete(self.smooth_demonstrations, index, axis=0)
        removed_demo = self.raw_demonstrations.pop(index)
        # Prompt for new best demonstration if the best is deleted
        if index == self.best_demo_index:
            self.raw_demonstrations = inverse_reframe_curves(self.raw_demonstrations, self.origin)
            self.process_raw_demonstrations()
            self.define_new_origin()
        

    def delete_a_demonstration(self):
        """Allows user to delete a prerecorded demonstration"""
        # Dont let user delete if there are only 2 demonstrations
        if len(self.smooth_demonstrations) <= 2:
            print("You must have at least two demonstrations for the canal surface.\n")
            return False
        # Show user curves to delete
        self.plot_curves()
        correct_demo = False
        while not correct_demo:
            # Prompt for which demonstrations should be deleted
            dem_to_delete = int(raw_input("What is the # of the demonstration that would you like to delete? (press '0' to quit): "))
            while dem_to_delete < 0 or dem_to_delete > len(self.smooth_demonstrations):
                dem_to_delete = int(raw_input("Please enter a valid positive integer that corresponds to a demonstration: "))
            # Break if user has a change of heart
            if dem_to_delete == 0:
                return False
            # Plot single curve to be deleted
            self.plot_curve_single(dem_to_delete-1)
            # Prompt to ensure the right demonstration is being deleted
            correct_demo = get_y_n("Are you sure you would like to delete this demonstration? (Y/N): ")
            if correct_demo:
                self.remove_demonstration(dem_to_delete - 1)
                return True
            else:
                # Re-show the plot of all demonstrations if user singled out the wrong demonstration to delete
                self.plot_curves()


    def process_raw_demonstrations(self):
        """ Provide user with an interface to smooth demonstrations """
        # Get global values for smoothing parameters
        global num_points_to_sample, spline_degree, smoothing_factor, start_cut, end_cut
        get_smoothing_parameters = get_y_n("\nWould you like to enter/update the smoothing parameters (otherwise deafults or last used parameters will be applied, Y/N)?: ")
        # Allow user to update parameters
        if get_smoothing_parameters:
            smoothing_is_good = False
            while not smoothing_is_good:
                num_points_to_sample = int(raw_input("\nEnter a number of points you would like the sampled curve to be: "))
                spline_degree = int(raw_input("\nEnter a degree for spline interpolation (integer 1 to 5): "))
                smoothing_factor = float(raw_input("\nEnter a smoothing factor (0.0 is no smoothing, the greater the number, the smoother the curve): "))
                start_cut = int(raw_input("\nEnter a number of indices to cut from the start of the raw demonstration: "))
                end_cut = int(raw_input("\nEnter a number of indices to remove from the end of the raw demonstrations: "))
                # Smooth based on updated parameters
                self.smooth_demonstrations = slice_and_sample_demos(self.raw_demonstrations, num_points_to_sample=num_points_to_sample, 
                                                                    spline_degree=spline_degree, smoothing_factor=smoothing_factor, 
                                                                    start_cut=start_cut, end_cut=end_cut+1)
                # Show user smoothed demonstrations, and confirm that the data is good to go
                self.plot_curves(raw=False)
                smoothing_is_good = get_y_n("Are you satisfied with the smoothing (Y/N)?: ")
        else:
            # Use global (current) values for smoothing/slicing
            self.smooth_demonstrations = slice_and_sample_demos(self.raw_demonstrations, num_points_to_sample=num_points_to_sample, 
                                                                spline_degree=spline_degree, smoothing_factor=smoothing_factor, 
                                                                start_cut=start_cut, end_cut=end_cut)


    def define_new_origin(self):
        """ Will prompt the user to say which demonstration is the best, the reframe the curves accordingly"""
        # Plot raw demonstrations
        self.plot_curves(raw=True)

        # Ask which demonstration is the best
        best_dem = int(raw_input("\nWhat is the # of the demonstration that you think is the best?: "))
        while best_dem < 1 or best_dem > len(self.raw_demonstrations):
            best_dem = int(raw_input("Please enter a valid positive integer that corresponds to a demonstration: "))

        # Assign chosen index
        self.best_demo_index = best_dem - 1

        # Assign origin based on chosen index
        self.origin = np.array([-self.raw_demonstrations[self.best_demo_index][0, -1], 
                                -self.raw_demonstrations[self.best_demo_index][1, -1], 
                                -self.raw_demonstrations[self.best_demo_index][2, -1]])
        
        # Reframe demonstrations accordingly
        self.raw_demonstrations, self.smooth_demonstrations = reframe_curves(self.raw_demonstrations, self.smooth_demonstrations, self.best_demo_index, self.reaching)


    def plot_curves(self, raw=False):
        """ Plot all of the current processed demonstrations """
        # Close any opened windows
        if plt.get_fignums():
            plt.close()
        # Plot curves according to self.plot_line_colors and the order they were recorded 
        patches = []
        ax = plt.axes(projection='3d')
        plt.ion()
        if raw:
            for i in range(len(self.raw_demonstrations)):
                line_color = self.plot_line_colors[i % len(self.plot_line_colors)]
                ax.plot3D(self.raw_demonstrations[i][0], self.raw_demonstrations[i][1], self.raw_demonstrations[i][2], line_color)
                patch = mpatches.Patch(color=line_color, label="Recorded Demonstration #" + str(i+1))
                patches.append(patch)
            # Plot a star on the graph to represent the robot base frame
            ax.plot3D([0.0], [0.0], [0.0], c="blue", marker = "*", markersize = 20)
        else:
            for i in range(len(self.smooth_demonstrations)):
                line_color = self.plot_line_colors[i % len(self.plot_line_colors)]
                ax.plot3D(self.smooth_demonstrations[i][0], self.smooth_demonstrations[i][1], self.smooth_demonstrations[i][2], line_color)
                patch = mpatches.Patch(color=line_color, label="Recorded Demonstration #" + str(i+1))
                patches.append(patch)
            # Plot a star on the graph to represent the robot base frame
            ax.plot3D([self.origin[0]], [self.origin[1]], [self.origin[2]], c="blue", marker = "*", markersize = 20)
        patches.append(mlines.Line2D([], [], color='blue', marker='*', linestyle='None', label='Robot Base'))
        plt.legend(handles=patches)
        # Set plot limits for 3D plot and make equal to show accurate representation of curve
        xl, xr = plt.xlim()
        yl, yr = plt.ylim()
        zb, zt = ax.get_zlim()
        make_plot_even(ax, xl, xr, yl, yr, zb, zt)
        # Update the plot curve ranges
        self.plot_curve_ranges = [plt.xlim(), plt.ylim(), ax.get_zlim()]
        plt.show()


    def plot_curve_single(self, index, raw=False):
        """ Plots a single curve, either smoothed or raw """
        # Close any opened plots
        if plt.get_fignums():
            plt.close()

        # Plot a single curve to show user before deleting
        ax = plt.axes(projection='3d')
        plt.ion()
        line_color = self.plot_line_colors[index % len(self.plot_line_colors)]
        if raw:
            ax.plot3D(self.raw_demonstrations[index][0], self.raw_demonstrations[index][1], self.raw_demonstrations[index][2], line_color)
            # Plot a star on the graph to represent the robot base frame
            ax.plot3D([0.0], [0.0], [0.0], c = "blue", marker = "*", markersize = 20)
            # Display pose array in Rviz
            self.display_pose_array(self.raw_demonstrations[index], [self.raw_pose_demos[index][0].orientation.x,
                                                                     self.raw_pose_demos[index][0].orientation.y,
                                                                     self.raw_pose_demos[index][0].orientation.z,
                                                                     self.raw_pose_demos[index][0].orientation.w])
        else:
            ax.plot3D(self.smooth_demonstrations[index][0], self.smooth_demonstrations[index][1], self.smooth_demonstrations[index][2], line_color)
            # Plot a star on the graph to represent the robot base frame
            ax.plot3D([self.origin[0]], [self.origin[1]], [self.origin[2]], c = "blue", marker = "*", markersize = 20)
            # Sey x, y, and z limits according to previously plotted demonstrations
            plt.xlim(self.plot_curve_ranges[0][0], self.plot_curve_ranges[0][1])
            plt.ylim(self.plot_curve_ranges[1][0], self.plot_curve_ranges[1][1])
            ax.set_zlim(self.plot_curve_ranges[2][0], self.plot_curve_ranges[2][1])

        # Set plot limits for 3D plot and make equal to show accurate representation of curve
        xl, xr = plt.xlim()
        yl, yr = plt.ylim()
        zb, zt = ax.get_zlim()
        make_plot_even(ax, xl, xr, yl, yr, zb, zt)

        # Create legend
        patches = []
        # If referencing the last demonstration, don't label according to index
        if index == -1:
            patches.append(mpatches.Patch(color=line_color, label="Recorded Demonstration"))
        else:
            patches.append(mpatches.Patch(color=line_color, label="Recorded Demonstration #" + str(index + 1)))
        patches.append(mlines.Line2D([], [], color='blue', marker='*', linestyle='None', label='Robot Base'))
        plt.legend(handles=patches)
            
        plt.show()


    def plot_canal_surface(self, reproduction=False):
        """ Plots a canal surface given current parameters, to be run after canal_surface_request """
        # Close any opened windows
        if plt.get_fignums():
            plt.close()
            
        # Plot curves according to self.plot_line_colors and the order they were recorded 
        patches = []
        ax = plt.axes(projection='3d')
        plt.ion()

        # Plot demonstrations
        line_color = "blue"
        for i in range(len(self.smooth_demonstrations)):
            ax.plot3D(self.smooth_demonstrations[i][0], self.smooth_demonstrations[i][1], self.smooth_demonstrations[i][2], line_color)
        patches.append(mpatches.Patch(color=line_color, label="Demonstrations"))

        # Plot directrix
        line_color = "red"
        ax.plot3D(self.directrix[0], self.directrix[1], self.directrix[2], line_color)
        patches.append(mpatches.Patch(color=line_color, label="Directrix"))

        # Plot circles
        line_color = "gray"
        num_circ = 20
        theta_density = 20
        theta = np.linspace(0, 2 * pi, theta_density)
        for i in range(0, len(self.radii), int(ceil(len(self.radii) / num_circ))):
            # Init array containers for x, y, and z of circles
            circ_x = np.zeros(theta_density, dtype=float)
            circ_y = np.zeros(theta_density, dtype=float)
            circ_z = np.zeros(theta_density, dtype=float)
            # Use calculated radii and the Normal and Binormal vectors to plot the circles
            for j in range(theta_density):
                circ_x[j] = self.directrix[0][i] + self.radii[i] * (self.N[0][i] * np.cos(theta[j]) + self.B[0][i] * np.sin(theta[j]))
                circ_y[j] = self.directrix[1][i] + self.radii[i] * (self.N[1][i] * np.cos(theta[j]) + self.B[1][i] * np.sin(theta[j]))
                circ_z[j] = self.directrix[2][i] + self.radii[i] * (self.N[2][i] * np.cos(theta[j]) + self.B[2][i] * np.sin(theta[j]))
                ax.plot3D(circ_x, circ_y, circ_z, line_color)
        patches.append(mpatches.Patch(color=line_color, label="Canal Surface Cross Sections"))

        # Plot a star on the graph to represent the robot base frame
        ax.plot3D([self.origin[0]], [self.origin[1]], [self.origin[2]], "black", marker = "*", markersize = 20)
        patches.append(mlines.Line2D([], [], color='black', marker='*', linestyle='None', label='Robot Base'))

        # Set plot limits for 3D plot and make equal to show accurate representation of canal surface
        xl, xr = plt.xlim()
        yl, yr = plt.ylim()
        zb, zt = ax.get_zlim()
        make_plot_even(ax, xl, xr, yl, yr, zb, zt)
        # Plot reproduced curve if there is one
        if reproduction:
            # Plot first point as a marker
            ax.plot3D([self.reproduced_curve[0][0]], [self.reproduced_curve[1][0]], [self.reproduced_curve[2][0]], c="green", marker='x', markersize=15)
            patches.append(mlines.Line2D([], [], color='green', marker='x', linestyle='None', label='Start of Reproduction'))
            # Plot the rest of the reproduced curve
            line_color = "lime"
            ax.plot3D(self.reproduced_curve[0], self.reproduced_curve[1], self.reproduced_curve[2], line_color)
            patches.append(mpatches.Patch(color=line_color, label="Reproduced Trajectory"))
            # Display trajectory in Rviz
            self.display_pose_array(np.array([self.reproduced_curve[0] - self.origin[0], 
                                              self.reproduced_curve[1] - self.origin[1], 
                                              self.reproduced_curve[2] - self.origin[2]]), 
                                    np.array([self.p0_for_reproduction.orientation.x, 
                                              self.p0_for_reproduction.orientation.y, 
                                              self.p0_for_reproduction.orientation.z, 
                                              self.p0_for_reproduction.orientation.w]))
        # Plot object of interest if user is performing a reaching task
        if self.reaching:
            # Plot point as a red x
            ax.plot3D([0.0], [0.0], [0.0], c="red", marker='x', markersize=20)
            patches.append(mlines.Line2D([], [], color='red', marker='x', linestyle='None', label='Object of Interest'))
        # Plot everything with a legend
        plt.legend(handles=patches)
        plt.show()


    def prompt_insertion_and_deletion(self):
        """ Driver that prompts user for insertion and deletion after an inital
            set of demonstrations has been collected"""
        # While the user does not want to proceed to canal surface calculation (wants to add/remove demonstrations)
        action = " "
        while action[0].upper() != "C":
            # Plot the smoothed demonstrations
            self.plot_curves()

            # Ask the user what action they would like to take
            prompt = ("\nPress 'D' to delete a demonstration.\nPress 'A' to add/record another demonstration\nPress 'C' to proceed to the canal surface\n\nEnter your response here: ")
            action = raw_input(prompt)
            while (len(action) == 0) or (not action[0].isalpha()) or (action[0].upper() not in ["D", "A", "C"]):
                action = raw_input(prompt)
            print("")

            # User wants to remove a demonstration
            if action[0].upper() == "D":
                self.delete_a_demonstration()

            # User wants to add a demonstration
            if action[0].upper() == "A":
                self.raw_demonstrations = inverse_reframe_curves(self.raw_demonstrations, self.origin)
                do_file_data = get_y_n("Would you like to load the demonstration from a file (Y/N)?: ")
                if do_file_data:
                    self.load_demonstration_from_file()
                else:
                    self.record_a_demonstration()
                self.process_raw_demonstrations()
                self.define_new_origin()


    def request_canal_surface(self, smooth=False):
        # Establish connection to canal surface server
        rospy.wait_for_service('/canal_surface')
        connect_to_algorithm = rospy.ServiceProxy('/canal_surface', CanalSrv)

        # Initialize a request variable for canal surface server
        canal_request = CanalSrvRequest()

        # Init demonstrations (must flatten to publish)
        dem_multi_arrays = []
        for demonstration in self.smooth_demonstrations:
            dem_multi_array = Float64MultiArray()
            dem_multi_array.data = np.frombuffer(demonstration.tobytes(), 'float64')
            dem_multi_array.layout.dim = [MultiArrayDimension()]
            dem_multi_array.layout.dim[0].stride = len(demonstration[0])
            dem_multi_arrays.append(dem_multi_array)
        canal_request.demonstrations = dem_multi_arrays

        # Call service
        canal_response = connect_to_algorithm(canal_request)

        # Recieve response and unpack variables
        self.directrix = np.array([canal_response.directrix.data[0:num_points_to_sample], 
                                   canal_response.directrix.data[num_points_to_sample:2*num_points_to_sample], 
                                   canal_response.directrix.data[2*num_points_to_sample:]])
        self.T = np.array([canal_response.tangent.data[0:num_points_to_sample], 
                                   canal_response.tangent.data[num_points_to_sample:2*num_points_to_sample], 
                                   canal_response.tangent.data[2*num_points_to_sample:]])
        self.N = np.array([canal_response.normal.data[0:num_points_to_sample], 
                                   canal_response.normal.data[num_points_to_sample:2*num_points_to_sample], 
                                   canal_response.normal.data[2*num_points_to_sample:]])
        self.B = np.array([canal_response.binormal.data[0:num_points_to_sample], 
                                   canal_response.binormal.data[num_points_to_sample:2*num_points_to_sample], 
                                   canal_response.binormal.data[2*num_points_to_sample:]])

        # If the user wishes, smooth the radii over time to create a
        # smoother path for the robot to follow during reproduction
        if smooth:
            self.radii = smooth_radii(canal_response.radii, step=3)
        else:
            self.radii = canal_response.radii


    def get_idx(self, random=False):
        """ Get the start and end values of where on the canal surface reproduction should take place """
        # Close any opened windows
        if plt.get_fignums():
            plt.close()
        
        # Open 2d Plot Showing radius over time
        ax2d = plt.axes()
        plt.title('Radii Over Time', size=16)
        ax2d.plot(range(len(self.radii)), self.radii)
        plt.show()

        # Ask for good start point for reproduction if reproducing randomly on such starting crossection
        if random:
            start = int(raw_input("\nAt what index would you like to start reproduction?: "))
            while start < 0 and start >= len(self.radii):
                start = int(raw_input("Plase enter a valid integer corresponding to a value on the x-axis of the shown graph: "))
        else:
            start = 0

        # Ask for good end point for reproduction
        end = (len(self.radii) + 1) - int(raw_input("\nAt what index would you like to end reproduction?: "))
        while end < 0 and end >= len(self.radii):
            end = (len(self.radii) + 1) - int(raw_input("Plase enter a valid integer corresponding to a value on the x-axis of the shown graph: "))

        return [start, end]


    def store_p0_as_current_position(self):
        """ Will record a single position of the robot to use as the starting point of reproduction """
        # Prompt user to move robot to the starting position
        print("\nPlease move the robot to your desired starting position for reproduction.\n\nPress 'Enter' when your robot it ready.")
        raw_input()
        # Record starting position, store, and return
        print("Recording starting position...\n")
        self.p0_for_reproduction = self.move_group.get_current_pose(self.eef_link).pose
        return self.p0_for_reproduction


    def get_reproduction(self, p0, idx):
        """ Calls CS algorithm function to create a reproduced curve given the CS, 
            a starting point, and a range saying where to reproduce """
        self.reproduced_curve = get_rep_traj(np.array([p0.position.x + self.origin[0], 
                                                       p0.position.y + self.origin[1], 
                                                       p0.position.z + self.origin[2]]), 
                                                       idx, self.directrix, self.radii, 
                                                       self.T, self.N, self.B)


    def store_random_p0(self, idx):
        """ Get a random starting position on the crossection of the canal surface where
            the user would like to begin reproduction """
        # Define a random angle of the initial point
        angle = np.random.uniform(0.0, (2.0 * np.pi))

        # Define a random magnitude for the initial point
        ratio = np.random.uniform(0.0, 1.0)
        mag = ratio * self.radii[idx[0]]

        # Determine point's x, y, and z based on angle, magnitude,
        # and the normal and binormal vectors at the starting directrix point
        self.p0_for_reproduction = geometry_msgs.msg.Pose()
        self.p0_for_reproduction.position.x = self.directrix[0][idx[0]] - self.origin[0] + mag * (self.N[0][idx[0]] * np.cos(angle) + self.B[0][idx[0]] * np.sin(angle))
        self.p0_for_reproduction.position.y = self.directrix[1][idx[0]] - self.origin[1] + mag * (self.N[1][idx[0]] * np.cos(angle) + self.B[1][idx[0]] * np.sin(angle))
        self.p0_for_reproduction.position.z = self.directrix[2][idx[0]] - self.origin[2] + mag * (self.N[2][idx[0]] * np.cos(angle) + self.B[2][idx[0]] * np.sin(angle))
        
        # Record current pose data to get valid orientation values
        curr_pose = self.move_group.get_current_pose(self.eef_link).pose
        self.p0_for_reproduction.orientation.x = curr_pose.orientation.x
        self.p0_for_reproduction.orientation.y = curr_pose.orientation.y
        self.p0_for_reproduction.orientation.z = curr_pose.orientation.z
        self.p0_for_reproduction.orientation.w = curr_pose.orientation.w

        # Return generated point
        return self.p0_for_reproduction


    def execute_reproduction(self, random=False):
        """ Attempts to execute the reproduced trajectory, prompting the robot to move 
            Returns T/F based on whether the robot was able to successfully reproduce the trajectory"""

        # Get the orientation of the robot's starting position
        orientation = np.array([self.p0_for_reproduction.orientation.x, 
                                self.p0_for_reproduction.orientation.y, 
                                self.p0_for_reproduction.orientation.z, 
                                self.p0_for_reproduction.orientation.w])

        # Shift curve back to coordinate system understood by the robot
        curve_to_reproduce = np.array([self.reproduced_curve[0] - self.origin[0], 
                                       self.reproduced_curve[1] - self.origin[1], 
                                       self.reproduced_curve[2] - self.origin[2]])

        # Create an array of waypoint pose objects based on 
        # the robot's starting orientation and the reproduced trajectory
        waypoints = xyz_to_pose(curve_to_reproduce, orientation)
        
        # Go to the first starting position, display plan to user
        self.move_group.set_goal_position_tolerance(0.001)
        self.move_group.set_goal_orientation_tolerance(0.1)
        self.move_group.set_start_state_to_current_state()
        plan = self.move_group.set_pose_target(waypoints[0])

        # Display plan in rviz
        rospy.sleep(2)
        self.display_trajectory(plan)
        rospy.sleep(2)

        # Go home, and stop. Plan multiple times if needed.
        print("\nPress 'Enter' to execute the current plan to the starting position of reproduction.\n")
        raw_input()
        plan_to_start_succeeded = self.move_group.go(wait=True)
        iterations = 0
        while not plan_to_start_succeeded and iterations < 3:
            self.move_group.set_start_state_to_current_state()
            plan = self.move_group.set_pose_target(waypoints[0])
            plan_to_start_succeeded = self.move_group.go(wait=True)
            self.move_group.stop()
            iterations += 1
        self.move_group.clear_pose_targets()

        # If unable to arrive at start, return failure
        if not plan_to_start_succeeded:
            print("Move group failed\n")
            return False
        else:
            print("\nArrived at starting position. Ready to follow path...")

        # Try to get a cartesian path plan for the robot using inverse kinematics. Simplify the plan if it fails.
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_goal_position_tolerance(0.02)
        self.move_group.set_goal_orientation_tolerance(2*np.pi)
        fraction = 0.0
        iterations = 0
        step = 1
        while fraction <= 0.99 and iterations < 3:
            print("Attempt #" + str(iterations+1) + " of cartesian path execution...\n")
            (plan, fraction) = self.move_group.compute_cartesian_path(
                                            waypoints[1::step],   # waypoints to follow
                                            0.01,                 # eef_step
                                            0.0)                  # jump_threshold
            rospy.sleep(0.5)
            self.move_group.set_goal_position_tolerance(self.move_group.get_goal_position_tolerance() + 0.01)
            step *= 2
            iterations += 1
        
        # Return false if no plan is found with 3 iterations
        if fraction < 0.99:
            print("Could not find plan.\n")
            return False

        print("Success! Path plan found (covers " + str(fraction * 100.0) + " percent of requested path). Publishing to Rviz...\n")

        # Display the trajectory in Rviz
        rospy.sleep(1)
        self.display_trajectory(plan)
        rospy.sleep(2)

        # Execute plan, and stop the robot
        print("Press 'Enter' to execute after viewing in RVIZ (MotionPlanning -> Planned Path -> Loop Animation)")
        raw_input()
        print("Executing...\n")
        self.move_group.execute(plan, wait=True)
        self.move_group.stop()

        print("All done!\n")

        return True


def main():
    try:
        """Main driver for user interaction"""
        print("Press 'Enter' to begin. Press 'Ctrl-D' to quit any time during the program.")
        raw_input()

        # Get number of demonstrations
        number_of_demonstrations = int(raw_input("How many demonstrations would you like to load or record?: "))
        while number_of_demonstrations < 2:
            number_of_demonstrations = int(raw_input("Please enter a valid number of demonstrations (> 1): "))
        
        # Ask whether the user is performing a reaching task
        do_reaching = get_y_n("\nAre you performing a reaching task, or any other task with a fixed end point (Y/N)?: ")

        # Init the ur5e arm moveit class
        ur5e_arm = MoveGroupPythonInterface(do_reaching)

        # Prompt the user for which type of data they will input;
        do_file_data = get_y_n("\nWould you like to load demonstrations from a file (Y/N)?: ")

        if do_file_data:
            # Collect initial set of raw demonstrations by recieving file input
            for i in range(number_of_demonstrations):
                saved_from_file = False
                while not saved_from_file:
                    print("For demonstration #" + str(i+1) + "...\n")
                    saved_from_file = ur5e_arm.load_demonstration_from_file()
                print("Demonstration #" + str(i+1) + "successfully loaded from file.\n")
        else:
            # Collect initial set of raw demonstrations by recording simulation
            for i in range(number_of_demonstrations):
                print("\nInitiating the recording for demonstration #" + str(i+1) + "...\n")
                rospy.sleep(3)
                recorded = False
                while (not recorded):
                    recorded = ur5e_arm.record_a_demonstration()
                    if recorded == False:
                        print("\nRestarting the recording for demonstration #" + str(i+1) + "...\n")
                        rospy.sleep(3)

        # Process initial set of raw demonstrations
        ur5e_arm.process_raw_demonstrations()

        # Ask user which demonstration is the best
        ur5e_arm.define_new_origin()
        
        # Ask the user if they would like to add/remove demonstrations before CS generation and reproduction
        raw_input("\nPress 'Enter' to display recorded demonstrations")
        ur5e_arm.prompt_insertion_and_deletion()

        # Whether the starting position is based off the recorded position of the robot, or just random
        random_p0 = False

        # Allow user to perform reproductions as many times as they want
        keep_reproducing = True
        while keep_reproducing:
            #Call service for canal surface generation
            ur5e_arm.request_canal_surface()
            ur5e_arm.plot_canal_surface(reproduction=False)

            #Initiate trajectory reproduction
            print("\nPress 'Enter' to proceed to reproduction.")
            raw_input()
            # Get boundries of reproduction and starting point        
            idx = ur5e_arm.get_idx(random = random_p0)
            if random_p0:
                p0 = ur5e_arm.store_random_p0(idx)
            else:  
                p0 = ur5e_arm.store_p0_as_current_position()
            # Get reproduction and show user
            ur5e_arm.get_reproduction(p0, idx)
            ur5e_arm.plot_canal_surface(reproduction=True)
            # Allow user to add and remove demonstrations to see how reproduction changes
            change_reproduction = get_y_n("Would you like to see how a reproduced trajectory would change by adding or removing demonstrations (Y/N)?: ")
            while change_reproduction:
                ur5e_arm.prompt_insertion_and_deletion()
                ur5e_arm.request_canal_surface()
                ur5e_arm.plot_canal_surface(reproduction=False)
                raw_input("Press 'Enter' to proceed to reproduction")
                # Get boundries and starting point again again
                idx = ur5e_arm.get_idx(random=random_p0)
                if random_p0:
                    p0 = ur5e_arm.store_random_p0(idx)
                else:
                    p0 = ur5e_arm.store_p0_as_current_position()
                # Show reproduction again, and prompt user if they would like to execute
                ur5e_arm.get_reproduction(p0, idx)
                ur5e_arm.plot_canal_surface(reproduction=True)
                change_reproduction = not get_y_n("Would you like the robot to execute this trajectory (Y/N)?: ")

            # Publish response from services to robot and initiate motion
            if not ur5e_arm.execute_reproduction(random=random_p0):
                # Smooth the canal surface and redo reproduction if unable to 
                # reproduce with current canal surface (likely due to noise)
                print("\nRestarting trajectory reproduction with a smoother canal surface. Don't move the robot.")
                ur5e_arm.request_canal_surface(smooth = True)
                if random_p0:
                    p0 = ur5e_arm.store_p0_as_current_position()
                ur5e_arm.get_reproduction(p0, idx)
                ur5e_arm.execute_reproduction(random=random_p0)

            

            # Prompt for more iterations of reproduction
            keep_reproducing = get_y_n("Would you like to do another reproduction (Y/N)?: ")
        
        print("\nPress 'Enter' to exit'")
        raw_input()

    except rospy.ROSInterruptException:
        return

    except KeyboardInterrupt:
        return

if __name__ == '__main__':
    main()
