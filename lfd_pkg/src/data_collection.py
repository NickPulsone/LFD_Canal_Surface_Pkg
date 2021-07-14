#!/usr/bin/env python

# Structure derived from program by Brendan Hertel, UML: https://github.com/brenhertel/Pearl-ur5e/blob/master/brendan_ur5e/src/scripts/demo_xyz_playback.py 

"""
   Will offer the user an interface to record demonstrations and process 
   them so that they can to be sent to an algorithm to construct a canal surface
   and perform trajectory reproduction
"""

import sys
import copy
# import keyboard
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi, ceil
from std_msgs.msg import String, Float64MultiArray, MultiArrayDimension
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from canal_surface_algorithm import smooth_a_demonstration, reframe_curves, inverse_reframe_curves, get_mean_old
from lfd_canal_surface_pkg.srv import CanalSrv, CanalSrvRequest


"""TUNABLE PARAMTERS"""
tolerance = 0.005
window_ratio = 0.1


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


def xyz_to_pose(cartesian_points):
    """Convert array of cartesian positions to array of pose objects (position and orientation)"""
    poses = []
    for i in range(len(cartesian_points[0])):
        pose = geometry_msgs.msg.Pose()
        pose.position.x = cartesian_points[0][i]
        pose.position.y = cartesian_points[1][i]
        pose.position.z = cartesian_points[2][i]
        poses.append(copy.deepcopy(pose))
    return poses


# How the robot is understood and controlled
class MoveGroupPythonInterface(object):
    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()
        #the moveit_commander is what is responsible for sending info the moveit controllers
        moveit_commander.roscpp_initialize(sys.argv)
        #initialize node
        rospy.init_node('data_collect', anonymous=True)
        #Instantiate a `RobotCommander`_ object. Provides information such as the robot's kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()
        #Instantiate a `PlanningSceneInterface`_ object. This provides a remote interface for getting, setting, and updating the robot's internal understanding of the surrounding world:
        scene = moveit_commander.PlanningSceneInterface()
        #Instantiate a `MoveGroupCommander`_ object.  This object is an interface to a planning group (group of joints), which in our moveit setup is named 'manipulator'
        group_name = "manipulator" 
        move_group = moveit_commander.MoveGroupCommander(group_name)
        #Create a `DisplayTrajectory`_ ROS publisher which is used to display trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
 
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
        # Colors used for plotting demonstrations
        self.plot_line_colors = np.array(["red", "green", "blue", "yellow", "pink", "black",
                                          "orange", "purple", "beige", "brown", "gray", "cyan", 
                                          "magenta"])
        # Ranges for plotting the curves and identifying them. Of the form [[xleft, xright], [yleft, yright], [zleft, zright]]
        self.plot_curve_ranges = []
        

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

        # Show user the demonstration
        cartesian_points = pose_to_xyz(wp)
        self.raw_demonstrations.append(cartesian_points)
        self.plot_curve_single(-1, raw=True)

        # Save demonstration if the user is satisfied, close plot
        save_dem = get_y_n("Would you like to save this demonstration (Y/N)?")
        plt.close()

        # Determine if the demonstration needs to be rerecorded. If not, add to data.
        if save_dem:
            self.raw_pose_demos.append(wp)
            smooth_demo = self.process_last_raw_demonstration()
            self.add_demonstration(smooth_demo)

        else:
            self.raw_demonstrations.pop()
        # Move robot back to start if user had wished to do so
        if return_to_start:
            self.move_group.go(start_joint_angles, wait=True)
            self.move_group.stop()
    
        return save_dem

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
            # Break if user changes mind
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

    def process_last_raw_demonstration(self):
        """Smooths and resamples the last recorded demonstration"""
        # Smooth the curve, resample based on given number of points
        num_pts_to_sample = 1000
        smooth_cartesian_points = smooth_a_demonstration(self.raw_demonstrations[-1], num_pts_to_sample)
        return smooth_cartesian_points

    def plot_curves(self):
        """ Plot all of the current processed demonstrations """
        # Close any opened windows
        if plt.get_fignums():
            plt.close()
        # Plot curves according to self.plot_line_colors and the order they were recorded 
        patches = []
        ax = plt.axes(projection='3d')
        plt.ion()
        for i in range(len(self.smooth_demonstrations)):
            line_color = self.plot_line_colors[i % len(self.plot_line_colors)]
            ax.plot3D(self.smooth_demonstrations[i][0], self.smooth_demonstrations[i][1], self.smooth_demonstrations[i][2], line_color)
            patch = mpatches.Patch(color=line_color, label="Recorded Demonstration #" + str(i+1))
            patches.append(patch)
        plt.legend(handles=patches)
        # Plot a star on the graph to represent the robot base frame
        ax.plot3D([-self.origin[0]], [-self.origin[1]], [-self.origin[2]], marker = "*", markersize = 20)
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
            ax.plot3D([0.0], [0.0], [0.0], marker = "*", markersize = 20)
        else:
            ax.plot3D(self.smooth_demonstrations[index][0], self.smooth_demonstrations[index][1], self.smooth_demonstrations[index][2], line_color)
            # Plot a star on the graph to represent the robot base frame
            ax.plot3D([-self.origin[0]], [-self.origin[1]], [-self.origin[2]], marker = "*", markersize = 20)
            # Sey x, y, and z limits according to previously plotted demonstrations
            plt.xlim(self.plot_curve_ranges[0][0], self.plot_curve_ranges[0][1])
            plt.ylim(self.plot_curve_ranges[1][0], self.plot_curve_ranges[1][1])
            ax.set_zlim(self.plot_curve_ranges[2][0], self.plot_curve_ranges[2][1])
        # If referencing the last demonstration, don't label according to index
        if index == -1:
            patch = mpatches.Patch(color=line_color, label="Recorded Demonstration")
        else:
            patch = mpatches.Patch(color=line_color, label="Recorded Demonstration #" + str(index + 1))
        # Create legend
        plt.legend(handles=[patch])
            
        plt.show()

    def plot_canal_surface(self):
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
            circ_x = np.empty(theta_density)
            circ_y = np.empty(theta_density)
            circ_z = np.empty(theta_density)
            # Use calculated radii and the Normal and Binormal vectors to plot the circles
            for j in range(theta_density):
                circ_x[j] = self.directrix[0][i] + self.radii[i] * (self.N[0][i] * np.cos(theta[j]) + self.B[0][i] * np.sin(theta[j]))
                circ_y[j] = self.directrix[1][i] + self.radii[i] * (self.N[1][i] * np.cos(theta[j]) + self.B[1][i] * np.sin(theta[j]))
                circ_z[j] = self.directrix[2][i] + self.radii[i] * (self.N[2][i] * np.cos(theta[j]) + self.B[2][i] * np.sin(theta[j]))
                ax.plot3D(circ_x, circ_y, circ_z, line_color)
        patches.append(mpatches.Patch(color=line_color, label="Canal Surface Cross Sections"))
        plt.legend(handles=patches)
        # Plot a star on the graph to represent the robot base frame
        ax.plot3D([-self.origin[0]], [-self.origin[1]], [-self.origin[2]], "black", marker = "*", markersize = 20)
        # Update the plot curve ranges
        self.plot_curve_ranges = [plt.xlim(), plt.ylim(), ax.get_zlim()]
        plt.show()

    def add_demonstration(self, demo):
        """ Adds a processed demonstration to the list of smoothed demonstrations, 
            calculates the new directrix, and reframes the coordinate system accordingly"""
        # Case of empty set of demonstrations 
        if len(self.smooth_demonstrations) == 0:
            self.smooth_demonstrations.append(demo)
            self.directrix = copy.deepcopy(demo)
        # Convert existing demonstrations into the standard coordinate system read by the robot
        else:
            self.smooth_demonstrations = inverse_reframe_curves(self.smooth_demonstrations, self.origin)
            self.smooth_demonstrations.append(demo)
            # Calculate direcrix
            self.directrix = get_mean_old(self.smooth_demonstrations)
        # Define new origin with new directrix, and reframe data accordingly
        self.origin[0] = self.directrix[0][-1]
        self.origin[1] = self.directrix[1][-1]
        self.origin[2] = self.directrix[2][-1]
        self.smooth_demonstrations = reframe_curves(self.smooth_demonstrations, self.origin)
        self.directrix = reframe_curves([self.directrix], self.origin)[0]
            

    def remove_demonstration(self, index):
        """ Removes a processed demonstration from the list 
            and recalculates the new directrix and coordinate reference """
        # Remove demonstration
        self.smooth_demonstrations.pop(index)
        # Convert existing demonstrations into the standard coordinate system read by the robot
        self.smooth_demonstrations = inverse_reframe_curves(self.smooth_demonstrations, self.origin)
        # Recalculate the directrix
        if len(self.smooth_demonstrations) == 1:
            self.directrix = copy.deepcopy(self.smooth_demonstrations[0])
        else:
            # Calculate direcrix
            self.directrix = get_mean_old(self.smooth_demonstrations)
        # Define new origin with new directrix, and reframe data accordingly
        self.origin[0] = self.directrix[0][-1]
        self.origin[1] = self.directrix[1][-1]
        self.origin[2] = self.directrix[2][-1]
        self.smooth_demonstrations = reframe_curves(self.smooth_demonstrations, self.origin)
        self.directrix = reframe_curves([self.directrix], self.origin)[0]

    def prompt_insertion_and_deletion(self):
        """ Driver that prompts user for insertion and deletion after an inital
            set of demonstrations has been collected"""
        # While the user does not want to proceed to canal surface calculation (wants to add/remove demonstrations)
        action = " "
        while action[0].upper() != "C":
            # Plot
            self.plot_curves()
            # Ask the user what action they would like to take
            prompt = ("\nPress 'D' to delete a demonstration.\nPress 'A' to add/record another demonstration\nPress 'C' to proceed to the canal surface\n\nEnter your response here: ")
            action = raw_input(prompt)
            while (len(action) == 0) or (not action[0].isalpha()) or (action[0].upper() not in ["D", "A", "C"]):
                action = raw_input(prompt)
            # User wants to remove a demonstration
            if action[0].upper() == "D":
                self.delete_a_demonstration()
            # User wants to add a demonstration
            if action[0].upper() == "A":
                self.record_a_demonstration()

    def request_canal_surface(self):
        # Establish connection to canal surface server
        rospy.wait_for_service('/canal_surface')
        connect_to_algorithm = rospy.ServiceProxy('/canal_surface', CanalSrv)
        # Initialize a request variable for canal surface server
        canal_request = CanalSrvRequest()
        # Init directrix (must flatten to publish)
        canal_request.directrix = Float64MultiArray()
        canal_request.directrix.data = np.frombuffer(self.directrix.tobytes(),'float64')
        canal_request.directrix.layout.dim = [MultiArrayDimension()]
        canal_request.directrix.layout.dim[0].stride = len(self.directrix[0])
        # Init demonstrations (must flatten to publish)
        dem_multi_arrays = []
        for demonstration in self.smooth_demonstrations:
            dem_multi_array = Float64MultiArray()
            dem_multi_array.data = np.frombuffer(demonstration.tobytes(), 'float64')
            dem_multi_array.layout.dim = [MultiArrayDimension()]
            dem_multi_array.layout.dim[0].stride = len(demonstration[0])
            dem_multi_arrays.append(dem_multi_array)
        canal_request.demonstrations = dem_multi_arrays
        # Define tolerance and window size according to global parameters
        canal_request.tolerance = tolerance
        canal_request.window_ratio = window_ratio
        # Call service, recieve response, unpack variables
        canal_response = connect_to_algorithm(canal_request)
        sliced_length = canal_response.sliced_directrix.layout.dim[0].stride
        self.directrix = np.array([canal_response.sliced_directrix.data[0:sliced_length], 
                                   canal_response.sliced_directrix.data[sliced_length:2*sliced_length], 
                                   canal_response.sliced_directrix.data[2*sliced_length:]])
        self.T = np.array([canal_response.tangent.data[0:sliced_length], 
                                   canal_response.tangent.data[sliced_length:2*sliced_length], 
                                   canal_response.tangent.data[2*sliced_length:]])
        self.N = np.array([canal_response.normal.data[0:sliced_length], 
                                   canal_response.normal.data[sliced_length:2*sliced_length], 
                                   canal_response.normal.data[2*sliced_length:]])
        self.B = np.array([canal_response.binormal.data[0:sliced_length], 
                                   canal_response.binormal.data[sliced_length:2*sliced_length], 
                                   canal_response.binormal.data[2*sliced_length:]])
        self.radii = canal_response.radii


def main():
    try:
        """Main driver for user interaction"""
        print("Press 'Enter' to begin. Press 'Ctrl-D' to quit any time during the program.")
        raw_input()

        # Get number of demonstrations
        number_of_demonstrations = int(raw_input("How many demonstrations would you like to record?: "))
        while number_of_demonstrations < 2:
            number_of_demonstrations = int(raw_input("Please enter a valid number of demonstrations (> 1): "))

        # Init the ur5e arm moveit class
        ur5e_arm = MoveGroupPythonInterface()

        # Loop through for each demonstration, providing an interface for users to record, save, and delete
        for i in range(number_of_demonstrations):
            print("\nInitiating the recording for demonstration #" + str(i+1) + "...\n")
            rospy.sleep(3)
            recorded = False
            while (not recorded):
                recorded = ur5e_arm.record_a_demonstration()
                if recorded == False:
                    print("\nRestarting the recording for demonstration #" + str(i+1) + "...\n")
                    rospy.sleep(3)
        raw_input("\nPress 'Enter' to display recorded demonstrations")
        ur5e_arm.prompt_insertion_and_deletion()

        """TODO: Call service for canal surface generation"""
        """
        ur5e_arm.request_canal_surface()
        ur5e_arm.plot_canal_surface()
        """

        """TODO: Call service for trajectory reproduction"""

        """TODO: Publish response from services to robot and initiate motion"""

        print("\nPress 'Enter' to exit'")
        raw_input()

    except rospy.ROSInterruptException:
        return

    except KeyboardInterrupt:
        return

if __name__ == '__main__':
    main()
