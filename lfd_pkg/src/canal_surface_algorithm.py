#!/usr/bin/env python
"""
Author: Nicholas Pulsone

Canal Surface Generator, and Trajectory Reproduction based on Algorithms 1 and 2 in
"Encoding Demonstrations and Learning New Trajectories using Canal Surfaces"
by S. Reza Ahmadzadeh and Sonia Chernova (2016)

The following program will generate a canal surface (a set of points on a directrix/mean curve
and a set of radii at each point) based on demonstration data. It will then use the generated
canal surface with an initial starting position to reproduce a trajectory according to the input.

IMPORTANT: 2 Parameters must be tuned for the program to yield accurate results: window_ratio and tolerance (see main)

NOTE: All 3D Curves assumed to be in the form [array_of_xs, array_of_ys, array_of_zs], where
array[:, 4] refers to the cartesian coordinate of the fourth point on the demonstration.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy import interpolate

def normalization(vector):
    """Normalizes a 3D vector"""
    if np.linalg.norm(vector) == 0.0:
        return np.array([0.0, 0.0, 0.0])
    return vector / np.linalg.norm(vector)


def get_demonstration_data(num_dem, time_st_ct):
    """Creates a set of synthetic trajectories used to generate the Canal Surface"""
    trajectories = []
    type_of_data = "Arctan Data"
    for i in range(num_dem):
        if type_of_data == "Wavy Data":
            z = np.linspace(0, 1, time_st_ct)
            y = np.array([np.sin(1.5 * z[i]) for i in range(time_st_ct)])
            y = np.add(y, 0.5 * np.random.rand())
            x = np.array([np.cos(1.5 * z[i]) for i in range(time_st_ct)])
            x = np.add(x, 0.5 * np.random.rand())
        elif type_of_data == "Hourglass Data":
            # Generate multipliers so that we get our hourglass shape
            x_multiplier = 5.0 * np.random.rand()
            y_multiplier = 5.0 * np.random.rand()
            if (i + 1) % 2 == 0:
                x_multiplier = -5.0
            if (i + 1) % 3 == 0 or (i + 1) % 4 == 0:
                y_multiplier = -5.0
            # Generate vector, normalize it
            vector = np.array([0.5 * x_multiplier, 0.5 * y_multiplier])
            vector = normalization(vector)
            # Define a reference x and y to form parabolic shape
            ref_x = np.linspace(-1, 1, time_st_ct)
            ref_y = -0.5 * np.square(ref_x)
            # Get x, y, and z based on reference x and y and the vector
            x = np.linspace(-1, 1, time_st_ct)
            y = np.array([vector[0] * ref_y[i] for i in range(time_st_ct)])
            z = np.array([vector[1] * ref_y[i] for i in range(time_st_ct)])
        elif type_of_data == "Noisy Sinewave Data":
            # Generate vector, normalize it
            vector = np.array([0.3 * np.random.rand(), 0.3 * np.random.rand()])
            vector = normalization(vector)
            # Define a reference x and y to form parabolic shape
            ref_x = np.linspace(0.0, 2 * np.pi, time_st_ct)
            ref_y = np.sin(ref_x)
            # Get x, y, and z based on reference x and y and the vector
            x = np.linspace(-1, 1, time_st_ct)
            y = np.array([vector[0] * ref_y[i] + 0.01 * np.random.randn() for i in range(time_st_ct)])
            z = np.array([vector[1] * ref_y[i] + 0.01 * np.random.randn() for i in range(time_st_ct)])
        else:
            """Noisy Arctan data"""
            # Generate vector, normalize it
            vector = np.array([np.random.rand(), np.random.rand()])
            vector = normalization(vector)
            # Define a reference x and y to form parabolic shape
            ref_x = np.linspace(0.0, 2 * np.pi, time_st_ct)
            ref_y = np.arctan(2 * ref_x)
            # Get x, y, and z based on reference x and y and the vector
            x = np.linspace(-1, 1, time_st_ct)
            y = np.array([vector[0] * ref_y[i] + 0.05 * np.random.randn() for i in range(time_st_ct)])
            z = np.array([vector[1] * ref_y[i] + 0.05 * np.random.randn() for i in range(time_st_ct)])

        trajectories.append([x, y, z])
    return np.array(trajectories)


def sample_raw_data(raw_data, num_generated_points):
    """Takes an array with raw demonstrations, uses spline interpolation to smooth and resample data
       with a number of points specified by num_generated points. Used mostly for synthetic data, or
       data loaded from a file."""
    generated_data = []
    # For each trajectory, interpolate the data and resample based on number of desired points
    num_trajectories = len(raw_data)
    for i in range(num_trajectories):
        # Define a time range based on the number of captured points
        num_captured_points = len(raw_data[i][0])
        time_range = np.array(range(num_captured_points))
        t_function = np.linspace(0.0, num_captured_points - 1, num_generated_points)

        # Gerneate 1D interpolation functions for the x, y, and z of each trajectory
        x_est_func = interpolate.UnivariateSpline(time_range, raw_data[i][0])
        y_est_func = interpolate.UnivariateSpline(time_range, raw_data[i][1])
        z_est_func = interpolate.UnivariateSpline(time_range, raw_data[i][2])
        # generated_functions.append(np.array([x_est_func, y_est_func, z_est_func]))

        # Sample from the interpolated functions
        x = np.array([x_est_func(t) for t in t_function])
        y = np.array([y_est_func(t) for t in t_function])
        z = np.array([z_est_func(t) for t in t_function])
        generated_data.append(np.array([x, y, z]))

    return np.array(generated_data)


def smooth_a_trajectory(raw_data, num_generated_points):
    """Smooths a single trajectory recorded from the robot. Used for real cases on the
       robot when smoothing is needed immedietly after recording a demonstration."""
    # Init New arrays for trimmed x, y, and z
    trim_x = [raw_data[0][0]]
    trim_y = [raw_data[1][0]]
    trim_z = [raw_data[2][0]]
    # Walk through the list of points and ignore adjacent points that are practically identical
    for i in range(len(raw_data[0]) - 1):
        if not (np.absolute(raw_data[0][i] - raw_data[0][i + 1]) < 0.0001 and np.absolute(
                raw_data[1][i] - raw_data[1][i + 1]) < 0.0001 and np.absolute(
                raw_data[2][i] - raw_data[2][i + 1]) < 0.0001):
            trim_x.append(raw_data[0][i + 1])
            trim_y.append(raw_data[1][i + 1])
            trim_z.append(raw_data[2][i + 1])

    # Define a time range based on the number of points after trimming, and a new time function for post-smoothing
    num_captured_points = len(trim_x)
    time_range = np.array(range(num_captured_points))
    t_function = np.linspace(0.0, num_captured_points - 1, num_generated_points)

    # Gerneate 1D interpolation functions for the x, y, and z of each trajectory
    # Could use interp1d, UnivariateSpline, or other methods for smooth interpolation
    """
    x_est_func = interpolate.UnivariateSpline(time_range, trim_x, s=10)
    y_est_func = interpolate.UnivariateSpline(time_range, trim_y, s=10)
    z_est_func = interpolate.UnivariateSpline(time_range, trim_z, s=10)
    """
    x_est_func = interpolate.interp1d(time_range, trim_x)
    y_est_func = interpolate.interp1d(time_range, trim_y)
    z_est_func = interpolate.interp1d(time_range, trim_z)

    # Sample from the interpolated functions
    x = np.array([x_est_func(t) for t in t_function])
    y = np.array([y_est_func(t) for t in t_function])
    z = np.array([z_est_func(t) for t in t_function])

    # Return smoothed and trimmed array
    smoothed_points = np.array([x, y, z])

    return smoothed_points


def get_mean_old(data):
    """(Python 2) Calculates the directrix curve given a set of trajectories"""
    """NOTE: Assumes data in the form of demonstrations with the same 
             number of points"""
    num_trajectories = len(data)
    # Init arrays for x, y, and z values of the directrix
    x_mean = np.zeros(len(data[0][0]), dtype=float)
    y_mean = np.zeros(len(data[0][1]), dtype=float)
    z_mean = np.zeros(len(data[0][2]), dtype=float)

    # Sum the x, y, z components of all the trajectories
    for trajectory in data:
        x_mean += trajectory[0]
        y_mean += trajectory[1]
        z_mean += trajectory[2]

    # Divide by number of trajectories to get the mean
    x_mean /= num_trajectories
    y_mean /= num_trajectories
    z_mean /= num_trajectories

    dirx = np.array([x_mean, y_mean, z_mean])
    return dirx


def get_mean(data):
    """(Python 3+) Calculates the directrix curve given a set of trajectories"""
    """NOTE: Assumes data in the form of demonstrations with the same 
             number of points"""
    # Init arrays for x, y, and z values of the directrix
    x_mean = np.mean(data[:, 0], 0)
    y_mean = np.mean(data[:, 1], 0)
    z_mean = np.mean(data[:, 2], 0)
    return np.array([x_mean, y_mean, z_mean])


def reframe_curves(curves, pt):
    """Make the given point the new origin, with respect to the given data.
       Data given in the form of an array of 3D curves."""
    for i in range(len(curves)):
        curves[i][0] -= pt[0]
        curves[i][1] -= pt[1]
        curves[i][2] -= pt[2]
    return curves


def inverse_reframe_curves(curves, pt):
    """Inverts the change made by reframe_curves to make data suitable to be
       interpreted by the robot."""
    for i in range(len(curves)):
        curves[i][0] += pt[0]
        curves[i][1] += pt[1]
        curves[i][2] += pt[2]
    return curves


def get_tnb(dirx):
    """Calculates the TNB frame at each time step (index) of the directrix per section 3 of 'Encoding Demonstrations'"""
    """Returns data in the form T: [[Tx], [Ty], [Tz]], N: [[Nx], [Ny], [Nz]], B: [[Bx], [By], [Bz]]"""
    # Calculate, normalize Tangent
    et = np.apply_along_axis(np.gradient, axis=1, arr=dirx)
    et = np.apply_along_axis(normalization, axis=0, arr=et)

    # Memory allocation for Normal and Binormal
    en = np.zeros((3, len(et[0]) + 1), dtype=float)
    eb = np.zeros((3, len(et[0])), dtype=float)

    # Get first Normal vector
    en[0, 0] = et[2, 0] * et[0, 0]
    en[1, 0] = et[2, 0] * et[1, 0]
    en[2, 0] = -(np.square(et[0, 0]) + np.square(et[1, 0]))
    en[::, 0] = normalization(en[::, 0])

    # Propogate N and B along directrix based on first Normal
    for i in range(len(et[0])):
        eb[::, i] = normalization(np.cross(en[::, i], et[::, i]))
        en[::, i + 1] = normalization(np.cross(et[::, i], eb[::, i]))

    # Normailze Normal and Binormal
    en = np.apply_along_axis(normalization, axis=0, arr=en)
    eb = np.apply_along_axis(normalization, axis=0, arr=eb)

    return et, en[::, 1::], eb


def get_distance(p1, p2):
    """Calculates the distance between 2 points in 3 dimensional space"""
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]) + np.square(p1[2] - p2[2]))


def define_plane(p, t):
    """Calculates the offset (d) parameter of a plane given a point
    on the directrix (p) and the tangent vector (t)"""
    return -1.0 * ((t[0] * p[0]) + (t[1] * p[1]) + (t[2] * p[2]))


def slice_data(demonstration_data, directrix, t, n, b, num_points, tolerance, window_size):
    """Slices the directrix at the ends so that at every directrix point, a plane can be defined
       that intersects every demonstration.
       Demonstration data = array of 3D curves
       Dirextrix = 3D mean curve
       t, n, b are the TNB frames
       Requires a defined tolerance and window_size (See main)"""
    # Define the number of trajectories
    num_traj = len(demonstration_data)

    # Start at beggining, find first point where a plane can be defined
    start_plane_defined = False
    start_i = 0
    while not start_plane_defined:
        # Define plane at point of interest
        d = define_plane(directrix[::, start_i], t[::, start_i])
        # Start by assuming it is defined
        start_plane_defined = True
        for i in range(num_traj):
            # Define comparison array, find point closest to plane of interest
            comparison_array = np.absolute(
                t[0, start_i] * demonstration_data[i][0] + t[1, start_i] * demonstration_data[i][1] + t[
                    2, start_i] * demonstration_data[i][2] + d)
            min_index = np.where(comparison_array == np.amin(comparison_array))
            min_index = min_index[0][0]
            # If the point is too far away, we know we can not define the start yet
            if comparison_array[min_index] > tolerance:
                start_plane_defined = False
            else:
                # Ensure that point that is tolerable is within a reasonable distance from the directrix point,
                # defined by window_size
                while comparison_array[min_index] <= tolerance and np.absolute(start_i - min_index) > window_size:
                    comparison_array = np.delete(comparison_array, min_index)
                    min_index = np.where(comparison_array == np.amin(comparison_array))
                    min_index = min_index[0][0]
                    # Break if no reasonable point is found on the demonstration
                    if len(comparison_array) == 0:
                        break
                # If no reasonable point found, move to the next point
                if not (np.absolute(start_i - min_index) <= window_size and comparison_array[min_index] <= tolerance and
                        len(comparison_array) != 0):
                    start_plane_defined = False
        start_i += 1

    # Start from end, find last point where plane can be defined
    end_plane_defined = False
    end_i = num_points - 1
    while not end_plane_defined:
        # Define plane at point of interest
        d = define_plane(directrix[::, end_i], t[::, end_i])
        # Start by assuming it is defined
        end_plane_defined = True
        for i in range(num_traj):
            # Define comparison array
            comparison_array = np.absolute(t[0, end_i] * demonstration_data[i][0] + t[1, end_i] * demonstration_data[i][1] + t[
                2, end_i] * demonstration_data[i][2] + d)
            min_index = np.where(comparison_array == np.amin(comparison_array))
            min_index = min_index[0][0]
            # If the point is too far away, we know we can not define the end yet
            if comparison_array[min_index] > tolerance:
                end_plane_defined = False
            else:
                # Ensure that point that is tolerable is within a reasonable distance from the directrix point,
                # defined by window_size
                while comparison_array[min_index] <= tolerance and np.absolute(end_i - min_index) > window_size:
                    comparison_array = np.delete(comparison_array, min_index)
                    min_index = np.where(comparison_array == np.amin(comparison_array))
                    min_index = min_index[0][0]
                    # Break if no reasonable point is found on the demonstration
                    if len(comparison_array) == 0:
                        break
                # If no reasonable point found, move to the next point
                if not (np.absolute(end_i - min_index) <= window_size and comparison_array[min_index] <= tolerance and
                        len(comparison_array) != 0):
                    end_plane_defined = False
        end_i -= 1
    # Return sliced directrix and TNB frames based on the start and end indices found
    return directrix[::, start_i:(end_i + 1)], t[::, start_i:(end_i + 1)], n[::, start_i:(end_i + 1)], b[::, start_i:(
            end_i + 1)]


def get_canal_surface(dirx, data, et, tolerance):
    """Generates a canal surface (list of radii with as many points as the directrix)
       Data = array of 3D curves
       dirx = 3D mean curve
       et = is the tangent vectors at each point on the directrix (See get_tnb)
       """
    # Define a number of points and number of trajectories
    num_points = len(dirx[0])

    # Define a container for the radii
    radii = np.empty(num_points)

    # Calculated distance to directrix of each demonstration for the point at (i - 1)
    prev_radii = np.zeros(len(data), dtype=float)

    # Loop over points in the directrix
    for i in range(num_points):
        # Define the directrix point at each step
        pt = dirx[:, i]

        # Define the plane at the current step
        d = define_plane(pt, et[:, i])

        # Loop over each demonstration
        possible_radii = np.empty(len(data))
        for j in range(len(data)):
            # Find closes point on the demonstration to the plane of interest
            points = data[j, :, :]
            point_comparisons = np.absolute(
                et[0][i] * points[0] + et[1][i] * points[1] + et[2][i] * points[2] + d)
            best_pt_index = np.where(point_comparisons == min(point_comparisons))
            best_pt_index = best_pt_index[0][0]

            # Calculate distance between point of interest and directrix point, store for later comparison
            possible_radii[j] = get_distance(points[:, best_pt_index], pt)
            if i != 0:
                # If there is a massive jump between radii, we know that the radius is inaccurate
                # (happens when plane intersects multiple points on a demonstration)
                check_values = np.zeros(len(point_comparisons), dtype=int)
                while possible_radii[j] >= prev_radii[j] * 1.1 or possible_radii[j] <= prev_radii[j] * 0.9:
                    # If it cant find a point within a tolerance, just maintain the radius from the previous point
                    # (In progress/To be modified)
                    if point_comparisons[best_pt_index] > tolerance:
                        possible_radii[j] = prev_radii[j - 1]
                        break
                    # Delete innaccurate point, find the next closest point to the plane
                    point_comparisons = np.delete(point_comparisons, best_pt_index)
                    best_pt_index = np.where(point_comparisons == min(point_comparisons))
                    best_pt_index = best_pt_index[0][0]
                    possible_radii[j] = get_distance(points[:, best_pt_index], pt)
            # Keep track of the radius before the current point
            prev_radii[j] = possible_radii[j]

        # Find maximum of relevant points
        radii[i] = np.amax(possible_radii)
    return radii


def get_p0(i, dirx, ri, en, eb):
    """Will generate a random point on the a cross section of the canal surface
       i = index of the cross section where the point in to be defined
       dirx = 3D mean curve
       ri = radii of the canal surface
       en, eb = Normal and Binormal vectors (see get_tnb)"""
    # Define a random magnitude and angle
    angle = np.random.uniform(0.0, (2.0 * np.pi))
    mag = np.random.uniform(0.0, ri[i])

    # Determine point's x, y, and z based on angle, magnitude,
    # and the normal and binormal vectors at the starting directrix point
    x = dirx[0][i] + mag * (en[0][i] * np.cos(angle) + eb[0][i] * np.sin(angle))
    y = dirx[1][i] + mag * (en[1][i] * np.cos(angle) + eb[1][i] * np.sin(angle))
    z = dirx[2][i] + mag * (en[2][i] * np.cos(angle) + eb[2][i] * np.sin(angle))
    return np.array([x, y, z])


def get_rep_traj(p0, dirx, ri, et, en, eb):
    """Create a reproduced trajectory from an initial starting point,
       p0 using the generated canal surface
       p0 = initial position of the robot [x, y, z]
       dirx = 3D mean curve
       ri = radii of the canal surface
       et, en, eb = arrays of tangent, normal, and binormal vectors respectively (see get_tnb)"""

    # Determine index of previous/starting point relative to the frames
    comparison_array = np.array([np.absolute(
        define_plane(np.array([dirx[0][j], dirx[1][j], dirx[2][j]]), np.array([et[0][j], et[1][j], et[2][j]])) +
        et[0][j] * p0[0] + et[1][j] * p0[1] + et[2][j] * p0[2]) for j in range(len(dirx[0]))])
    p0_index = np.where(comparison_array == np.amin(comparison_array))
    p0_index = p0_index[0][0]

    # Calculate length between directrix and starting point
    p0c0 = get_distance(p0, np.array([dirx[0][p0_index], dirx[1][p0_index], dirx[2][p0_index]]))

    # Determine ratio
    if p0c0 > ri[p0_index]:
        ratio = 1.0
    else:
        ratio = p0c0 / ri[p0_index]

    # Define lists for the x, y, and z of the reproduced trajectory
    x = []
    y = []
    z = []

    # Center point vector at the origin
    p0 = np.array([p0[0] - dirx[0][p0_index], p0[1] - dirx[1][p0_index], p0[2] - dirx[2][p0_index]])

    # Define transformation matrix between XYZ and initial point TNB
    p0_frame = np.array([[et[0][p0_index], et[1][p0_index], et[2][p0_index]],
                         [en[0][p0_index], en[1][p0_index], en[2][p0_index]],
                         [eb[0][p0_index], eb[1][p0_index], eb[2][p0_index]]])

    # Perform XYZ to Inital Point's TNB Transformation
    p0 = np.matmul(p0_frame, p0)

    # Force tangent component to 0 for minimal error accumulation
    p0[0] = 0.0

    # Translate the point across the canal surface, once for each point on the directrix
    i = p0_index
    while i < len(ri):
        # Determine transformation matrix between XYZ and current point's TNB frame
        curr_frame = np.array([[et[0][i], et[1][i], et[2][i]],
                               [en[0][i], en[1][i], en[2][i]],
                               [eb[0][i], eb[1][i], eb[2][i]]])

        # Determine transformation matrix between the first points TNB frame and the current points TNB frame
        T = np.array([[(np.dot(p0_frame[0], curr_frame[0])), (np.dot(p0_frame[1], curr_frame[0])),
                       (np.dot(p0_frame[2], curr_frame[0]))],
                      [(np.dot(p0_frame[0], curr_frame[1])), (np.dot(p0_frame[1], curr_frame[1])),
                       (np.dot(p0_frame[2], curr_frame[1]))],
                      [(np.dot(p0_frame[0], curr_frame[2])), (np.dot(p0_frame[1], curr_frame[2])),
                       (np.dot(p0_frame[2], curr_frame[2]))]])

        # Perform initial point's TNB frame to current point's TNB frame transformation
        new_pt = np.matmul(T, p0)

        # Force tangent component to 0 for minimal error accumulation
        new_pt[0] = 0.0

        # Perform current point's TNB frame to XYZ transformation using the inverse of current point's TNB frame
        new_pt = np.matmul(np.rot90(np.fliplr(curr_frame)), new_pt)

        # Scale the point in accordance with the ratio rule
        new_pt = (ratio * ri[i] * new_pt) / np.linalg.norm(new_pt)

        # Add to new point to the list of x, y, and z of reproduced curve
        x.append(dirx[0][i] + new_pt[0])
        y.append(dirx[1][i] + new_pt[1])
        z.append(dirx[2][i] + new_pt[2])

        # Move to the next point
        i += 1
    return np.array([x, y, z])


def plot_canal_surface_and_result(data, dirx, ri, reproduced_traj, et, en, eb, num_circ=20):
    """Plot the demonstration data, canal surface, and reproduction all at once.
       Used almost solely for debugging, as in real cases data is shown to the user periodically.
       data = array of 3D curves
       dirx = 3D mean curve
       ri = radii of the canal surface
       reproduced_traj = reproduced trajectory using canal surface
       et, en, eb = arrays of tangent, normal, and binormal vectors respectively (see get_tnb)
       num_circ = number of cricles to show on the plot (not exact since data can have varying lengths)"""

    # Choose what will be shown in the plot
    do_plot_demonstrations = True
    do_plot_directrix = True
    do_plot_circles = True
    do_plot_reproduction = True
    do_plot_TNB = False
    do_plot_point_markers = False
    do_plot_radii_v_time = False

    # Set up 3D plot
    ax = plt.axes(projection='3d')

    # Plot Demonstrations
    if do_plot_demonstrations:
        for dem in data:
            ax.plot3D(dem[0], dem[1], dem[2], 'blue')

    # Plot the Directrix
    if do_plot_directrix:
        ax.plot3D(dirx[0], dirx[1], dirx[2], 'red')

    # Plot the canal surface using paramaterized functions for circles
    if do_plot_circles:
        # Define angle function
        theta_step_ct = 20
        theta = np.linspace(0, 2 * np.pi, theta_step_ct)
        for i in range(0, len(ri), int(math.ceil(len(ri) / num_circ))):
            # Init array containers for x, y, and z of circles
            x = np.zeros(theta_step_ct)
            y = np.zeros(theta_step_ct)
            z = np.zeros(theta_step_ct)
            # Use calculated radii and the Normal and Binormal vectors to plot the circles
            for j in range(theta_step_ct):
                x[j] = dirx[0][i] + ri[i] * (en[0][i] * np.cos(theta[j]) + eb[0][i] * np.sin(theta[j]))
                y[j] = dirx[1][i] + ri[i] * (en[1][i] * np.cos(theta[j]) + eb[1][i] * np.sin(theta[j]))
                z[j] = dirx[2][i] + ri[i] * (en[2][i] * np.cos(theta[j]) + eb[2][i] * np.sin(theta[j]))
                ax.plot3D(x, y, z, 'gray')

    # Plot radius as a function of time
    if do_plot_circles and do_plot_radii_v_time:
        ax2d = plt.axes()
        plt.title('Radii over time', size=16)
        ax2d.plot(range(len(et[0])), ri)

    # Plot starting point and reproduced trajectory
    if do_plot_reproduction:
        # Plot initial point
        ax.plot3D([reproduced_traj[0][0]], [reproduced_traj[1][0]], [reproduced_traj[2][0]], marker='*', markersize=15)
        # Plot Reproduced trajectory line
        ax.plot3D(reproduced_traj[0], reproduced_traj[1], reproduced_traj[2], 'lime')

    # Plot point markers (for debugging reproduced curve)
    if do_plot_point_markers:
        for i in range(len(reproduced_traj[0])):
            ax.plot3D(reproduced_traj[0][i], reproduced_traj[1][i], reproduced_traj[2][i], marker='.', markersize=20)

    # Plot T, N, B parameters (for various debugging)
    if do_plot_TNB:
        # Change factor based on size of data
        et *= 0.2
        en *= 0.2
        eb *= 0.2
        for i in range(0, len(en[0]), 5):
            ax.quiver(dirx[0][i], dirx[1][i], dirx[2][i], et[0][i], et[1][i], et[2][i], color=['r'])
        for i in range(0, len(en[0]), 5):
            ax.quiver(dirx[0][i], dirx[1][i], dirx[2][i], en[0][i], en[1][i], en[2][i], color=['y'])
        for i in range(0, len(eb[0]), 5):
            ax.quiver(dirx[0][i], dirx[1][i], dirx[2][i], eb[0][i], eb[1][i], eb[2][i], color=['b'])

    plt.show()


def main():
    """Driver for testing algorithm with precreated demonstration data"""
    # Set the number of data points (time steps), the number of demonstrations,
    # and a ballpark for the desired number of circles of the canal surface to be
    # visible (wont be exact due to data inconsistencies)
    time_step_ct = 1000
    num_demonstrations = 5
    num_circles = 20

    # How much of the curve is covered from either end until it intersects a plane multiple times
    window_ratio = 0.1 # <----- ~~~~~~ MUST BE TUNED ACCORDING TO DATA ~~~~~
    window_size = int(time_step_ct * window_ratio)

    # Set this tolerance to scale with the size of the data (for how close
    # points have to be to be considered close enough the the plane defined by tangents)
    tolerance = 0.005 # <----- ~~~~~~ MUST BE TUNED ACCORDING TO DATA ~~~~~~

    # Create synthetic demonstration data 
    raw_data = get_demonstration_data(num_demonstrations, 200)

    # Resample data to get new demonstration data
    demonstration_data = sample_raw_data(raw_data, time_step_ct)

    # Calculate the mean/directrix curve from trajectories
    directrix = get_mean(demonstration_data)

    # Shift the origin to be the object at the end of the directrix
    new_origin = directrix[:, -1]
    demonstration_data = reframe_curves(demonstration_data, new_origin)
    directrix = reframe_curves([directrix], new_origin)[0]

    # Get the TNB frames at each point on the directrix
    t, n, b = get_tnb(directrix)

    # Slice the data at the ends so that every demonstration intersects
    # the plane defined by the tangent at every directrix point
    directrix, t, n, b = slice_data(demonstration_data, directrix, t, n, b, time_step_ct, tolerance, window_size)

    # Generate the canal surface
    radii = get_canal_surface(directrix, demonstration_data, t, tolerance)

    # Generate a random starting point for reproduction
    p0 = get_p0(0, directrix, radii, n, b)

    # Reproduction phase
    result = get_rep_traj(p0, directrix, radii, t, n, b)

    # Plot canal surface and results from collected data
    plot_canal_surface_and_result(demonstration_data, directrix, radii, result, t, n, b, num_circles)


if __name__ == '__main__':
    main()
