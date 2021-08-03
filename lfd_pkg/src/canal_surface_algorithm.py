#!/usr/bin/env python
"""
Author: Nicholas Pulsone

Canal Surface Generator, and Trajectory Reproduction based on Algorithms 1 and 2 in
"Encoding Demonstrations and Learning New Trajectories using Canal Surfaces"
by S. Reza Ahmadzadeh and Sonia Chernova (2016)

The following program will generate a canal surface (a set of points on a directrix/mean curve
and a set of radii at each point) based on demonstration data. It will then use the generated
canal surface with an initial starting position to reproduce a trajectory according to the input.

IMPORTANT: For best results, the keyword argument parameters for the functions in main() must be tuned
           (most notably, window_size and smoothing_factor)

NOTE: All 3D Curves assumed to be in the form [array_of_xs, array_of_ys, array_of_zs], where
array[:, 4] refers to the cartesian coordinate of the fourth point on the demonstration.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy import interpolate
from scipy.io import loadmat
from scipy.signal import argrelextrema


def normalization(vector):
    """Normalizes a 3D vector"""
    if np.linalg.norm(vector) == 0.0:
        return np.array([0.0, 0.0, 0.0])
    return vector / np.linalg.norm(vector)


def get_synthetic_demonstration_data(num_dem, num_points_to_sample):
    """Creates a set of synthetic trajectories used to generate the Canal Surface"""
    trajectories = []
    type_of_data = "Hourglass Data"
    for i in range(num_dem):
        if type_of_data == "Wavy Data":
            z = np.linspace(0, 1, num_points_to_sample)
            y = np.array([np.sin(1.5 * z[i]) for i in range(num_points_to_sample)])
            y = np.add(y, 0.5 * np.random.rand())
            x = np.array([np.cos(1.5 * z[i]) for i in range(num_points_to_sample)])
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
            ref_x = np.linspace(-1, 1, num_points_to_sample)
            ref_y = -0.5 * np.square(ref_x)
            # Get x, y, and z based on reference x and y and the vector
            x = np.linspace(-1, 1, num_points_to_sample)
            y = np.array([vector[0] * ref_y[i] + 0.01 * np.random.randn() for i in range(num_points_to_sample)])
            z = np.array([vector[1] * ref_y[i] + 0.01 * np.random.randn() for i in range(num_points_to_sample)])
        elif type_of_data == "Noisy Sinewave Data":
            # Generate vector, normalize it
            vector = np.array([0.3 * np.random.rand(), 0.3 * np.random.rand()])
            vector = normalization(vector)
            # Define a reference x and y to form parabolic shape
            ref_x = np.linspace(0.0, 2 * np.pi, num_points_to_sample)
            ref_y = np.sin(ref_x)
            # Get x, y, and z based on reference x and y and the vector
            x = np.linspace(-1, 1, num_points_to_sample)
            y = np.array([vector[0] * ref_y[i] + 0.01 * np.random.randn() for i in range(num_points_to_sample)])
            z = np.array([vector[1] * ref_y[i] + 0.01 * np.random.randn() for i in range(num_points_to_sample)])
        else:
            """Noisy Arctan data"""
            # Generate vector, normalize it
            vector = np.array([np.random.rand(), np.random.rand()])
            vector = normalization(vector)
            # Define a reference x and y to form parabolic shape
            ref_x = np.linspace(0.0, 2 * np.pi, num_points_to_sample)
            ref_y = np.arctan(2 * ref_x)
            # Get x, y, and z based on reference x and y and the vector
            x = np.linspace(-1, 1, num_points_to_sample)
            y = np.array([vector[0] * ref_y[i] + 0.05 * np.random.randn() for i in range(num_points_to_sample)])
            z = np.array([vector[1] * ref_y[i] + 0.05 * np.random.randn() for i in range(num_points_to_sample)])

        trajectories.append([x, y, z])
    return np.array(trajectories)


def get_mean(data):
    """Calculates the directrix curve given a set of trajectories"""
    """NOTE: Assumes data in the form of demonstrations with the same 
             number of points"""
    # Init arrays for x, y, and z values of the directrix
    x_mean = np.mean(data[:, 0], 0)
    y_mean = np.mean(data[:, 1], 0)
    z_mean = np.mean(data[:, 2], 0)
    return np.array([x_mean, y_mean, z_mean])


def reframe_curves(raw_data, data, best_demo_index=0, reaching=True):
    """Make the given point the new origin, with respect to the given data.
       Data given in the form of an array of 3D curves."""
    # Define new origin based on best demonstration
    new_origin = copy.deepcopy(raw_data[best_demo_index][:, -1])
    if reaching:
        # Reframe demonstrations to make the best target point the origin
        for i in range(len(raw_data)):
            # Slightly shift raw demonstrations to meet at target point
            distance_to_shift_raw = np.array([new_origin[0] - raw_data[i][0, -1], new_origin[1] - raw_data[i][1, -1],
                                              new_origin[2] - raw_data[i][2, -1]])
            if i != best_demo_index:
                raw_data[i][0] += distance_to_shift_raw[0]
                raw_data[i][1] += distance_to_shift_raw[1]
                raw_data[i][2] += distance_to_shift_raw[2]
            # Slightly shift smooth demonstrations to meet at target point
            distance_to_shift_smooth = np.array([new_origin[0] - data[i][0, -1], new_origin[1] - data[i][1, -1],
                                                 new_origin[2] - data[i][2, -1]])
            data[i][0] += distance_to_shift_smooth[0]
            data[i][1] += distance_to_shift_smooth[1]
            data[i][2] += distance_to_shift_smooth[2]

    # Move to new origin
    for i in range(len(raw_data)):
        raw_data[i][0] -= new_origin[0]
        raw_data[i][1] -= new_origin[1]
        raw_data[i][2] -= new_origin[2]
        data[i][0] -= new_origin[0]
        data[i][1] -= new_origin[1]
        data[i][2] -= new_origin[2]

    return raw_data, data


def inverse_reframe_curves(raw_data, position_of_base_origin):
    """ Inverts the shift in a frame of reference of 3D curves,
        given the position of the old origin in the new frame of reference """
    # Define the current origin
    for i in range(len(raw_data)):
        raw_data[i][0] -= position_of_base_origin[0]
        raw_data[i][1] -= position_of_base_origin[1]
        raw_data[i][2] -= position_of_base_origin[2]
    return raw_data


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
    en[:, 0] = normalization(en[:, 0])

    # Propogate N and B along directrix based on first Normal

    for i in range(len(et[0])):
        eb[:, i] = normalization(np.cross(en[:, i], et[:, i]))
        en[:, i + 1] = normalization(np.cross(et[:, i], eb[:, i]))

    # Normailze Normal and Binormal
    en = np.apply_along_axis(normalization, axis=0, arr=en)
    eb = np.apply_along_axis(normalization, axis=0, arr=eb)

    return et, en[:, 0:(len(en[0]) - 1)], eb


def get_distance(p1, p2):
    """Calculates the distance between 2 points in 3 dimensional space"""
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]) + np.square(p1[2] - p2[2]))


def define_plane(p, t):
    """Calculates the offset (d) parameter of a plane given a point
    on the directrix (p) and the tangent vector (t)"""
    return -1.0 * ((t[0] * p[0]) + (t[1] * p[1]) + (t[2] * p[2]))


def project_point_to_plane(p, t, other_p):
    """Projects a point (other p) to a plane defined by an origin point (p)
    and the tangent vector at that point (t). Returns projected point."""
    # Define a vector between the points
    v = np.array([other_p[0] - p[0], other_p[1] - p[1], other_p[2] - p[2]])
    # Get distance to subtract from other p
    dist = v[0] * t[0] + v[1] * t[1] + v[2] * t[2]
    dists_to_subtract = dist * t
    # Subtract distance from other_p to get projected point
    return np.subtract(other_p, dists_to_subtract)


def slice_and_sample_demos(raw_demos, num_points_to_sample=500, spline_degree=5, smoothing_factor=0.001, start_cut="auto", end_cut=50):
    """Multi-step procedure to process raw demonstrations for calculation.
       Takes a list of raw demonstrations, each being a list of x y z of a certain length
       based on how the demonstrations were recorded.
       Returns an array of smoothed demonstrations."""
    # Define number of raw demonstrations
    num_demos = len(raw_demos)

    """ Step 1) Remove duplicate points in the demos"""
    # Init New arrays for trimmed x, y, and z
    trim_raw_demos = []
    for raw_demo in raw_demos:
        # Define a number of points
        num_points = len(raw_demo[0])

        # Init arrays for new x, y, and z
        trim_x = [raw_demo[0][0]]
        trim_y = [raw_demo[1][0]]
        trim_z = [raw_demo[2][0]]

        # Walk through the list of points and ignore adjacent points that are practically identical
        for i in range(num_points - 1):
            if not (np.absolute(raw_demo[0][i] - raw_demo[0][i + 1]) < 0.0001 and np.absolute(
                    raw_demo[1][i] - raw_demo[1][i + 1]) < 0.0001 and np.absolute(
                    raw_demo[2][i] - raw_demo[2][i + 1]) < 0.0001):
                trim_x.append(raw_demo[0][i + 1])
                trim_y.append(raw_demo[1][i + 1])
                trim_z.append(raw_demo[2][i + 1])
        trim_raw_demos.append(np.array([trim_x, trim_y, trim_z]))

    """Step 2) Slice the ends of the raw demonstrations"""
    # pts_per_demo = np.empty(num_demos, dtype=int)
    smoothed_sliced_demos = []
    for i in range(num_demos):
        # For each demonstration, define a slope/tangent function
        t = np.apply_along_axis(np.gradient, axis=1, arr=trim_raw_demos[i])
        t = np.apply_along_axis(normalization, axis=0, arr=t)

        # Slice the beggining of the current demonstrations, to make it level with all other demos
        if start_cut == "auto":
            start_i = 0
            for j in range(num_demos):
                if j != i:
                    start_plane_defined = False
                    while not start_plane_defined:
                        start_plane_defined = True
                        # Define the number of points of the current demo
                        num_points = len(trim_raw_demos[j][0])
                        # Define the current plane itself
                        d = define_plane(trim_raw_demos[i][:, start_i], t[:, start_i])
                        # Get indices of intersection with the plane
                        plane_intersections = argrelextrema(np.absolute(
                            t[0, start_i] * trim_raw_demos[j][0] + t[1, start_i] *
                            trim_raw_demos[j][1] + t[2, start_i] *
                            trim_raw_demos[j][2] + d), np.less)[0]
                        # First intersection within the first chunk of demonstration tells us that the two compared demonstrations
                        # are level enough
                        if len(plane_intersections) < 1 or plane_intersections[0] not in range(0, int(0.2 * num_points)):
                            start_i += 1
                            start_plane_defined = False
        else:
            start_i = start_cut

        # Defined the newly sliced demo to be smoothed, based on calculated start and given end cut parameter
        sliced_demo = trim_raw_demos[i][:, start_i:-end_cut]

        """ Step 3) Smooth the demonstrations"""
        # Define a time range based on the current number of raw points
        num_captured_points = len(sliced_demo[0])
        time_range = np.linspace(0.0, 1.0, num_captured_points)
        t_function = np.linspace(0.0, 1.0, num_points_to_sample)

        # Gerneate 1D interpolation functions for the x, y, and z of each trajectory
        x_est_func = interpolate.UnivariateSpline(time_range, sliced_demo[0], k=spline_degree, s=smoothing_factor)
        y_est_func = interpolate.UnivariateSpline(time_range, sliced_demo[1], k=spline_degree, s=smoothing_factor)
        z_est_func = interpolate.UnivariateSpline(time_range, sliced_demo[2], k=spline_degree, s=smoothing_factor)
        
        # Compute sampled arrays for x y z
        x = np.array([x_est_func(t) for t in t_function])
        y = np.array([y_est_func(t) for t in t_function])
        z = np.array([z_est_func(t) for t in t_function])

        """ Evenly spaces points in cartesian space. Can provide more accurate results with less points, though not neccessary. """
        """
        # Compute arc length of curve by sampling at high rate (num points in t_function)
        x_arc = np.array([np.square(x_est_func(t_function[t]) - x_est_func(t_function[t - 1])) for t in range(1, len(t_function))])
        y_arc = np.array([np.square(y_est_func(t_function[t]) - y_est_func(t_function[t - 1])) for t in range(1, len(t_function))])
        z_arc = np.array([np.square(z_est_func(t_function[t]) - z_est_func(t_function[t - 1])) for t in range(1, len(t_function))])
        arc_length = np.sum(np.sqrt(x_arc + y_arc + z_arc))

        # Determine distance between evenly spaced points based on arc length
        approx_delta_dist = arc_length / 400.0

        # Evenly space the points (derived from https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values/19118984#19118984)
        j, idx = 0, [0]
        while j + 1 < len(x_arc):
            total_dist = 0
            for k in range(j+1, len(x_arc)):
                total_dist += np.sqrt(x_arc[k] + y_arc[k] + z_arc[k])
                if total_dist > approx_delta_dist:
                    idx.append(k)
                    break
            j = k+1
        pts_per_demo[i] = len(idx)
        
        # Create arrays for evenly spaced points
        x_f = x[idx]
        y_f = y[idx]
        z_f = z[idx]
        smoothed_sliced_demos.append(np.array([x_f, y_f, z_f]))
        """

        # Add smooth, sliced demonstration to the list to be returned
        smoothed_sliced_demos.append(np.array([x, y, z]))

    """ For evenly spaced algorithm to ensure equal number of points across demonstrations """
    """
    # Slice a couple points in the begginging of some demonstrations to ensure that
    # There is the same number of points <--- TO BE CHANGED
    for i in range(num_demos):
        if len(smoothed_sliced_demos[i][0]) > min(pts_per_demo):
            while len(smoothed_sliced_demos[i][0]) != min(pts_per_demo):
                smoothed_sliced_demos[i] = np.delete(smoothed_sliced_demos[i], 0, axis=1)
    """
    return np.array(smoothed_sliced_demos)


def get_canal_surface(dirx, data, et):
    """Uses a defined window size to construct a canal surface from a set of smooothed
       demonstrations (data), a directrix/spine curve (dirx), and the tangent vectors at
       each directrix point (et). Returns an array of radii, one for each directrix point.
       dirx = mean curve
       data = smoothed demonstrations
       et = array of tangent vectors"""

    # Define number of demonstrations
    num_demo = len(data)
    # Define the number of points and a corresponding window size
    num_points = len(dirx[0])
    window_size = int(num_points / 3.0)
    # Init array container for radii of the canal surface
    radii = np.empty(num_points)

    # Loop over points in the directrix
    for i in range(num_points):
        # Define the directrix point at each step
        pt = dirx[:, i]

        # Define the window at the current step
        increment = int(max(0, i - math.floor(window_size / 2)))
        window = range(increment, (window_size + increment))
        if max(window) >= num_points:
            window = range((num_points - window_size), num_points)

        # Define the plane at the current step
        d = define_plane(pt, et[:, i])

        # Loop over each demonstration
        possible_radii = np.zeros(len(data), dtype=float)
        for j in range(num_demo):
            # Find the closest point on the demonstration to the current plane within the window
            windowed_points = data[j, :, min(window):max(window) + 1]
            windowed_comparisons = np.absolute(
                et[0][i] * windowed_points[0] + et[1][i] * windowed_points[1] + et[2][i] * windowed_points[2] + d)
            effective_point = windowed_points[:, np.where(windowed_comparisons == min(windowed_comparisons))[0][0]]
            projected_effective_point = project_point_to_plane(pt, et[:, i], effective_point)

            # Calculate distance to that point to get the possisble radius for that demonstration
            possible_radii[j] = get_distance(projected_effective_point, pt)

        # Find maximum distance of relevant points, use as radius
        radii[i] = np.amax(possible_radii)
        # Print which demonstration is determining the radius at each time step
        # print(radii[i], np.where(possible_radii == np.amax(possible_radii))[0][0])
    return radii


def smooth_radii(ri):
    """ Will smooth the radii of the canal surface cross sections over time
        ri = raw radii calculated from the canal surface algorithm
        returns smooth radii array with the same length as ri
    """
    # Interpolate based on every tenth value
    radii = ri[::10]
    smooth_radii_func = interpolate.UnivariateSpline(range(len(radii)), radii, k=3, s=0.00001)
    # Construct the array with the interpolating function
    smooth_radii = np.array([smooth_radii_func(t) for t in np.linspace(0, len(radii), len(ri))])
    # Create a threshold to increase the radii by slightly to account for places
    # where the some of the circles may have dipped below one of the demonstrations
    threshold = 0.05 * np.mean([max(ri), min(ri)])
    return smooth_radii + threshold


def get_p0(i, dirx, ri, en, eb):
    """Will generate a random point on the a cross section of the canal surface
       i = index of the cross section where the point in to be defined
       dirx = 3D mean curve
       ri = radii of the canal surface
       en, eb = Normal and Binormal vectors (see get_tnb)"""
    # Define a random magnitude and angle
    angle = np.random.uniform(0.0, (2.0 * np.pi))
    angle = np.pi / 4.0
    mag = np.random.uniform(0.0, ri[i])
    mag = ri[i]

    # Determine point's x, y, and z based on angle, magnitude,
    # and the normal and binormal vectors at the starting directrix point
    x = dirx[0][i] + mag * (en[0][i] * np.cos(angle) + eb[0][i] * np.sin(angle))
    y = dirx[1][i] + mag * (en[1][i] * np.cos(angle) + eb[1][i] * np.sin(angle))
    z = dirx[2][i] + mag * (en[2][i] * np.cos(angle) + eb[2][i] * np.sin(angle))
    return np.array([x, y, z])


def get_rep_traj(p0, idx, dirx, ri, et, en, eb):
    """Create a reproduced trajectory from an initial starting point,
       p0, using the generated canal surface.
       p0 = initial position of the robot [x, y, z]
       dirx = mean curve
       ri = radii of the canal surface
       et, en, eb = arrays of tangent, normal, and binormal vectors respectively"""

    # Redefine given directrix, radii, and TNB frames based on given reproduction range
    dirx = dirx[:, idx[0]:-idx[1]]
    et = et[:, idx[0]:-idx[1]]
    en = en[:, idx[0]:-idx[1]]
    eb = eb[:, idx[0]:-idx[1]]
    ri = ri[idx[0]:-idx[1]]

    # Determine closest plane to the starting point (withing first section of the canal surface)
    distances_to_each_plane = np.array([np.absolute(
        define_plane(np.array([dirx[0][j], dirx[1][j], dirx[2][j]]), np.array([et[0][j], et[1][j], et[2][j]])) +
        et[0][j] * p0[0] + et[1][j] * p0[1] + et[2][j] * p0[2]) for j in range(int(len(dirx[0])/3.0))])
    p0_index = np.where(distances_to_each_plane == np.amin(distances_to_each_plane))[0][0]

    # Project the point to that plane
    projected_p0 = project_point_to_plane(dirx[:, p0_index], et[:, p0_index], p0)

    # Get the length between directrix and starting point, used for calculating the ratio
    p0c0 = get_distance(projected_p0, np.array([dirx[0][p0_index], dirx[1][p0_index], dirx[2][p0_index]]))

    # Determine ratio
    if p0c0 > ri[p0_index]:
        ratio = 1.0
    else:
        ratio = p0c0 / ri[p0_index]

    # Initialize a container for the reproduced trajectory (projjjjjj)
    reproduced_trajectory = np.empty((3, (len(ri) - p0_index)), dtype=float)
    reproduced_trajectory[0][0] = projected_p0[0]
    reproduced_trajectory[1][0] = projected_p0[1]
    reproduced_trajectory[2][0] = projected_p0[2]

    # Center point vector at the origin
    projected_p0 = np.array([projected_p0[0] - dirx[0][p0_index], projected_p0[1] - dirx[1][p0_index], projected_p0[2] - dirx[2][p0_index]])

    # Define TNB frame at initial cross section
    p0_frame = np.rot90(np.fliplr(np.array([et[:, p0_index],
                                            en[:, p0_index],
                                            eb[:, p0_index]])))

    # Translate the point across the canal surface, once for each point on the directrix
    i = p0_index + 1
    rep_traj_index = 1
    while i < len(ri):
        # Define TNB frame at current cross section
        curr_frame = np.rot90(np.fliplr(np.array([et[:, i],
                                                  en[:, i],
                                                  eb[:, i]])))

        # Determine transformation matrix between p0's TNB frame and the current cross section's TNB frame
        T = np.array([[(np.dot(p0_frame[0], curr_frame[0])), (np.dot(p0_frame[1], curr_frame[0])),
                       (np.dot(p0_frame[2], curr_frame[0]))],
                      [(np.dot(p0_frame[0], curr_frame[1])), (np.dot(p0_frame[1], curr_frame[1])),
                       (np.dot(p0_frame[2], curr_frame[1]))],
                      [(np.dot(p0_frame[0], curr_frame[2])), (np.dot(p0_frame[1], curr_frame[2])),
                       (np.dot(p0_frame[2], curr_frame[2]))]])

        # Perform the transformation to get the new translated point
        new_pt = np.matmul(T, projected_p0)

        # Scale the point in accordance with the ratio rule
        new_pt = (ratio * ri[i] * new_pt) / np.linalg.norm(new_pt)

        # Add to new point to the reproduced trajectory
        reproduced_trajectory[0][rep_traj_index] = dirx[0][i] + new_pt[0]
        reproduced_trajectory[1][rep_traj_index] = dirx[1][i] + new_pt[1]
        reproduced_trajectory[2][rep_traj_index] = dirx[2][i] + new_pt[2]

        # Move to the next point
        i += 1
        rep_traj_index += 1

    return reproduced_trajectory


def plot_canal_surface_and_result(raw_data, data, dirx, ri, reproduced_traj, et, en, eb, num_circ=20):
    """Plot the demonstration data, canal surface, and reproduction all at once.
       Used almost solely for debugging, as in real cases data is shown to the user periodically.
       data = array of 3D curves
       dirx = 3D mean curve
       ri = radii of the canal surface
       reproduced_traj = reproduced trajectory using canal surface
       et, en, eb = arrays of tangent, normal, and binormal vectors respectively (see get_tnb)
       num_circ = number of cricles to show on the plot (not exact since data can have varying lengths)"""

    # Choose what will be shown in the plot
    do_plot_raw = False
    do_plot_demonstrations = True
    do_plot_directrix = True
    do_plot_circles = True
    do_plot_reproduction = True
    do_plot_TNB = False
    do_plot_point_markers = False
    do_plot_radii_v_time = False
    do_plot_raw_demos_axis_wise = False

    # Set up 3D plot
    ax = plt.axes(projection='3d')

    # Plot raw demonstrations
    if do_plot_raw:
        for raw_dem in raw_data:
            ax.plot3D(raw_dem[0], raw_dem[1], raw_dem[2], 'orange')

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

    # Plot starting point and reproduced trajectory
    if do_plot_reproduction:
        # Plot initial point
        ax.plot3D([reproduced_traj[0][0]], [reproduced_traj[1][0]], [reproduced_traj[2][0]], marker='*', markersize=15)
        # Plot Reproduced trajectory line
        ax.plot3D(reproduced_traj[0], reproduced_traj[1], reproduced_traj[2], 'lime')

    # Plot point markers (for debugging reproduced curve)
    if do_plot_point_markers:
        for i in range(len(data)):
            for j in range(len(data[0][0])):
                ax.plot3D([data[i][0][j]], [data[i][1][j]], [data[i][2][j]], marker="*", markersize=5)

    # Plot T, N, B parameters (for various debugging)
    if do_plot_TNB:
        # Change factor based on size of data
        et *= 0.02
        en *= 0.02
        eb *= 0.02
        for i in range(0, len(en[0]), 5):
            ax.quiver(dirx[0][i], dirx[1][i], dirx[2][i], et[0][i], et[1][i], et[2][i], color=['r'])
        for i in range(0, len(en[0]), 5):
            ax.quiver(dirx[0][i], dirx[1][i], dirx[2][i], en[0][i], en[1][i], en[2][i], color=['y'])
        for i in range(0, len(eb[0]), 5):
            ax.quiver(dirx[0][i], dirx[1][i], dirx[2][i], eb[0][i], eb[1][i], eb[2][i], color=['b'])

    # Set plot limits for 3D plot, make equal, then plot
    xl, xr = plt.xlim()
    yl, yr = plt.ylim()
    zb, zt = ax.get_zlim()
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

    plt.show()

    """ 2D/Debugging Plots """
    # Plot radius as a function of time
    if do_plot_circles and do_plot_radii_v_time:
        ax2d2 = plt.figure(2)
        ax2d = plt.axes()
        plt.title('Radii over time', size=16)
        ax2d.plot(range(len(et[0])), ri)
        plt.show()

    # Plot the x, y, and z components of the raw demonstrations to check for smoothness
    if do_plot_raw_demos_axis_wise:
        fig, axs = plt.subplots(3, len(raw_data))
        for i in range(len(raw_data)):
            axs[0, i].plot(np.linspace(0.0, (len(raw_data[i][0]) - 1), len(data[i][0])), data[i][0])
            axs[0, i].plot(raw_data[i][0], "red")
            axs[0, i].set_title("X for Demo #" + str(i+1))
            axs[1, i].plot(np.linspace(0.0, (len(raw_data[i][0]) - 1), len(data[i][0])), data[i][1])
            axs[1, i].plot(raw_data[i][1], "red")
            axs[1, i].set_title("Y for Demo #" + str(i + 1))
            axs[2, i].plot(np.linspace(0.0, (len(raw_data[i][0]) - 1), len(data[i][0])), data[i][2])
            axs[2, i].plot(raw_data[i][2], "red")
            axs[2, i].set_title("Z for Demo #" + str(i + 1))
        plt.show()


def main():
    """Driver for testing algorithm with preloaded demonstration data"""
    # Keep as false since I have modified the mat files used for debugging
    do_get_file_data = True
    file_data_type = "Writing"

    # Set the number of data points (time steps), the number of demonstrations,
    # and a ballpark for the desired number of circles of the canal surface to be
    # visible (wont be exact due to data inconsistencies)
    num_demonstrations = 5

    # Load demonstration data either from files or synthetically
    if do_get_file_data:
        raw_data = []
        if file_data_type == "Reaching":
            for i in range(1, 11):
                mat = loadmat("C:\\Users\\psych\\Downloads\\reach_" + str(i) + ".mat")
                raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))
        elif file_data_type == "Smooth Reaching":
            for i in range(1, 11):
                mat = loadmat("C:\\Users\\psych\\OneDrive\\Documents\\MATLAB\\sr" + str(i) + ".mat")
                raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))
        elif file_data_type == "Writing":
            for i in range(1, 11):
                mat = loadmat("C:\\Users\\psych\\OneDrive\\Documents\\MATLAB\\write_raw_smooth" + str(i) + ".mat")
                raw_data.append(np.swapaxes(np.array(mat["raw"]), 0, 1))
        elif file_data_type == "Smooth Writing":
            for i in range(1, 11):
                mat = loadmat("C:\\Users\\psych\\OneDrive\\Documents\\MATLAB\\write_raw_smooth" + str(i) + ".mat")
                raw_data.append(np.swapaxes(np.array(mat["smooth"]), 0, 1))
        else:
            """Pushing Data"""
            mat = loadmat("C:\\Users\\psych\\Downloads\\push_1.mat")
            raw_data.append(np.swapaxes(np.array(mat["push1"]), 0, 1))

            mat = loadmat("C:\\Users\\psych\\Downloads\\push_2.mat")
            raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))

            mat = loadmat("C:\\Users\\psych\\Downloads\\push_3.mat")
            raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))

            mat = loadmat("C:\\Users\\psych\\Downloads\\push_4.mat")
            raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))

            mat = loadmat("C:\\Users\\psych\\Downloads\\push_5.mat")
            raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))

            mat = loadmat("C:\\Users\\psych\\Downloads\\push_6.mat")
            raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))

            mat = loadmat("C:\\Users\\psych\\Downloads\\push_7.mat")
            raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))

            mat = loadmat("C:\\Users\\psych\\Downloads\\push_8.mat")
            raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))

            mat = loadmat("C:\\Users\\psych\\Downloads\\push_9.mat")
            raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))

            mat = loadmat("C:\\Users\\psych\\Downloads\\push_10.mat")
            raw_data.append(np.swapaxes(np.array(mat["test"]), 0, 1))
            raw_data = np.array(raw_data)
    else:
        raw_data = get_synthetic_demonstration_data(num_demonstrations, 250)

    # Resample data to get new demonstration data
    if do_get_file_data and (file_data_type == "Smooth Reaching" or file_data_type == "Smooth Writing"):
        demonstration_data = np.array(raw_data)
    else:
        demonstration_data = slice_and_sample_demos(raw_data, num_points_to_sample=500, spline_degree=5, smoothing_factor=0.025, start_cut="auto", end_cut=30)

    # Shift the origin to be the object at the end of the directrix
    raw_data, demonstration_data = reframe_curves(raw_data, demonstration_data, best_demo_index=0, reaching=False)

    # Calculate the mean/directrix curve from trajectories
    directrix = get_mean(demonstration_data)

    # Get the TNB frames at each point on the directrix
    t, n, b = get_tnb(directrix)

    # Generate the canal surface
    radii = get_canal_surface(directrix, demonstration_data, t, window_size=200)

    # Generate a random starting point for reproduction
    p0 = get_p0(0, directrix, radii, n, b)

    # Reproduction phase
    result = get_rep_traj(p0, directrix, radii, t, n, b)

    # Plot canal surface and results from collected data
    plot_canal_surface_and_result(raw_data, demonstration_data, directrix, radii, result, t, n, b)


if __name__ == '__main__':
    main()
