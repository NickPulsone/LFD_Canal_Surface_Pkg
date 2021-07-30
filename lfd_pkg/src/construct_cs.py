#!/usr/bin/env python

import sys
import copy
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import numpy as np
from lfd_canal_surface_pkg.srv import CanalSrv, CanalSrvResponse
from canal_surface_algorithm import get_mean, get_tnb, get_canal_surface

def cs_callback(req):
    # Intialize a custom canal surface resonse variable
    canal_response = CanalSrvResponse()

    # Get the demonstrations from the service request
    data = []
    num_points = int((req.demonstrations[0]).layout.dim[0].stride)
    for demonstration in req.demonstrations:
        data.append(np.array([demonstration.data[0:num_points], demonstration.data[num_points:2*num_points], demonstration.data[2*num_points:]]))
    data = np.array(data)

    # Get the directrix based on the demonstrations
    dirx = get_mean(data)

    # Get the TNB frames of the directrix
    t, n, b = get_tnb(dirx)

    # Calculate the radii of the canal surface
    radii = get_canal_surface(dirx, data, t)

    # Assign radii to response
    canal_response.radii = radii

    # Assign new sliced directrix to repsonse
    canal_response.directrix = Float64MultiArray()
    canal_response.directrix.data = np.frombuffer(dirx.tobytes(),'float64')
    canal_response.directrix.layout.dim = [MultiArrayDimension()]
    canal_response.directrix.layout.dim[0].stride = num_points

    # Assign tangent component of TNB frame arrays to response
    canal_response.tangent = Float64MultiArray()
    canal_response.tangent.data = np.frombuffer(t.tobytes(),'float64')
    canal_response.tangent.layout.dim = [MultiArrayDimension()]
    canal_response.tangent.layout.dim[0].stride = num_points

    # Assign normal component of TNB frame arrays to response
    canal_response.normal = Float64MultiArray()
    canal_response.normal.data = np.frombuffer(n.tobytes(),'float64')
    canal_response.normal.layout.dim = [MultiArrayDimension()]
    canal_response.normal.layout.dim[0].stride = num_points
    
    # Assign binormal component of TNB frame arrays to response
    canal_response.binormal = Float64MultiArray()
    canal_response.binormal.data = np.frombuffer(b.tobytes(),'float64')
    canal_response.binormal.layout.dim = [MultiArrayDimension()]
    canal_response.binormal.layout.dim[0].stride = num_points

    return canal_response


def server_controller():
    # Setup ros node
    rospy.init_node('cs_constructor', anonymous=True)

    # Launch server that will recieve user requests to generate canal surfaces
    construct_cs_server = rospy.Service("/canal_surface", CanalSrv, cs_callback)
    
    rospy.spin()


if __name__ == '__main__':
    try:
        server_controller()
    except rospy.ROSInterruptException:
        pass
    
