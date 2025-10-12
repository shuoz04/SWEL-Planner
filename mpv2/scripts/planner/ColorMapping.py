#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2021/12/21 下午1:00
"""
from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>

# Most of the content of this file is copied from
# https://bsou.io/posts/color-gradients-with-python

from numpy import random as rnd
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import rospy


def convert_numpy_2_pointcloud2_color(points, ccc, stamp=None, frame_id=None, maxDistColor=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.

    This function will automatically assign RGB values to each point. The RGB values are
    determined by the distance of a point from the origin. Use maxDistColor to set the distance
    at which the color corresponds to the farthest distance is used.

    points: A NumPy array of Nx3.
    stamp: An alternative ROS header stamp.
    frame_id: The frame id. String.
    maxDisColor: Should be positive if specified..

    This function get inspired by
    https://github.com/spillai/pybot/blob/master/pybot/externals/ros/pointclouds.py
    https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
    (https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/)
    and expo_utility.xyz_array_to_point_cloud_msg() function of the AirSim package.

    ROS sensor_msgs/PointField.
    http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html

    More references on mixed-type NumPy array, structured array.
    https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    https://stackoverflow.com/questions/37791134/merge-width-x-height-x-3-numpy-uint8-array-into-width-x-height-x-1-uint32-array
    https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html
    '''
    DIST_COLORS = [
        "#2980b9",
        "#27ae60",
        "#f39c12",
        "#c0392b",
    ]

    DIST_COLOR_LEVELS = 20

    # Clipping input.
    dist = np.linalg.norm(points, axis=1)
    if maxDistColor is not None and maxDistColor > 0:
        dist = np.clip(dist, 0, maxDistColor)

    # Compose color.
    # cr, cg, cb = color_map(dist, DIST_COLORS, DIST_COLOR_LEVELS)

    C = np.zeros((len(points), 4), dtype=np.uint8) + 255

    # C[:, 0] = cb.astype(np.uint8)
    # C[:, 1] = cg.astype(np.uint8)
    # C[:, 2] = cr.astype(np.uint8)
    C[:, 0] = (ccc[:, 2]*255).astype(np.uint8)
    C[:, 1] = (ccc[:, 1]*255).astype(np.uint8)
    C[:, 2] = (ccc[:, 0]*255).astype(np.uint8)


    C = C.view("uint32")

    # Structured array.
    pointsColor = np.zeros((points.shape[0], 1),
                           dtype={
                               "names": ("x", "y", "z", "rgba"),
                               "formats": ("f4", "f4", "f4", "u4")})

    points = points.astype(np.float32)

    pointsColor["x"] = points[:, 0].reshape((-1, 1))
    pointsColor["y"] = points[:, 1].reshape((-1, 1))
    pointsColor["z"] = points[:, 2].reshape((-1, 1))
    pointsColor["rgba"] = C

    header = Header()

    if stamp is None:
        header.stamp = rospy.Time().now()
    else:
        header.stamp = stamp

    if frame_id is None:
        header.frame_id = "None"
    else:
        header.frame_id = frame_id

    msg = PointCloud2()
    msg.header = header

    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = points.shape[0]

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]

    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = pointsColor.tobytes()

    return msg


def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i + 2], 16) for i in range(1, 6, 2)]


def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#" + "".join(["0{0:x}".format(v) if v < 16 else
                          "{0:x}".format(v) for v in RGB])


def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
    return {"hex": [RGB_to_hex(RGB) for RGB in gradient],
            "r": [RGB[0] for RGB in gradient],
            "g": [RGB[1] for RGB in gradient],
            "b": [RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j]))
            for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)


def rand_hex_color(num=1):
    ''' Generate random hex colors, default is one,
        returning a string. If num is greater than
        1, an array of strings is returned. '''
    colors = [
        RGB_to_hex([x * 255 for x in rnd.rand(3)])
        for i in range(num)
    ]
    if num == 1:
        return colors[0]
    else:
        return colors


def polylinear_gradient(colors, n):
    ''' returns a list of colors forming linear gradients between
        all sequential pairs of colors. "n" specifies the total
        number of desired output colors '''
    # The number of colors per individual linear gradient
    n_out = int(float(n) / (len(colors) - 1))
    # returns dictionary defined by color_dict()
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)

    if len(colors) > 1:
        for col in range(1, len(colors) - 1):
            next = linear_gradient(colors[col], colors[col + 1], n_out)
            for k in ("hex", "r", "g", "b"):
                # Exclude first point to avoid duplicates
                gradient_dict[k] += next[k][1:]

    return gradient_dict


def color_map(data, colors, nLevels):
    # Get the color gradient dict.
    gradientDict = polylinear_gradient(colors, nLevels)

    # Get the actual levels generated.
    n = len(gradientDict["hex"])

    # Level step.
    dMin = data.min()
    dMax = data.max()
    step = (dMax - dMin) / (n - 1)

    stepIdx = (data - dMin) / step
    stepIdx = stepIdx.astype(np.int32)

    rArray = np.array(gradientDict["r"])
    gArray = np.array(gradientDict["g"])
    bArray = np.array(gradientDict["b"])

    r = rArray[stepIdx]
    g = gArray[stepIdx]
    b = bArray[stepIdx]

    return r, g, b


if __name__ == "__main__":
    colors = [
        "#2980b9",
        "#27ae60",
        "#f39c12",
        "#c0392b",
    ]

    data = np.linspace(0, 99, 100)

    r, g, b = color_map(data, colors, 20)
