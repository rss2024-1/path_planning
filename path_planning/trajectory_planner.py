import rclpy
from rclpy.node import Node
import numpy as np
import tf_transformations as tf

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, PoseArray, Pose, Quaternion, PoseStamped
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

import numpy as np
from typing import Sequence, Tuple, List, Optional, Union

import collections
from abc import abstractmethod
import dataclasses
import math
import random
import tf_transformations as tf


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map = np.array([[]])
        self.map_resolution = 0
        self.map_origin_orientation = 0
        self.map_origin_poistion = 0
        self.start_point = None
        self.end_point = None

    def map_cb(self, msg):
        self.get_logger().info("map callback")
        self.map = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.map_origin_orientation = msg.info.origin.orientation
        self.map_origin_poistion = msg.info.origin.position
        #self.get_logger().info("map: {}".format(' '.join(map(str, self.map))))
        self.plan_path(self.start_point, self.end_point, self.map)

    
    # def pose_to_xyth(self, pose): 
    #     #
    #     # Extract position
    #     # try: 
    #     #     position = pose.pose.pose.position
    #     #     orientation = pose.pose.pose.orientation
    #     # except: 
    #     #     print('except!!')
    #     #     position = pose.pose.position
    #     #     orientation = pose.pose.orientation
    #     position = pose.pose.position
    #     orientation = pose.pose.orientation

    #     x = position.x
    #     y = position.y
    #     # self.get_logger().info(f"x y is {x, y}")
    #     # odom_euler = tf.euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))
    #     # self.get_logger().info(f"odom_quat is: {odom_euler}")
    #     # th = odom_euler[2] # maybe negative of this?? idk??
    #     q = [orientation.x,
    #         orientation.y,
    #         orientation.z,
    #         orientation.w]
    #     rotation_matrix = tf.quaternion_matrix(q)[:3, :3]
    #     metric_coordinates = np.dot(rotation_matrix, np.array([x, y, 0]))
        
    #     # Apply the translation
    #     x_real = metric_coordinates[0] - position.x
    #     y_real = metric_coordinates[1] - position.y
    #     #(-10, -1)

    #     return (x_real, y_real)
    
    def pose_to_xyth(self, pose): 
        position = pose.pose.position
        orientation = pose.pose.orientation
        x = position.x
        y = position.y
        return (x, y)
    
    # def pose_to_xyth(self, pose): 
    #     position = pose.pose.position
    #     orientation = pose.pose.orientation
    
    # def pixel_to_xy(self, coord, msg): 
    #     # (msg.info.height, msg.info.width)
    #     u, v = coord[0], coord[1]
    #     resolution = msg.info.resolution
    #     x= u*resolution
    #     y= v*resolution
    #     position = msg.info.origin.position
    #     orientation = msg.info.origin.orientation

    #     q = [orientation.x,
    #         orientation.y,
    #         orientation.z,
    #         orientation.w]
    #     rotation_matrix = tf.quaternion_matrix(q)[:3, :3]
    #     metric_coordinates = np.dot(rotation_matrix, np.array([x, y, 0]))
        
    #     # Apply the translation
    #     x_real = metric_coordinates[0] + position.x
    #     y_real = metric_coordinates[1] + position.y

    #     return (x_real, y_real)

    def pixel_to_xy(self, coord, msg): 
        u, v = coord[0], coord[1]
        resolution = msg.info.resolution
        position = msg.info.origin.position
        orientation = msg.info.origin.orientation
        x= u*resolution
        y= v*resolution

        q = [orientation.x,
            orientation.y,
            orientation.z,
            orientation.w]
        roll, pitch, yaw = tf.euler_from_quaternion(q)

        # Apply the translation
        x = x*np.cos(yaw) - y*np.sin(yaw)
        y = x*np.sin(yaw) + y*np.cos(yaw)
        x += position.x
        y += position.y
        return (x, y)
    

    # def pixel_to_xy(self, coord, msg):
    #     """ Convert pixel coordinates (u, v) to real-world coordinates (x, y). """
    #     u, v = coord
    #     # Extract information from msg
    #     resolution = msg.info.resolution
    #     position = [msg.info.origin.position.x, msg.info.origin.position.y, 0]
    #     orientation = [msg.info.origin.orientation.x, msg.info.origin.orientation.y, 
    #                 msg.info.origin.orientation.z, msg.info.origin.orientation.w]
        
    #     # Create transformation matrix from quaternion and position
    #     transformation_matrix = tf.quaternion_matrix(orientation)
    #     transformation_matrix[0:3, 3] = position[0:3]
        
    #     # Scale pixel coordinates by resolution
    #     pixel_coords = np.array([u * resolution, v * resolution, 0, 1])
        
    #     # Apply transformation
    #     real_coords = transformation_matrix @ pixel_coords
        
    #     # Return real-world coordinates (ignore the homogeneous coordinate)
    #     return real_coords[0], real_coords[1]

        
    # def xy_to_pixel(self, coord, msg): 
    #     x, y = coord[0], coord[1]
    #     position = msg.info.origin.position
    #     orientation = msg.info.origin.orientation
    #     resolution = msg.info.resolution

    #     #cells to meters
    #     u = (x - position.x)/resolution
    #     v = (y - position.y)/resolution

    #     # #
    #     # T = np.array([[1, 0, tx], 
    #     #               [0, 1, ty], 
    #     #               [0, 0, 1]])
    #     # np.linalg.inv()
    #     return (u, v)
    
    def xy_to_pixel(self, coord, msg): 
        x, y = coord[0], coord[1]
        orientation = msg.info.origin.orientation
        resolution = msg.info.resolution
        map_x = msg.info.origin.position.x
        map_y = msg.info.origin.position.y
        
        # transformation_matrix = tf.quaternion_matrix(orientation)
        q = [orientation.x,
            orientation.y,
            orientation.z,
            orientation.w]
        roll, pitch, yaw = tf.euler_from_quaternion(q)

        rotated_x = (x - map_x)*np.cos(yaw) - (y - map_y)*np.sin(yaw)
        rotated_y = (x - map_x)*np.sin(yaw) + (y - map_y)*np.cos(yaw)

        pixel_x = int(rotated_x/resolution)
        pixel_y = int(rotated_y/resolution)

        return pixel_x, pixel_y
    # def xy_to_pixel(self, coord, msg): 
    #     x, y = coord[0], coord[1]
    #     # translation = msg.info.origin.position
    #     orientation = msg.info.origin.orientation
    #     position = [msg.info.origin.position.x, msg.info.origin.position.y, 0]
    #     orientation = [orientation.x,
    #         orientation.y,
    #         orientation.z,
    #         orientation.w]
    #     resolution = msg.info.resolution
    #     transformation_matrix = tf.quaternion_matrix(orientation)
    #     transformation_matrix[0:3, 3] = position[0:3]

    #     real_coords = np.array([x, y, 0, 1])

    #     # Apply inverse transformation
    #     inv_transformation_matrix = np.linalg.inv(transformation_matrix)
    #     pixel_coords = inv_transformation_matrix @ real_coords

    #     # Scale by inverse resolution
    #     pixel_coords /= resolution

    #     # return int(pixel_coords[0]), int(pixel_coords[1])
    #     return int(pixel_coords[1]), int(pixel_coords[0])

    
    # def coord_to_pose(self, coord): 
    #     x, y, th = coord
    #     pose_msg = Pose() 
    #     pose_msg.position.x = x
    #     pose_msg.position.y = y
    #     pose_msg.position.z = 0.0
    #     quat = tf.quaternion_from_euler(0, 0, th)
    #     pose_msg.orientation = Quaternion(x=quat[0], y=quat[1], z=quat [2], w=quat[3])
    #     return pose_msg
    
    # def mappixel_to_xy(map_msg): 
    #     u, v = coord[0], coord[1]
    #     resolution = msg.info.resolution
    #     x= u*resolution
    #     y= v*resolution
    #     position = msg.info.origin.position
    #     orientation = msg.info.origin.orientation

    #     q = [orientation.x,
    #         orientation.y,
    #         orientation.z,
    #         orientation.w]
    #     rotation_matrix = tf.quaternion_matrix(q)[:3, :3]
    #     metric_coordinates = np.dot(rotation_matrix, np.array([x, y, 0]))
        
    #     # Apply the translation
    #     x_real = metric_coordinates[0] + position.x
    #     y_real = metric_coordinates[1] + position.y

    #     return (x_real, y_real)

    def map_cb(self, msg):
        self.map_input = msg
        self.occupancy_grid = np.reshape(msg.data, (msg.info.height, msg.info.width))
        self.obstacles = np.argwhere(self.occupancy_grid == 100)
        # self.unknown = np.argwhere(self.)
        self.get_logger().info(f"obstacle locations are {self.obstacles}")
        self.get_logger().info(f"num obstacles {len(self.obstacles)}")
        self.get_logger().info(f"num non obstacles {(len(self.occupancy_grid))**2}")

        self.get_logger().info('map + occupancy grid init')
    
    def pose_cb(self, pose):
        self.get_logger().info("pose callback")
        start_x, start_y = self.map_to_pixel(pose.pose.pose.position.x, pose.pose.pose.position.y)
        self.start_point = (start_x, start_y)
        self.plan_path(self.start_point, self.end_point, self.map)

    def goal_cb(self, msg):
        self.get_logger().info("goal callback")
        end_x, end_y = self.map_to_pixel(msg.pose.position.x, msg.pose.position.y)
        self.end_point = (end_x, end_y)
        self.plan_path(self.start_point, self.end_point, self.map)

        xyth = (self.pose_to_xyth(msg))
        self.end_coord = self.xy_to_pixel(xyth, self.map_input)

        self.get_logger().info(f"end coord is {self.end_coord}")

        self.plan_path(self.start_coord, self.end_coord, self.map_input)
        self.get_logger().info(f"start, end coord is {self.start_coord, self.end_coord}")

        self.get_logger().info('finish path plan')
        # self.get_logger().info(f"start, end coord is {self.start_coord, self.end_coord}")
        

    # def sample(self): 
    #     #generate a random sample in robot config space 
    #     # self.start[:2]
    #     return (random.uniform(0, len(self.occupancy_grid)), random.uniform(0, len(self.occupancy_grid)))
    
    def sample(self):
        """Generate a random sample in the robot configuration space."""
        return RobotConfig(random.uniform(0, len(self.occupancy_grid)),
                           random.uniform(0, len(self.occupancy_grid)))
        # return RobotConfig(random.uniform(-60, 25),
        #                    random.uniform(-20, 15))

    
    # def distance(self, coord1, coord2): 
    #     #return distance between two robot configs 
    #     return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    # def euclidean_distance(a, b):
    #     return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


    def collide(self, robot_config): 
        """
        checks if input robot config collides with some obstacle  
        """
        if self.occupancy_grid[int(robot_config.y)][int(robot_config.x)] ==-1: 
            return True

        for (u, v) in self.obstacles: 
            if np.abs(robot_config.x - v) <= 1 and np.abs(robot_config.y - u) <= 1: 
                return True
            # if np.abs(robot_config[0] - v) <= 1 and np.abs(robot_config[1] - u) <= 1: 
            #     self.get_logger().info('error from here---------')
            #     return True
            
        return False
        # return any(obs.collide(robot_config) for obs in self.obstacles)

    # def line_path_collides(self, start, end): 
    #     """
    #     checks if input robot config collides with some obstacle  
    #     """
        
    #     if self.occupancy_grid[int(robot_config.y)][int(robot_config.x)] ==-1: 
    #         return True

    #     for (u, v) in self.obstacles: 
    #         if np.abs(robot_config.x - v) <= 1 and np.abs(robot_config.y - u) <= 1: 
    #             return True
    #         # if np.abs(robot_config[0] - v) <= 1 and np.abs(robot_config[1] - u) <= 1: 
    #         #     self.get_logger().info('error from here---------')
    #         #     return True
            
    #     return False

    
    def plan_path(self, start_point, end_point, map):
        #Dijkstra's goes here
        #need to transform neighbor points to pixel space
        #when looking up obstacles in map

        #my main question is if start/end points are floats or not
        #and how to index into map if so 
        #and how rotation/translation works to translate into pixel space

        #IDEA FROM OH
        #GET MAP/STARTING/ENDING POINTS
        #TRANSLATE ALL TO PIXEL SPACE
        #RUN DIJKSTRA'S
        #TRANSLATE FINAL PATH BACK TO MAP SPACE
        #APPARENTLY TRANSLATION HANDLES ISSUE WITH NON INTEGERS
        #ONLY HARD PART IS FIGURING OUT TRANSLATION 

        #BFS FOR NOW
        if start_point == None or end_point == None:
            return 
        
        queue = [[start_point]]
        visited = set()
        visited.add(start_point)
        goal_path = []

        self.get_logger().info("starting bfs....")
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == end_point:
                self.get_logger().info("ended bfs....")
                goal_path = path
                break 

            neighbors = [(node[0] + 1, node[1]), (node[0] - 1, node[1]), 
                         (node[0], node[1] + 1), (node[0], node[1] - 1)]
            for neighbor in neighbors:
                if neighbor not in visited and neighbor[0] >= 0 and neighbor[0] < len(self.map[0]):
                    if neighbor[1] >= 0 and neighbor[1] < len(self.map): 
                        occupancy_value = self.map[neighbor[1], neighbor[0]]
                        if occupancy_value < 100 and occupancy_value != -1:
                            new_path = path.copy()
                            new_path.append(neighbor)
                            queue.append(new_path)
                        visited.add(neighbor)

        # pixel_to_map_arrayfunc = np.vectorize(self.pixel_to_map)
        # map_goal_path = pixel_to_map_arrayfunc(np.array(goal_path))   
        self.get_logger().info("path callback")
        #USE SOME NUMPY FUNC LATER FOR PERFOMANCE 
        map_goal_path = []
        for pixel_point in goal_path:
            map_point_x, map_point_y = self.pixel_to_map(pixel_point[0], pixel_point[1])
            map_goal_path.append((map_point_x, map_point_y))
    
        self.trajectory.points = map_goal_path
        
        self.get_logger().info(str(self.trajectory.points))
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def map_to_pixel(self, x, y):
        # euler = tf.euler_from_quaternion((self.map_origin_orientation.x, self.map_origin_orientation.y, self.map_origin_orientation.z, self.map_origin_orientation.w))
        # th = -euler[2]

        # rotMatrix = np.array([[np.cos(th), -np.sin(th)],
        #                       [np.sin(th), np.cos(th)]])
        # rotPoint = np.array([[x, y]]) @ rotMatrix

        # u, v = x - self.map_origin_poistion.x, y - self.map_origin_poistion.y

        # u, v = rotPoint[0, 0] * (1/self.map_resolution), rotPoint[0, 1] * (1/self.map_resolution)
        # return u, v

        euler = tf.euler_from_quaternion((self.map_origin_orientation.x, self.map_origin_orientation.y, self.map_origin_orientation.z, self.map_origin_orientation.w))
        th = euler[2]

        rot_x, rot_y = (x - self.map_origin_poistion.x)*np.cos(th) - (y - self.map_origin_poistion.y)*np.sin(th), (x - self.map_origin_poistion.x)*np.sin(th) + (y - self.map_origin_poistion.y)*np.cos(th)
        u, v = rot_x * (1/self.map_resolution), rot_y * (1/self.map_resolution)

        return int(u), int(v)
    
    def pixel_to_map(self, u, v):
        # x, y = u*self.map_resolution, v*self.map_resolution
        # euler = tf.euler_from_quaternion((self.map_origin_orientation.x, self.map_origin_orientation.y, self.map_origin_orientation.z, self.map_origin_orientation.w))
        # th = euler[2]

        # x, y = rotPoint[0, 0] + self.map_origin_poistion.x, rotPoint[0, 1] + self.map_origin_poistion.y
        
        # rotMatrix = np.array([[np.cos(th), -np.sin(th)],
        #                       [np.sin(th), np.cos(th)]])
        # rotPoint = np.array([[x, y]]) @ rotMatrix
        # return x, y

        x, y = u*self.map_resolution, v*self.map_resolution

        euler = tf.euler_from_quaternion((self.map_origin_orientation.x, self.map_origin_orientation.y, self.map_origin_orientation.z, self.map_origin_orientation.w))
        th = euler[2]

        rot_x, rot_y = x*np.cos(th) - y*np.sin(th), x*np.sin(th) + y*np.cos(th)
        x, y = rot_x + self.map_origin_poistion.x, rot_y + self.map_origin_poistion.y
        
        return x, y




def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
