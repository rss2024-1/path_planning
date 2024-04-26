import rclpy
import numpy as np
import random as r
import tf_transformations as tf
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, PointStamped
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import time


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('rrt_points', 10e3)
        self.declare_parameter('rrt_step', 60.0) # value in pixels - note that 1 pixel on stata basement map is approx. 0.05 meters irl
        self.declare_parameter('rrt_bias', 0.1)
        self.declare_parameter('car_size', 2.0)
        self.declare_parameter('goal_dist', 1.0)
        self.declare_parameter('goal_angle', 0.3)

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.max_rrt_points = self.get_parameter('rrt_points').value
        self.rrt_step_size = self.get_parameter('rrt_step').value
        self.rrt_goal_bias = self.get_parameter('rrt_bias').value
        self.robot_clearance = self.get_parameter('car_size').value
        self.goal_distance_th = self.get_parameter('goal_dist').value
        self.goal_angle_th = self.get_parameter('goal_dist').value

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
        
        self.point_pub = self.create_publisher(
            PointStamped,
            "/rrt/nodes",
            10
        )

        self.proposed_pub = self.create_publisher(
            PointStamped,
            "/rrt/proposed",
            10
        )
        
        self.nearest_pub = self.create_publisher(
            PointStamped,
            "/rrt/nearest",
            10
        )

        self.sample_pub = self.create_publisher(
            PointStamped,
            "/rrt/sample",
            10
        )

        self.point_test = self.create_publisher(
            PointStamped,
            "/rrt/tests",
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
        
        # vars to store data from subs TODO set reasonable empty/default values
        self.map_input = None
        self.map_width = -1
        self.map_height = -1
        self.occupancy_grid = None
        self.obstacles = None
        self.div_obst = None
        self.goal_pose = None
        self.start_pose = None
        
        self.rrt_tree = None
        self.rrt_path = None
        self.rrt_cost = np.inf
    
    def pixel_to_xyth(self, coord, msg): 
        u, v, pix_th = coord[0], coord[1], coord[2]
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
        if not pix_th == None: theta = pix_th + yaw
        else: theta = None
        
        return (x, y, theta)
    
    def xyth_to_pixel(self, coord, msg): 
        x, y, theta = coord[0], coord[1], coord[2]
        orntn = msg.info.origin.orientation
        res = msg.info.resolution
        map_x = msg.info.origin.position.x
        map_y = msg.info.origin.position.y
        
        # transformation_matrix = tf.quaternion_matrix(orientation)
        q = [orntn.x, orntn.y, orntn.z, orntn.w]
        roll, pitch, yaw = tf.euler_from_quaternion(q)

        rotated_x = (x - map_x)*np.cos(yaw) - (y - map_y)*np.sin(yaw)
        rotated_y = (x - map_x)*np.sin(yaw) + (y - map_y)*np.cos(yaw)

        pixel_x = int(rotated_x/res)
        pixel_y = int(rotated_y/res)
        if not theta == None: pixel_th = theta + yaw
        else: pixel_th = None
        
        return (pixel_x, pixel_y, pixel_th)

    # def angle_to(p2,p1): # not using b/c f'n calls are slow
    #     '''returns the (pixel grid) angle to point 2 from point 1, with correct quadrant'''
    #     return float(np.arctan2(p2[0]-p1[0], p2[1]-p1[1]))
    
    # def dist_to(p2,p1): # not using b/c f'n calls are slow
    #     '''returns the distance to point 2 from point 1'''
    #     return float(np.sqrt((p2[0]-p1[0])^2 + (p2[1]-p1[1])^2))
    
    def nearest_pt(self, points, target):
        '''returns the nearest existing point in the tree to a proposed new point'''
        nearest, index_of, best_dist = None, None, np.inf
        for i, point in enumerate(points):
            this_dist = np.sqrt((target[0]-point[0][0])**2 + (target[1]-point[0][1])**2)
            if this_dist < best_dist:
                nearest, index_of, best_dist = point, i, this_dist
        return nearest, index_of, best_dist
    
    def n_nearest(self, points, target, n: int):
        point_list, nearby_pts = [], []
        for point in points:
            if point != target: point_list.append(point)
        while 0 < len(nearby_pts) < n:
            next = self.nearest_pt(point_list, target)
            point_list.remove(next)
            nearby_pts.append(next)
        return nearby_pts
    
    def collision(self, start, end, dist):
        '''checks for a collision between a proposed path segment and obstacles on the map'''
        collides, divs, path_pix = False, int(round(4*dist)), []
        
        # x_quads, y_quads = [], []
        # start_quad = [int(start[0]//self.rrt_step_size), int(start[1]//self.rrt_step_size)]
        # end_quad = [int(end[0]//self.rrt_step_size), int(end[1]//self.rrt_step_size)]
        # if start_quad[0] == end_quad[0]: x_quads.append(start_quad[0])
        # else: x_quads.extend((start_quad[0], end_quad[0]))
        # if start_quad[1] == end_quad[1]: y_quads.append(start_quad[1])
        # else: y_quads.extend((start_quad[1], end_quad[1]))
        # # self.get_logger().info(str("x quadrants: {}, y quadrants: {}".format(x_quads, y_quads)))
        
        xlist, ylist = np.linspace(start[0], end[0], divs), np.linspace(start[1], end[1], divs)
        path_pix.append([xlist[0], ylist[0]])
        # end_x, end_y = int(end[0]), int(end[1])
        
        if self.occupancy_grid[int(end[0])][int(end[1])] != 0:
            collides = True
            return collides
        
        for i in range(1, divs-1):
            pixel = [round(xlist[i]), round(ylist[i])]
            if not np.array_equiv(path_pix[-1], pixel):
                path_pix.append(pixel)
                
        for p in path_pix:
            if self.occupancy_grid[int(p[0])][int(p[1])] != 0:
                collides = True
                return collides
                  
                # for x in x_quads:
                #     for y in y_quads:                
                #         for obstacle in self.div_obst[x][y]:
                #             if np.array_equiv(pixel, obstacle):
                #                 collides = True
                #                 return collides
        
        # self.get_logger().info(str("For start: {} and end: {}, collision found: {}, pixel path is:\n{}".format(start, end, collides, path_pix)))
        return collides
    
    def path_traceback(self, tree):
        path = []        
        parent_ptr = len(tree)-1
        cost = tree[parent_ptr][2]
        while not parent_ptr == None:
            path.insert(0, tree[parent_ptr][0])
            parent_ptr = tree[parent_ptr][1]
        return path, cost


    def map_cb(self, msg): # nav_msgs/msg/OccupancyGrid.msg
        self.map_input, self.start_pose, self.goal_pose, step_size = msg, None, None, self.rrt_step_size
        self.map_width, self.map_height = msg.info.width, msg.info.height
        self.occupancy_grid = np.transpose(np.reshape(msg.data, (msg.info.height, msg.info.width)))
        self.obstacles = np.argwhere(self.occupancy_grid==100)
        x_divs = int(self.map_width//step_size) + 1
        y_divs = int(self.map_height//step_size) + 1
        self.div_obst = []
        for i in range(x_divs):
            self.div_obst.append([])
            for j in range(y_divs):
                self.div_obst[i].append([])
        for obst in self.obstacles:
            x_quad, y_quad = int(obst[0]//step_size), int(obst[1]//step_size)
            self.div_obst[x_quad][y_quad].append([obst[0], obst[1]])
        # self.get_logger().info(str("divided obstacles: {}".format(self.div_obst)))
        if True: # print statements for debug - set True/False
            # self.get_logger().info(str("\nobstacles:\n" + str(self.obstacles)))
            '''# write obstacles list to text file
            with open("Obstacles.txt", "w") as text_file:
                for i in range(len(self.obstacles)):
                    text_file.write("{}\n".format(self.obstacles[i]))
            '''
            # self.get_logger().info(str("\noccupancy grid:\n" + str(self.occupancy_grid)))
            '''# write occupancy grid to text file
            with open("Occupancy.txt", "w") as text_file:
                for i in range(self.map_height):
                    text_file.write("[")
                    for j in range(self.map_width):
                        text_file.write("{},".format(self.occupancy_grid[i][j]))
                    text_file.write("]\n")
            '''
        # TODO implement map dilation/erosion/whatever for robot clearance
        if False:
            test_point = PointStamped()
            test_point.header.frame_id = "map"
            self.get_logger().info("Publishing test points")
            for i in [1, 865, 1729]:
                for j in [1, 650, 1299]:
                    xy = self.pixel_to_xyth((i,j,None), self.map_input)
                    self.get_logger().info("Publishing point at ({},{})".format(xy[0],xy[1]))
                    test_point.header.stamp = self.get_clock().now().to_msg()
                    test_point.point.x, test_point.point.y, test_point.point.z = xy[0], xy[1], 0.0
                    self.point_test.publish(test_point)
            for i, obs in enumerate(self.obstacles):
                if i%100 != 0: continue
                else:
                    xy = self.pixel_to_xyth((obs[0],obs[1],None), self.map_input)
                    self.get_logger().info("Publishing point at ({},{})".format(xy[0],xy[1]))
                    test_point.header.stamp = self.get_clock().now().to_msg()
                    test_point.point.x, test_point.point.y, test_point.point.z = xy[0], xy[1], 0.0
                    self.point_test.publish(test_point)   
            self.get_logger().info("Finished publishing test points")
        self.get_logger().info("Map received - waiting on poses")

    def pose_cb(self, msg): # geometry_msgs/msg/PoseWithCovarianceStamped
        if self.map_input == None:
            self.get_logger().info("Waiting on map")
            return
        xyth = (msg.pose.pose.position.x, msg.pose.pose.position.y, 2*np.arccos(msg.pose.pose.orientation.w)-np.pi)
        self.start_pose = self.xyth_to_pixel(xyth, self.map_input)
        if not self.goal_pose == None:
            self.get_logger().info(str("Planning path from ({},{}) to ({},{})".format
                                       (self.start_pose[0], self.start_pose[1], self.goal_pose[0], self.goal_pose[1])))
            self.plan_path()
        else: self.get_logger().info("Waiting on goal pose")

    def goal_cb(self, msg): # geometry_msgs/msg/PoseStamped
        if self.map_input == None:
            self.get_logger().info("Waiting on map")
            return
        xyth =  (msg.pose.position.x, msg.pose.position.y, 2*np.arccos(msg.pose.orientation.w)-np.pi)
        self.goal_pose = self.xyth_to_pixel(xyth, self.map_input)
        if not self.start_pose == None:
            self.get_logger().info(str("Planning path from ({},{}) to ({},{})".format
                                       (self.start_pose[0], self.start_pose[1], self.goal_pose[0], self.goal_pose[1])))
            self.plan_path()
        else: self.get_logger().info("Waiting on start pose")

    def plan_path(self):
        start_time, end_time = time.time(), None
        if self.rrt():
            end_time = time.time()
            map_msg = self.map_input
            path, cost = self.path_traceback(self.rrt_tree)
            self.get_logger().info(str("RRT found a path of length {}".format(cost)))
            self.trajectory.points = [self.pixel_to_xyth(point, map_msg) for point in path]
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
        else:
            end_time = time.time()
            self.get_logger().info("RRT did not find a path")
        self.get_logger().info(str("time to solution: {} seconds".format(end_time-start_time)))
        
    def rrt(self):
        '''implements rrt for a path planning node'''
        r.seed()
        
        # local variables
        samp_exp = -3 # can cause bugs if raised above zero
        p, f, solved = 0, 0, False # points placed, failed attempts, solution found
        max_pts, max_step = self.max_rrt_points, self.rrt_step_size
        points = []
        
        # add root of tree - [(x, y, theta), index of parent, cost to reach]
        points.append([(self.start_pose[0], self.start_pose[1], self.start_pose[2]), None, 0.0])
        
        # rrt loop
        while (p+f)<(5*max_pts) and not solved:
            # self.get_logger().info(str("attempted samples: {}, of allowed samples: {}".format(p+f,5*max_pts)))
            # loop variables
            sample, nearest, near_idx, new_coord = None, None, None, None
            
            # set sample point with goal bias
            if r.random() < self.rrt_goal_bias: sample = (self.goal_pose[0], self.goal_pose[1], self.goal_pose[2])
            # else:
            #     while sample == None:
            #         coord = (r.randint(0, self.map_width-1), r.randint(0, self.map_height-1), None)
                    # if self.occupancy_grid[coord[0]][coord[1]] == 0: sample = coord
            else: sample = (r.randint(-samp_exp, self.map_width+samp_exp), r.randint(-samp_exp, self.map_height+samp_exp), None)
            if False:
                sample_xy = self.pixel_to_xyth(sample, self.map_input)
                sample_point = PointStamped()
                sample_point.header.frame_id = "map"
                sample_point.header.stamp = self.get_clock().now().to_msg()
                sample_point.point.x, sample_point.point.y, sample_point.point.z = sample_xy[0], sample_xy[1], 0.0
                self.sample_pub.publish(sample_point)
            
            # find the nearest existing point and its index in 'points'
            nearest, near_idx, segment_distance = self.nearest_pt(points, sample) # TODO implement a ?more efficient? f'n to find nearest point
            near_coord = nearest[0]
            if near_coord == sample: continue
            if False:
                near_xy = self.pixel_to_xyth(near_coord, self.map_input)
                near_point = PointStamped()
                near_point.header.frame_id = "map"
                near_point.header.stamp = self.get_clock().now().to_msg()
                near_point.point.x, near_point.point.y, near_point.point.z = near_xy[0], near_xy[1], 0.0
                self.nearest_pub.publish(near_point)
            
            # calculate the coordinates of the new point
            if segment_distance <= max_step:
                new_coord = sample
            else:
                angle = np.arctan2(sample[1]-near_coord[1], sample[0]-near_coord[0])
                new_coord = (near_coord[0]+round(max_step*np.cos(angle)), near_coord[1]+round(max_step*np.sin(angle)), None)
                segment_distance = np.sqrt((new_coord[0]-near_coord[0])**2 + (new_coord[1]-near_coord[1])**2)
            if False:
                prop_xy = self.pixel_to_xyth(new_coord, self.map_input)
                prop_point = PointStamped()
                prop_point.header.frame_id = "map"
                prop_point.header.stamp = self.get_clock().now().to_msg()
                prop_point.point.x, prop_point.point.y, prop_point.point.z = prop_xy[0], prop_xy[1], 0.0
                self.proposed_pub.publish(prop_point)
            
            # check for collisions on path segment, if no collision then add point to tree.
            if not self.collision(near_coord, new_coord, segment_distance):
                cost = points[near_idx][2]+segment_distance
                points.append([new_coord, near_idx, cost])
                p += 1
                if False: # publish coordinates of added RRT nodes
                    xy_coord = self.pixel_to_xyth(new_coord, self.map_input)
                    ros_point = PointStamped()
                    ros_point.header.frame_id = "map"
                    ros_point.header.stamp = self.get_clock().now().to_msg()
                    ros_point.point.x, ros_point.point.y, ros_point.point.z = float(xy_coord[0]), float(xy_coord[1]), 0.0
                    self.point_pub.publish(ros_point)
            else:
                f += 1
                continue
            
            # check if latest point is close enough to the goal pose, if yes then trace the path back to the head of the list and mark as solved
            if np.sqrt((self.goal_pose[0]-new_coord[0])**2 + (self.goal_pose[1]-new_coord[1])**2) <= 0.01:
                self.rrt_tree = points
                self.rrt_path, self.rrt_cost = self.path_traceback(points)
                solved = True
                
        # TODO actual error handling
        if not solved: print("No path found in {} attempts".format(p+f))
        return solved

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
