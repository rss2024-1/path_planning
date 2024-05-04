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
        self.declare_parameter('rrt_step', 40.0) # value in pixels - note that 1 pixel on stata basement map is approx. 0.05 meters irl
        self.declare_parameter('rrt_bias', 0.05) # rrt goal bias - fraction of time to specifically sample the goal point
        self.declare_parameter('car_size', 2.0) # radius (in pixels) to give as clearance for the robot
        self.declare_parameter('run_rrt*', True) # bool to run rrt* optimization after rrt finds a solution
        self.declare_parameter('rrt*_n_exp', 30) # number of nearest nodes to check in rrt* restructuring
        self.declare_parameter('rrt*_d_exp', 3.0) # radius (as multiple of rrt_step) to check for nearby nodes in rrt* restructuring


        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.max_rrt_points = self.get_parameter('rrt_points').value
        self.rrt_step_size = self.get_parameter('rrt_step').value
        self.rrt_goal_bias = self.get_parameter('rrt_bias').value
        self.robot_clearance = self.get_parameter('car_size').value
        self.run_star = self.get_parameter('run_rrt*').value
        self.num_node_expand = self.get_parameter('rrt*_n_exp').value
        self.dist_node_expand = self.get_parameter('rrt*_d_exp').value

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
        
        # > Debug/extra publishers below <
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
        # ^ Debug/extra publishers above ^
        
        # vars to store node data
        self.map_input, self.map_width, self.map_height, self.map_res = None, None, None, None
        self.occupancy_grid, self.goal_pose, self.start_pose = None, None, None
        
        self.rrt_tree, self.rrt_path, self.rrt_cost = None, None, np.inf
        self.rrt_star_tree, self.rrt_star_path, self.rrt_star_cost = None, None, np.inf
    
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
    #     return float(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
    
    # def dist_to(p2,p1): # not using b/c f'n calls are slow
    #     '''returns the distance to point 2 from point 1'''
    #     return float(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
    
    def nearest_pt(self, points, target):
        '''returns the nearest existing node in the tree to a proposed new point'''
        nearest, index_of, best_dist = None, None, np.inf
        for i, point in enumerate(points):
            this_dist = np.sqrt((target[0]-point[0][0])**2 + (target[1]-point[0][1])**2)
            if this_dist < best_dist:
                nearest, index_of, best_dist = point, i, this_dist
        return nearest, index_of, best_dist
    
    def within_d(self, nodes, target):
        '''returns a list of nodes nearby the target, with each element being [index of node in RRT tree, distance to node from target]'''
        max_d, nearby = self.rrt_step_size * self.dist_node_expand, []
        for i, node in enumerate(nodes):
            distance = np.sqrt((node[0][0]-target[0][0])**2+(node[0][1]-target[0][1])**2)
            if 0.01 < distance < max_d:
                nearby.append([i, distance])
        return nearby
        
    # def n_nearest(self, points, target):
    #     point_list, nearby_pts, n = [], [], self.num_node_expand
    #     for point in points:
    #         if point != target: point_list.append(point)
    #     while 0 < len(nearby_pts) < n:
    #         next = self.nearest_pt(point_list, target)
    #         point_list.remove(next)
    #         nearby_pts.append(next)
    #     return nearby_pts
    
    def collision(self, start, end, dist):
        '''checks for a collision between a proposed path segment and obstacles on the map'''
        collides, divs, path_pix = False, int(round(4*dist)), [] # local f'n variables
        # Construct lists of points that should intersect with all pixels traversed
        
        # self.get_logger().info(str("(x1,y1) = ({},{}), (x2,y2) = ({},{}), num divs = {}".format(start[0], start[1], end[0], end[1], divs)))
        xlist, ylist = np.linspace(start[0], end[0], divs), np.linspace(start[1], end[1], divs)
        path_pix.append([xlist[0], ylist[0]])
        
        # Check end point first, return immediatly if fails
        if self.occupancy_grid[int(end[0])][int(end[1])] != 0:
            collides = True
            return collides
        
        # Build list of pixels traversed
        for i in range(1, divs-1):
            pixel = [round(xlist[i]), round(ylist[i])]
            if not np.array_equiv(path_pix[-1], pixel):
                path_pix.append(pixel)
        
        # Check each pixel in list against the occupancy grid, return immediately if any fail
        for p in path_pix:
            if self.occupancy_grid[int(p[0])][int(p[1])] != 0:
                collides = True
                return collides
        
        # Return collides - should still be false if reached this point
        return collides
    
    def path_traceback(self, tree):
        '''Trace a solved RRT tree from the end node back to the root'''
        # RRT stops adding nodes when it reaches the goal, so the
        # last eltement of the tree array should be the goal
        path, parent_ptr = [], len(tree)-1
        cost = tree[parent_ptr][2]
        
        # Only node w/no parent should be root - follow parent nodes
        # back to there and add them to the path in the process
        while not parent_ptr == None:
            path.insert(0, tree[parent_ptr][0])
            parent_ptr = tree[parent_ptr][1]
        return path, cost

    def map_cb(self, msg): # nav_msgs/msg/OccupancyGrid.msg
        self.map_input, self.start_pose, self.goal_pose, step_size = msg, None, None, self.rrt_step_size
        self.map_width, self.map_height, self.map_res = msg.info.width, msg.info.height, msg.info.resolution
        self.occupancy_grid = np.transpose(np.reshape(msg.data, (msg.info.height, msg.info.width)))
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
        return

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
        return

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
        return

    def plan_path(self):
        self.rrt_tree, self.rrt_path, self.rrt_cost = None, None, np.inf
        self.rrt_star_tree, self.rrt_star_path, self.rrt_star_cost = None, None, np.inf
        start_time, end_time = time.time(), None
        if self.rrt():
            end_time = time.time()
            path, cost = self.rrt_path, self.rrt_cost
            self.get_logger().info(str("RRT found a path of length {:.2f} meters".format(cost * self.map_res)))
            self.trajectory.points = [self.pixel_to_xyth(point, self.map_input) for point in path]
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
        else:
            end_time = time.time()
            self.get_logger().info("RRT did not find a path")
        self.get_logger().info(str("RRT time to completion was {:.2f} seconds".format(end_time-start_time)))
        
        if self.run_star and self.rrt_cost < np.inf:
            self.get_logger().info("Starting RRT* optimization")
            start_time = time.time()
            savings = self.rrt_star()
            end_time = time.time()
            path, cost = self.rrt_star_path, self.rrt_star_cost
            self.get_logger().info(str("RRT* found a path that is {:.1f} percent shorter than RRT".format(100*savings/self.rrt_cost)))
            self.get_logger().info(str("Final path length was {:.2f} meters".format(self.rrt_star_cost * self.map_res)))
            self.get_logger().info(str("RRT* path optimization took {:.2f} seconds".format(end_time-start_time)))
            # self.get_logger().info("Publishing new path in:")
            self.trajectory.points = [self.pixel_to_xyth(point, self.map_input) for point in path]
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
            
        return
        
    def rrt(self):
        '''implements rrt for a path planning node'''
        r.seed()
        
        # local variables
        samp_exp = -1 # can cause bugs if raised above zero
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
            else: sample = (r.randint(-samp_exp, self.map_width+samp_exp), r.randint(-samp_exp, self.map_height+samp_exp), None)
            if False: # Published the sampled coordinates
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
            if False: # Publish the coordinates of the nearest existing node
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
            if False: # Publish the coordinates of the proposed new point
                prop_xy = self.pixel_to_xyth(new_coord, self.map_input)
                prop_point = PointStamped()
                prop_point.header.frame_id = "map"
                prop_point.header.stamp = self.get_clock().now().to_msg()
                prop_point.point.x, prop_point.point.y, prop_point.point.z = prop_xy[0], prop_xy[1], 0.0
                self.proposed_pub.publish(prop_point)
            
            # check for collisions on path segment, if no collision then add point to tree and increment the points count
            if not self.collision(near_coord, new_coord, segment_distance):
                cost = points[near_idx][2]+segment_distance
                points.append([new_coord, near_idx, cost])
                p += 1
                if False: # Publish the coordinates of the added RRT node
                    xy_coord = self.pixel_to_xyth(new_coord, self.map_input)
                    ros_point = PointStamped()
                    ros_point.header.frame_id = "map"
                    ros_point.header.stamp = self.get_clock().now().to_msg()
                    ros_point.point.x, ros_point.point.y, ros_point.point.z = float(xy_coord[0]), float(xy_coord[1]), 0.0
                    self.point_pub.publish(ros_point)
            else: # otherwise increment the fail count
                f += 1
                continue
            
            # check if latest point is close enough to the goal pose, if yes then trace the path back to the head of the list and mark as solved
            if np.sqrt((self.goal_pose[0]-new_coord[0])**2 + (self.goal_pose[1]-new_coord[1])**2) <= 0.01:
                self.rrt_tree = points
                self.rrt_path, self.rrt_cost = self.path_traceback(points)
                solved = True
                return solved
                
        # TODO actual error handling?
        if not solved: print("No path found in {} attempts".format(p+f))
        return solved # should be false if we get here
    
    def rrt_star(self):
        points, adj_lists = [elt[:] for elt in self.rrt_tree], []
        updates, num_pts = 1, len(points)
        
        self.get_logger().info("Starting adjacency lists")
        start_time = time.time()
        # Builds adjacency lists, records distance for each edge
        for point in points:
            nearby_adj = []
            nearby_nodes = self.within_d(points, point)
            for node in nearby_nodes:
                # self.get_logger().info(str("Node at index {} is coord {}, parent {}, cost {}".format(node[0], points[node[0]][0], points[node[0]][1], points[node[0]][2])))
                if not self.collision(points[node[0]][0], point[0], node[1]):
                    nearby_adj.append(node)
            adj_lists.append(nearby_adj)
        mid_time = time.time()
        self.get_logger().info(str("Adjacency lists finished in {:.2f} seconds".format(mid_time-start_time)))
        
        self.get_logger().info("Starting update loops")
        # Update loop - goes until there is nothing else to update
        while updates > 0:
            updates = 0
            for i in range(num_pts):
                best_parent, best_cost = points[i][1], points[i][2]
                for j, k in adj_lists[i]:
                    # self.get_logger().info(str("".format()))
                    cost = points[j][2] + k
                    if cost < best_cost:
                        updates += 1
                        best_parent, best_cost = j, cost
                points[i][1], points[i][2] = best_parent, best_cost
        end_time = time.time()
        self.get_logger().info(str("Update loops finished in {:.2f} seconds".format(end_time-mid_time)))
        
        self.rrt_star_tree = points            
        self.rrt_star_path, self.rrt_star_cost = self.path_traceback(points)
        cost_savings = self.rrt_cost - self.rrt_star_cost
        return cost_savings

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
