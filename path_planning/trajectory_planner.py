import rclpy
import numpy as np
import random as r
import tf_transformations as tf
from rclpy.node import Node
from queue import PriorityQueue

assert rclpy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from .utils import LineTrajectory
import time

from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class QueueItem:
    priority: int
    item: Any=field(compare=False)
    
class Path():
    def __init__(self, path, cost):
        self.path = path
        self.cost = cost
        self.start_pose = None
        self.goal_pose = None

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('map_dilate', 5) # integer number of pixels to dilate obstacles on map to give clearance for the robot
        self.declare_parameter('internode_distance', 4) # integer number of pixels to space between nodes while building graph; 1 pixel = 0.05 meters irl
        
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.map_dilate = self.get_parameter('map_dilate').value
        self.node_dist = self.get_parameter('internode_distance').value
        self.lane_trajectory = [(-19.99921417236328, 1.3358267545700073), (-18.433984756469727, 7.590575218200684), (-15.413466453552246, 10.617328643798828), (-6.186201572418213, 21.114534378051758), (-5.5363922119140625, 25.662315368652344), (-19.717021942138672, 25.677358627319336), (-20.30797004699707, 26.20694923400879), (-20.441822052001953, 33.974945068359375), (-55.0716438293457, 34.07769775390625), (-55.30067825317383, 1.4463690519332886)]
        self.enter_wall = [(-25.3534, -3.56703), (-25.3534, 2.86331)]
        self.exit_wall = [(-51.9571, -3.179), (-51.9571, 2.4451)]
        self.shell_run_wall = [(-20.6, 2.46), (-15.4899, 2.31881)]

        ''' # Parameters for rrt
        # self.declare_parameter('rrt_points', 3e3) # maximum allowed size of rrt tree
        # self.declare_parameter('samp_free_only', False) # RRT takes samples only from free space; technically correct for RRT, but produces much larger trees under some circumstances
        # self.declare_parameter('rrt_step', 40.0) # value in pixels - note that 1 pixel on stata basement map is approx. 0.05 meters irl
        # self.declare_parameter('rrt_bias', 0.05) # rrt goal bias - fraction of time to specifically sample the goal point
        # self.declare_parameter('run_rrt*', True) # bool to run rrt* optimization after rrt finds a solution
        # # self.declare_parameter('rrt*_n_exp', 30) # number of nearest nodes to check in rrt* restructuring
        # self.declare_parameter('rrt*_d_exp', 2.0) # radius (as multiple of rrt_step) to check for nearby nodes in rrt* restructuring
        # self.declare_parameter('star_timeout', 30.0) # timout in seconds for rrt* optimization attempt
        
        # self.max_rrt_points = self.get_parameter('rrt_points').value
        # self.free_space_only = self.get_parameter('samp_free_only').value
        # self.rrt_step_size = self.get_parameter('rrt_step').value
        # self.rrt_goal_bias = self.get_parameter('rrt_bias').value
        # self.run_star = self.get_parameter('run_rrt*').value
        # # self.num_node_expand = self.get_parameter('rrt*_n_exp').value
        # self.dist_node_expand = self.get_parameter('rrt*_d_exp').value
        # self.rrt_star_timeout = self.get_parameter('star_timeout').value
        '''

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
        
        self.traj_update_pub = self.create_publisher(
            Bool,
            "/trajectory/updating",
            1
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )
        
        self.odom_updates = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_cb,
            10
        )

        self.shell_points_sub = self.create_subscription(
            PoseArray, 
            "/shell_points", 
            self.shell_cb, 
            1)
        
        self.graph_pub = self.create_publisher(
            PointStamped,
            "/a_star/nodes",
            10
        )
        
        self.visiting_pub = self.create_publisher(
            PointStamped,
            "/a_star/visiting",
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        self.lane_trajectory_viz = LineTrajectory(node=self, viz_namespace="/lane_trajectory")
        self.enter_trajectory_viz = LineTrajectory(node=self, viz_namespace="/enter_trajectory")
        self.exit_trajectory_viz = LineTrajectory(node=self, viz_namespace="/exit_trajectory")
        self.shell_run_close_trajectory_viz = LineTrajectory(node=self, viz_namespace="/shell_run_trajectory")
        self.shell_paths = []
        self.shell_goals = []
        self.shell_wall_pixels = []
        self.reached_goal = True
        ''' # Debug publishers for rrt
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
        '''
        
        # vars to store map, graph data
        self.map_input, self.map_width, self.map_height, self.map_res = None, None, None, None
        self.occupancy_grid, self.goal_pose, self.start_pose, self.current_pose = None, None, None, None
        
        self.graph = dict()
        # self.path_index, self.path_list = 0, []
        
        # self.update = False
        # self.rrt_tree, self.rrt_path, self.rrt_cost = None, None, np.inf
        # self.rrt_star_tree, self.rrt_star_path, self.rrt_star_cost = None, None, np.inf

    def pixel_to_xyth(self, coord, msg):
        u, v, pix_th = coord[0], coord[1], None
        if len(coord) == 3: pix_th = coord[2]
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
        x, y, theta = coord[0], coord[1], None
        if len(coord) == 3: theta = coord[2]
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
    
    def build_graph(self):
        d = self.node_dist # pixel distance between graph nodes
        nodes = dict() # initialize dictionary to build graph
        
        # build a dictionary ("nodes") of each open point (node)
        for i in range(0, self.map_width-1, d):
            for j in range(0, self.map_height-1, d):
                if self.occupancy_grid[i][j] == 0:
                    nodes[(i,j)] = dict()
        
        # build adjacency lists/dicts ("nodes[node]") for each open point
        for node in nodes.keys():
            x1, y1 = node
            # directly adjacent nodes
            for x2, y2 in [(d,0), (0,d), (-d,0), (0,-d)]:
                adj_node, dist = (x1+x2, y1+y2), float(d)
                if (adj_node in nodes.keys()):
                    if node in nodes[adj_node].keys(): nodes[node][adj_node] = nodes[adj_node][node]
                    elif not self.collision(node, adj_node, dist): nodes[node][adj_node] = dist
                    else: continue
            # diagonally adjacent nodes, causes suboptimal paths for some reason that I don't fully understand
            # for x2, y2 in [(d,d), (-d,d), (-d,-d), (d,-d)]:
            #     adj_node, dist = (x1+x2, y1+y2), d*np.sqrt(2)
            #     if (adj_node in nodes.keys()):
            #         if node in nodes[adj_node].keys(): nodes[node][adj_node] = nodes[adj_node][node]
            #         elif not self.collision(node, adj_node, dist): nodes[node][adj_node] = dist
            #         else: continue

        # assign graph to global variable and return its size
        self.graph = nodes
        return len(nodes)
    
    def collision(self, start, end, dist):
        '''checks for a collision between a proposed path segment and obstacles on the map'''
        collides, divs, path_pix = False, int(2*dist), [] # local f'n variables
        # Construct lists of points that should intersect with all pixels traversed
        
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

    def map_cb(self, msg): # nav_msgs/msg/OccupancyGrid.msg
        self.map_input, self.start_pose, self.goal_pose = msg, None, None
        self.map_width, self.map_height, self.map_res = msg.info.width, msg.info.height, msg.info.resolution
        self.occupancy_grid = np.transpose(np.reshape(msg.data, (msg.info.height, msg.info.width)))
        
        self.get_logger().info(f"Starting map import and graph construction")
        start_time = time.time()

        #ADDING LANE TRAJECTORY AS OBSTACLE FOR FINAL CHALLENGE 
        lane_traj_pixels = [self.xyth_to_pixel((point[0], point[1]), msg) for point in self.lane_trajectory]
        lane_pixels = []
        for ix in range(len(lane_traj_pixels)):
            if ix < len(lane_traj_pixels) - 1:
                point1 = lane_traj_pixels[ix]
                point2 = lane_traj_pixels[ix+1]
                lane_segment_pixels = self.bresenham_line_supercover(point1[0], point1[1], point2[0], point2[1])
                lane_pixels = lane_pixels + lane_segment_pixels
        for lane_pixel in lane_pixels:
            self.occupancy_grid[lane_pixel[0]][lane_pixel[1]] = 100
        #self.occupancy_grid[lane_pixels] = 100
        self.lane_trajectory_viz.clear()
        for ix, lane_pixel in enumerate(lane_pixels):
            xy_lane_cell = self.pixel_to_xyth((lane_pixel[0], lane_pixel[1]), msg)
            self.lane_trajectory_viz.addPoint(xy_lane_cell)
        self.lane_trajectory_viz.publish_viz()  
        
        # dilate obstacles on map for robot clearance
        self.map_dilate = 0 # skip map dilation
        if self.map_dilate > 0:        
            from scipy.ndimage import binary_dilation

            # Create a binary mask where 100 (walls) are True
            wall_mask = self.occupancy_grid == 100

            # Dilate the mask by {self.map_dilate} pixels
            dilated_wall_mask = binary_dilation(wall_mask, iterations=self.map_dilate)

            # Apply the dilated mask back to the map, setting these pixels to 100
            self.occupancy_grid[dilated_wall_mask] = 100
            self.get_logger().info(f"Map obstacles dilated by {self.map_dilate} pixels")
        else: pass

        #Block of non valid regions for final challenge
        enter_pixel1 = self.xyth_to_pixel((self.enter_wall[0][0], self.enter_wall[0][1]), msg)
        enter_pixel2 = self.xyth_to_pixel((self.enter_wall[1][0], self.enter_wall[1][1]), msg)
        enter_wall_pixels = self.bresenham_line_supercover(enter_pixel1[0], enter_pixel1[1], enter_pixel2[0], enter_pixel2[1])
        
        self.enter_trajectory_viz.clear()
        for ix, lane_pixel in enumerate(enter_wall_pixels):
            xy_lane_cell = self.pixel_to_xyth((lane_pixel[0], lane_pixel[1]), msg)
            self.enter_trajectory_viz.addPoint(xy_lane_cell)
        self.enter_trajectory_viz.publish_viz()  

        exit_pixel1 = self.xyth_to_pixel((self.exit_wall[0][0], self.exit_wall[0][1]), msg)
        exit_pixel2 = self.xyth_to_pixel((self.exit_wall[1][0], self.exit_wall[1][1]), msg)
        exit_wall_pixels = self.bresenham_line_supercover(exit_pixel1[0], exit_pixel1[1], exit_pixel2[0], exit_pixel2[1])
        
        self.exit_trajectory_viz.clear()
        for ix, lane_pixel in enumerate(exit_wall_pixels):
            xy_lane_cell = self.pixel_to_xyth((lane_pixel[0], lane_pixel[1]), msg)
            self.exit_trajectory_viz.addPoint(xy_lane_cell)
        self.exit_trajectory_viz.publish_viz()  

        wall_pixels = enter_wall_pixels + exit_wall_pixels

        for wall_pixel in wall_pixels:
            self.occupancy_grid[wall_pixel[0]][wall_pixel[1]] = 100
        
        shell_wall_px1 = self.xyth_to_pixel((self.shell_run_wall[0][0], self.shell_run_wall[0][1]), msg)
        shell_wall_px2 = self.xyth_to_pixel((self.shell_run_wall[1][0], self.shell_run_wall[1][1]), msg)
        shell_wall_pixels = self.bresenham_line_supercover(shell_wall_px1[0], shell_wall_px1[1], shell_wall_px2[0], shell_wall_px2[1])
        
        self.shell_run_close_trajectory_viz.clear()
        for ix, shell_wall_pixel in enumerate(shell_wall_pixels):
            xy_shell_wall = self.pixel_to_xyth((shell_wall_pixel[0], shell_wall_pixel[1]), msg)
            self.shell_run_close_trajectory_viz.addPoint(xy_shell_wall)
        self.shell_wall_pixels = shell_wall_pixels

        # build the graph to run searchs on
        num_nodes = self.build_graph()
        if num_nodes > 0: self.get_logger().info(f"Graph constructed with {num_nodes} nodes")
        
        run_time = time.time() - start_time
        self.get_logger().info(f"Map import took {run_time} seconds")
        
        if False: # publishers for debug/development
            start_time = time.time()
            test_point = PointStamped()
            test_point.header.frame_id = "map"
            self.get_logger().info("Publishing test points")
            if False: # points to check relative map orientation
                for i in [1, int(self.map_height/2)]:
                    for j in [1, int(self.map_width/2)]:
                        xy = self.pixel_to_xyth((i,j,None), self.map_input)
                        self.get_logger().info("Publishing point at ({},{})".format(xy[0],xy[1]))
                        test_point.header.stamp = self.get_clock().now().to_msg()
                        test_point.point.x, test_point.point.y, test_point.point.z = xy[0], xy[1], 0.0
                        self.point_test.publish(test_point)
            if False:
                grid = self.occupancy_grid
                x_pix, y_pix = len(grid)-1, len(grid[0])-1
                r_divs, region = 5, 1
                for i in range(int(x_pix*(region/r_divs)), int(x_pix*((region+1)/r_divs))):
                    for j in range(y_pix):
                        is_edge = False
                        if grid[i][j] == 100: # r.random() > 0.5 and
                            for k in [-1, 1]:
                                if grid[i+k][j] == 0:
                                    is_edge: True
                                    break
                            for l in [-1, 1]:
                                if grid[i][j+l] == 0:
                                    is_edge = True
                                    break
                        # if grid[i][j] == 100 and r.random() > 0.98:
                        if is_edge and r.random() > 0.5:
                            xy = self.pixel_to_xyth((i+1,j-2,None), self.map_input)
                            # self.get_logger().info("Publishing point at ({},{})".format(xy[0],xy[1]))
                            test_point.header.stamp = self.get_clock().now().to_msg()
                            test_point.point.x, test_point.point.y, test_point.point.z = xy[0], xy[1], 0.0
                            time.sleep(0.01)
                            self.point_test.publish(test_point)
                            # self.point_test.publish(test_point)
                            time.sleep(0.02)
            if False: # publish graph nodes
                for coord in self.graph.keys():
                    if 800 <= coord[0] <= 1000 and 850 <= coord[1] <= 1050:
                        node_xy = self.pixel_to_xyth(coord, self.map_input)
                        graph_node = PointStamped()
                        graph_node.header.frame_id = "/map"
                        graph_node.header.stamp = self.get_clock().now().to_msg()
                        graph_node.point.x, graph_node.point.y, graph_node.point.z = node_xy[0], node_xy[1], 0.0
                        self.graph_pub.publish(graph_node)
                        time.sleep(0.01)
            run_time = time.time() - start_time
            self.get_logger().info(f"Finished publishing test points in {run_time} seconds")
        self.get_logger().info("Map received - waiting on poses")
        return

    def pose_cb(self, msg): # geometry_msgs/msg/PoseWithCovarianceStamped
        # Check if a map has been published yet
        if self.map_input == None:
            self.get_logger().info("Waiting on map")
            return
        
        # Reset path variables, confirm start pose is valid
        # self.rrt_tree, self.rrt_path, self.rrt_cost = None, None, np.inf
        # self.rrt_star_tree, self.rrt_star_path, self.rrt_star_cost = None, None, np.inf
        xyth = (msg.pose.pose.position.x, msg.pose.pose.position.y, 2*np.arccos(msg.pose.pose.orientation.w)-np.pi)
        indic_start = self.xyth_to_pixel(xyth, self.map_input)
        if self.occupancy_grid[indic_start[0]][indic_start[1]] != 0:
            update_status = Bool()
            update_status.data = True
            for i in range(3): self.traj_update_pub.publish(update_status)
            time.sleep(0.2)
            self.get_logger().info("Invalid start pose")
            return
        self.start_pose = indic_start
        
        # Check if a goal pose has been published, if yes start path planning
        if not self.goal_pose == None:
            self.get_logger().info(f"Planning path from ({self.start_pose[0]},{self.start_pose[1]}) to ({self.goal_pose[0]},{self.goal_pose[1]})")
            self.plan_path()
        else: self.get_logger().info("Waiting on goal pose")
        return

    def goal_cb(self, msg): # geometry_msgs/msg/PoseStamped
        # Check if a map has been published yet
        if self.map_input == None:
            self.get_logger().info("Waiting on map")
            return
        
        # Check if goal position is valid
        xyth =  (msg.pose.position.x, msg.pose.position.y, 2*np.arccos(msg.pose.orientation.w)-np.pi)
        indic_goal = self.xyth_to_pixel(xyth, self.map_input)
        if self.occupancy_grid[indic_goal[0]][indic_goal[1]] != 0:
            self.get_logger().info("Invalid goal pose")
            return
        self.goal_pose = indic_goal
        
        # Check if a start pose has been published, if yes start path planning
        if not self.start_pose == None:
            self.start_pose = self.current_pose
            self.get_logger().info(f"Planning path from ({self.start_pose[0]},{self.start_pose[1]}) to ({self.goal_pose[0]},{self.goal_pose[1]})")
            self.plan_path()
        else: self.get_logger().info("Waiting on start pose")
        return
    
    def odom_cb(self, odom_msg):
        reached_goal = False
        if self.start_pose != None:
            odom_euler = tf.euler_from_quaternion((odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
                                                    odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w))
            odom_pose = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_euler[2])
            new_pose = self.xyth_to_pixel(odom_pose, self.map_input)
            self.current_pose = new_pose
        if self.goal_pose != None and self.shell_paths:
            reached_goal = self.reached_goal_point()
            if reached_goal == True:
                self.get_logger().info("REACHED GOAL POINT")
                self.trajectory.points = self.shell_paths.pop(0)
                self.goal_pose = self.shell_goals.pop(0)
                self.traj_pub.publish(self.trajectory.toPoseArray())
                self.trajectory.publish_viz()
            else:
                self.get_logger().info("DID NOT REACH GOAL POINT")

    def reached_goal_point(self):
        current = np.array(self.current_pose)
        goal = np.array(self.goal_pose)
        distance = np.linalg.norm(current-goal)
        self.get_logger().info(f"DISTANCE: {distance}")
        if distance < 7.0:
            return True
        return False

    def shell_cb(self, shell_arr_msg):
        if self.map_input == None:
            self.get_logger().info("Waiting on map")
            return
        if self.start_pose == None:
            self.get_logger().info("Waiting on start pose")
            return 
        
        original_start_pose = self.start_pose
        self.goal_pose = self.start_pose
        shell_poses = shell_arr_msg.poses 
        shell_poses.append(original_start_pose)
        
        for ix, shell_pose in enumerate(shell_poses):
            if ix == 1:
                for shell_wall_px in self.shell_wall_pixels:
                    self.occupancy_grid[shell_wall_px[0]][shell_wall_px[1]] = 100
                self.shell_run_close_trajectory_viz.publish_viz()
                self.build_graph()
            if ix == 3:
                shell_goal = shell_pose
            else:
                shell_xyth = (shell_pose.position.x, shell_pose.position.y, 2*np.arccos(shell_pose.orientation.w)-np.pi)
                shell_goal = self.xyth_to_pixel(shell_xyth, self.map_input)
            if self.occupancy_grid[shell_goal[0]][shell_goal[1]] != 0:
                self.get_logger().info("Invalid shell pose")
                return
            self.start_pose = self.goal_pose
            self.goal_pose = shell_goal
            self.shell_goals.append(shell_goal)
            self.get_logger().info(f"Planning path from ({self.start_pose[0]},{self.start_pose[1]}) to ({self.goal_pose[0]},{self.goal_pose[1]})")
            if ix == 3:
                self.get_logger().info(f"Driving back to start")
            else:
                self.get_logger().info(f"Driving to shell {ix}")
            self.plan_path(shell_mode=True)
        
        #for shell_path in self.shell_paths:
        self.trajectory.points = self.shell_paths.pop(0)
        self.start_pose = original_start_pose
        self.goal_pose = self.shell_goals.pop(0)
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


    def plan_path(self, shell_mode=False):
        # reset variables, kill active functions, publish updating status
        self.update = True
        self.reached_goal = False
        update_status = Bool()
        update_status.data = True
        for i in range(3): self.traj_update_pub.publish(update_status)
        time.sleep(0.2)
        self.update = False
        
        ####### Call to A* goes here: #######    
        # run A* with timer
        start_time = time.time()
        path, cost = self.a_star(self.start_pose, self.goal_pose)
        run_time = time.time() - start_time
        if path == None or cost == np.inf:
            self.get_logger().info(f"A* did not find a solution after searching for {run_time:.2f} seconds")
            return

        self.get_logger().info(f"A* found a path of length {cost * self.map_res:.2f} meters in {run_time:.2f} seconds")
        # if shell_mode:
        #     shell_traj_points = [self.pixel_to_xyth(point, self.map_input) for point in path]
        #     self.shell_paths.append(shell_traj_points)
        # else:
        #     self.trajectory.points = [self.pixel_to_xyth(point, self.map_input) for point in path]
        #     self.traj_pub.publish(self.trajectory.toPoseArray())
        #     self.trajectory.publish_viz()
        
        if True:
            start_time = time.time()
            path_s, cost_s = self.smooth_path(path)
            run_time = time.time() - start_time
            self.get_logger().info(f"Path smoothing found a path of length {cost_s*self.map_res:.2f} meters in {run_time:.2f} seconds")
            if shell_mode:
                shell_traj_points = [self.pixel_to_xyth(point, self.map_input) for point in path_s]
                self.shell_paths.append(shell_traj_points)
            else:
                self.trajectory.points = [self.pixel_to_xyth(point, self.map_input) for point in path_s]
                self.traj_pub.publish(self.trajectory.toPoseArray())
                self.trajectory.publish_viz()
        
        # new_path = Path(path, cost)
        # new_path.start_pose = self.start_pose
        # new_path.goal_pose = self.goal_pose
        # self.path_list.append(new_path)

        
        
        ''' # Call to rrt
        # run rrt with timer
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
        # self.get_logger().info(str("RRT time to completion was {:.2f} seconds".format(end_time-start_time)))
        self.get_logger().info(f"RRT time to completion was {end_time-start_time:.2f} seconds")
        '''
        
        # publish updating status
        update_status.data = False
        for i in range(3): self.traj_update_pub.publish(update_status)
        
        ''' # Call to rrt*
        # run rrt* optimization on existing rrt tree if indicated
        if self.run_star and self.rrt_cost < np.inf:
            self.get_logger().info("Starting RRT* optimization")
            start_time = time.time()
            savings = self.rrt_star()
            if savings > 0:
                end_time = time.time()
                path, cost = self.rrt_star_path, self.rrt_star_cost
                self.get_logger().info(str("RRT* found a path that is {:.1f} percent shorter than RRT".format(100*savings/self.rrt_cost)))
                self.get_logger().info(str("Final path length was {:.2f} meters".format(self.rrt_star_cost * self.map_res)))
                self.get_logger().info(str("RRT* path optimization took {:.2f} seconds".format(end_time-start_time)))
                # self.get_logger().info("Publishing new path in:")
                self.trajectory.points = [self.pixel_to_xyth(point, self.map_input) for point in path]
                self.traj_pub.publish(self.trajectory.toPoseArray())
                self.trajectory.publish_viz()
        '''
        
        return
    
    def nearest_node(self, point_input):
        # get all the valid nodes at the corners of the square in which the point resides
        point, d, nearby, = np.array((point_input[0], point_input[1])), self.node_dist, []
        exes = [int(np.floor(point[0]/d)*d), int(np.ceil(point[0]/d)*d)]
        whys = [int(np.floor(point[1]/d)*d), int(np.ceil(point[1]/d)*d)]
        for x in exes:
            for y in whys:
                try:
                    if len(self.graph[(x,y)]) > 0: nearby.append(np.array([x,y]))
                except: pass
        
        # calculate the closest of the nearby nodes 
        best_node, best_dist = None, np.inf
        for node in nearby:
            this_dist = np.linalg.norm(node - point)
            if this_dist < best_dist:
                best_node = node
                best_dist = this_dist
        
        return tuple(best_node)
    
    def a_star(self, start_pose, goal_pose):
        start_node = self.nearest_node(start_pose)
        if start_node[0] == None:
            self.get_logger().info("Invalid start pose")
            return None, np.inf
        goal_node = self.nearest_node(goal_pose)
        if goal_node[0] == None:
            self.get_logger().info("Invalid goal pose")
            return None, np.inf
        goal_pt = np.array(goal_node) # goal as np array for calculations
        costs, visited = dict(), [] # initialize visited list and a dict to store cost to each node
        
        # initialize queue with starting node as first item
        max_q, costs[start_node] = len(self.graph), 0
        queue, item1 = PriorityQueue(maxsize=max_q), QueueItem(priority=0, item=[start_node])
        # item1.priority, item1.item = 0, [start_node]
        queue.put(item1)
        # self.get_logger().info(f"queue is currently {queue}")
        
        # retrieve and expand nodes from the priority queue
        while not queue.empty():
            current_item = queue.get()
            node_path = current_item.item # recover the path from the queue item
            node = node_path[-1]
            node_cost = costs[node]
            if node in visited: continue
            else:
                visited.append(node) # add the node to the visited list
                if False: # Publish the node being visited
                    node_xy = self.pixel_to_xyth(node, self.map_input)
                    visit_point = PointStamped()
                    visit_point.header.frame_id = "/map"
                    visit_point.header.stamp = self.get_clock().now().to_msg()
                    visit_point.point.x, visit_point.point.y, visit_point.point.z = node_xy[0], node_xy[1], 0.0
                    self.visiting_pub.publish(visit_point)
                    time.sleep(0.02)
                # self.get_logger().info(f"Running A* - current queue size is about {queue.qsize()}, current item is {current_item}, visited list has {len(visited)} nodes")
                # expand the node by adding all adjacent nodes to the queue
                for adjacent in self.graph[node].keys():
                    if adjacent in visited: continue # node already visited, don't add to queue
                    else: # add the adjacent node to the queue
                        heuristic = np.linalg.norm(goal_pt - np.array(adjacent)) # admissible heuristic -> euclidian distance to goal point
                        costs[adjacent] = node_cost + self.graph[adjacent][node]
                        this_cost = costs[adjacent]
                        this_priority = int(round((this_cost + heuristic)*1000000000.0))
                        this_path = node_path.copy() 
                        this_path.append(adjacent)
                        if adjacent == goal_node: return this_path, this_cost
                        this_item = QueueItem(this_priority, this_path)
                        queue.put(this_item)
                        # self.get_logger().info(f"Adding {adjacent} to queue with priority {this_priority} and path {this_path}")
                        # self.get_logger().info(f"Adding {adjacent} to queue with cost {this_cost} and heuristic {heuristic}")
        # exhausted queue without solution, return no path with infinite cost
        return None, np.inf
    
    def smooth_path(self, path):
        i, num_nodes, new_path, new_cost = 0, len(path), [path[0]], 0
        while i < num_nodes-1:
            i = self.lookahead(path, i, 0, num_nodes-1)
            new_path.append(path[i])
        for j in range(1, len(new_path)):
            new_cost += np.linalg.norm(np.array(new_path[j]) - np.array(new_path[j-1]))
        return new_path, new_cost
    
    def lookahead(self, path, index, forward_min, forward_max):
        # base case return
        if forward_min == forward_max: return index + forward_max
        
        # guard against index out of bounds
        f_min, f_max, max_i = forward_min, forward_max, len(path)-1
        if index + forward_max > max_i: f_max = max_i - index
        
        # calculate intermediate values
        f_avg = int(np.ceil((f_min+f_max)/2))
        start_pt, end_pt = np.array(path[index]), np.array(path[index+f_avg])
        segment_dist = np.linalg.norm(end_pt - start_pt)
        
        # recursive call
        if not self.collision(start_pt, end_pt, segment_dist): return self.lookahead(path, index, f_avg, f_max)
        else: return self.lookahead(path, index, f_min, f_avg-1)
            
            
            
        return
    
    def bresenham_line_supercover(self, x0, y0, x1, y1):
        ystep, xstep = 1, 1
        error = 0
        errorprev = 0
        x, y = x0, y0
        dx, dy = x1-x0, y1-y0
        line = [(x, y)]
        if (dy < 0): 
            ystep = -1
            dy = -dy
        if (dx < 0):
            xstep = -1
            dx = -dx 
        ddy = 2*dy
        ddx = 2*dx 
        if (ddx>=ddy):
            errorprev = dx
            error = dx
            for _ in range(dx):
                    x += xstep
                    error += ddy
                    if error >= ddx:
                        y += ystep
                        error -= ddx
                        if (error + errorprev < ddx):
                            line.append((x, y-ystep))
                        elif (error + errorprev > ddx):
                            line.append((x-xstep, y))
                        else:
                            line.append((x, y-ystep))
                            line.append((x-xstep, y))
                    line.append((x, y))
                    errorprev = error
        else:
             errorprev = dy
             error = dy 
             for _ in range(dy):
                y += ystep
                error += ddx
                if error > ddy:
                    x += xstep
                    error -= ddy
                    if (error + errorprev < ddy):
                        line.append((x-xstep, y))
                    elif(error + errorprev > ddy):
                        line.append((x, y-ystep))
                    else:
                        line.append((x-xstep, y))
                        line.append((x, y-ystep))
                line.append((x, y))
                errorprev = error
        return line

  
    ''' # Functions from RRT
    # def nearest_pt(self, points, target):
    #     ''returns the nearest existing node in the tree to a proposed new point''
    #     nearest, index_of, best_dist = None, None, np.inf
    #     for i, point in enumerate(points):
    #         if self.update: return
    #         this_dist = np.sqrt((target[0]-point[0][0])**2 + (target[1]-point[0][1])**2)
    #         if this_dist < best_dist:
    #             nearest, index_of, best_dist = point, i, this_dist
    #     return nearest, index_of, best_dist
    
    # def within_d(self, nodes, target):
    #     ''returns a list of nodes nearby the target, with each element
    #     being [index of node in RRT tree, distance to node from target]''
    #     max_d, nearby = self.rrt_step_size * self.dist_node_expand, []
    #     for i, node in enumerate(nodes):
    #         if self.update: return
    #         distance = np.sqrt((node[0][0]-target[0][0])**2+(node[0][1]-target[0][1])**2)
    #         if 0.01 < distance < max_d:
    #             nearby.append([i, distance])
    #     return nearby
    
    # def n_nearest(self, points, target):
    #     point_list, nearby_pts, n = [], [], self.num_node_expand
    #     for point in points:
    #         if point != target: point_list.append(point)
    #     while 0 < len(nearby_pts) < n:
    #         next = self.nearest_pt(point_list, target)
    #         point_list.remove(next)
    #         nearby_pts.append(next)
    #     return nearby_pts
    
    # def path_traceback(self, tree):
    #     ''Trace a solved RRT tree from the end node back to the root''
    #     # RRT stops adding nodes when it reaches the goal, so the
    #     # last eltement of the tree array should be the goal
    #     path, parent_ptr = [], len(tree)-1
    #     cost = tree[parent_ptr][2]
        
    #     # Only node w/no parent should be root - follow parent nodes
    #     # back to there and add them to the path in the process
    #     while not parent_ptr == None:
    #         if self.update: return
    #         path.insert(0, tree[parent_ptr][0])
    #         parent_ptr = tree[parent_ptr][1]
    #     return path, cost 

    # def rrt(self):
    #     ''#'implements rrt for a path planning node''#'
    #     r.seed()
        
    #     # local variables
    #     samp_exp = -1 # can cause bugs if raised above zero
    #     p, f, solved = 0, 0, False # points placed, failed attempts, solution found
    #     max_pts, max_step = self.max_rrt_points, self.rrt_step_size
    #     points = []
        
    #     # add root of tree - [(x, y, theta), index of parent, cost to reach]
    #     points.append([(self.start_pose[0], self.start_pose[1], self.start_pose[2]), None, 0.0])
        
    #     # rrt loop
    #     while (p+f)<(5*max_pts) and not solved:
    #         if self.update: return
    #         # self.get_logger().info(str("attempted samples: {}, of allowed samples: {}".format(p+f,5*max_pts)))
    #         # loop variables
    #         sample, nearest, near_idx, new_coord = None, None, None, None
            
    #         # set sample point with goal bias
    #         if r.random() < self.rrt_goal_bias: sample = (self.goal_pose[0], self.goal_pose[1], self.goal_pose[2])
    #         elif self.free_space_only: 
    #             while sample == None:
    #                 try_sample = (r.randint(-samp_exp, self.map_width+samp_exp), r.randint(-samp_exp, self.map_height+samp_exp), None)
    #                 if self.occupancy_grid[try_sample[0]][try_sample[1]] == 0: sample = try_sample
    #         else: sample = (r.randint(-samp_exp, self.map_width+samp_exp), r.randint(-samp_exp, self.map_height+samp_exp), None)
    #         if False: # Published the sampled coordinates
    #             sample_xy = self.pixel_to_xyth(sample, self.map_input)
    #             sample_point = PointStamped()
    #             sample_point.header.frame_id = "/map"
    #             sample_point.header.stamp = self.get_clock().now().to_msg()
    #             sample_point.point.x, sample_point.point.y, sample_point.point.z = sample_xy[0], sample_xy[1], 0.0
    #             self.sample_pub.publish(sample_point)
            
    #         # find the nearest existing point and its index in 'points'
    #         nearest, near_idx, segment_distance = self.nearest_pt(points, sample) # TODO implement a ?more efficient? f'n to find nearest point
    #         near_coord = nearest[0]
    #         if near_coord == sample: continue
    #         if False: # Publish the coordinates of the nearest existing node
    #             near_xy = self.pixel_to_xyth(near_coord, self.map_input)
    #             near_point = PointStamped()
    #             near_point.header.frame_id = "/map"
    #             near_point.header.stamp = self.get_clock().now().to_msg()
    #             near_point.point.x, near_point.point.y, near_point.point.z = near_xy[0], near_xy[1], 0.0
    #             self.nearest_pub.publish(near_point)
            
    #         # calculate the coordinates of the new point
    #         if segment_distance <= max_step:
    #             new_coord = sample
    #         else:
    #             angle = np.arctan2(sample[1]-near_coord[1], sample[0]-near_coord[0])
    #             new_coord = (near_coord[0]+round(max_step*np.cos(angle)), near_coord[1]+round(max_step*np.sin(angle)), None)
    #             segment_distance = np.sqrt((new_coord[0]-near_coord[0])**2 + (new_coord[1]-near_coord[1])**2)
    #         if False: # Publish the coordinates of the proposed new point
    #             prop_xy = self.pixel_to_xyth(new_coord, self.map_input)
    #             prop_point = PointStamped()
    #             prop_point.header.frame_id = "/map"
    #             prop_point.header.stamp = self.get_clock().now().to_msg()
    #             prop_point.point.x, prop_point.point.y, prop_point.point.z = prop_xy[0], prop_xy[1], 0.0
    #             self.proposed_pub.publish(prop_point)
            
    #         # check for collisions on path segment, if no collision then add point to tree and increment the points count
    #         if not self.collision(near_coord, new_coord, segment_distance):
    #             cost = points[near_idx][2]+segment_distance
    #             points.append([new_coord, near_idx, cost])
    #             p += 1
    #             if False: # Publish the coordinates of the added RRT node
    #                 xy_coord = self.pixel_to_xyth(new_coord, self.map_input)
    #                 ros_point = PointStamped()
    #                 ros_point.header.frame_id = "map"
    #                 ros_point.header.stamp = self.get_clock().now().to_msg()
    #                 ros_point.point.x, ros_point.point.y, ros_point.point.z = float(xy_coord[0]), float(xy_coord[1]), 0.0
    #                 self.point_pub.publish(ros_point)
    #         else: # otherwise increment the fail count
    #             f += 1
    #             continue
            
    #         # check if latest point is close enough to the goal pose, if yes then trace the path back to the head of the list and mark as solved
    #         if np.sqrt((self.goal_pose[0]-new_coord[0])**2 + (self.goal_pose[1]-new_coord[1])**2) <= 0.01:
    #             self.rrt_tree = points
    #             self.rrt_path, self.rrt_cost = self.path_traceback(points)
    #             solved = True
    #             self.get_logger().info(f"rrt tree size: {len(self.rrt_tree)} nodes")
    #             return solved
                
    #     # TODO actual error handling?
    #     if not solved: print("No path found in {} attempts".format(p+f))
    #     return solved # should be false if we get here
    
    # def rrt_star(self):
    #     points, adj_lists = [elt[:] for elt in self.rrt_tree], []
    #     updates, num_pts = 1, len(points)
        
    #     self.get_logger().info("Starting adjacency lists")
    #     start_time = time.time()
        
    #     # Builds adjacency lists, records distance for each edge
    #     for point in points:
    #         if time.time() - start_time > self.rrt_star_timeout:
    #             self.get_logger().info(f"RRT* timed out at {self.rrt_star_timeout} seconds")
    #             return 0
    #         nearby_adj = []
    #         nearby_nodes = self.within_d(points, point)
    #         for node in nearby_nodes:
    #             if self.update: return
    #             # self.get_logger().info(str("Node at index {} is coord {}, parent {}, cost {}".format(node[0], points[node[0]][0], points[node[0]][1], points[node[0]][2])))
    #             if not self.collision(points[node[0]][0], point[0], node[1]):
    #                 nearby_adj.append(node)
    #         adj_lists.append(nearby_adj)
    #     mid_time = time.time()
    #     self.get_logger().info(str("Adjacency lists finished in {:.2f} seconds".format(mid_time-start_time)))
        
    #     self.get_logger().info("Starting update loops")
    #     # Update loop - goes until there is nothing else to update
    #     while updates > 0:
    #         if self.update: return
    #         updates = 0
    #         for i in range(num_pts):
    #             best_parent, best_cost = points[i][1], points[i][2]
    #             for j, k in adj_lists[i]:
    #                 if self.update: return
    #                 # self.get_logger().info(str("".format()))
    #                 cost = points[j][2] + k
    #                 if cost < best_cost:
    #                     updates += 1
    #                     best_parent, best_cost = j, cost
    #             points[i][1], points[i][2] = best_parent, best_cost
    #     end_time = time.time()
    #     self.get_logger().info(str("Update loops finished in {:.2f} seconds".format(end_time-mid_time)))
        
    #     self.rrt_star_tree = points            
    #     self.rrt_star_path, self.rrt_star_cost = self.path_traceback(points)
    #     cost_savings = self.rrt_cost - self.rrt_star_cost
    #     return cost_savings
    '''

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
