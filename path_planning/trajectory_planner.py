import rclpy
import numpy as np
import random as r
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('rrt_attempts', 10e4)
        self.declare_parameter('rrt_points', 10e3)
        self.declare_parameter('rrt_step', 10.0)
        self.declare_parameter('rrt_bias', 0.05)
        self.declare_parameter('car_size', 2.0)
        self.declare_parameter('goal_dist', 1.0)
        self.declare_parameter('goal_angle', 0.3)

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.max_rrt_attempts = self.get_parameter('rrt_attempts').value
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
        self.map_grid = np.array()
        self.map_width = -1
        self.map_height = -1
        self.goal_pose = [-1.0, -1.0, 0.0]
        self.crnt_pose = [-1.0, -1.0, 0.0]
        
        self.path = None


    def angle_to(p2,p1): # angle to point 2 from point 1
        return np.arctan2(p2[0]-p1[0], p2[1]-p1[1])
    
    def dist_to(p2,p1): # distance to point 2 from point 1
        return np.sqrt((p2[0]-p1[0])^2 + (p2[1]-p1[1])^2)
    
    def nearest_pt(self, points, target):
        nearest, index_of, best_dist = None, np.inf
        for i, point in enumerate(points):
            this_dist = self.dist_to(target, point)
            if this_dist < best_dist:
                nearest, best_dist = point, this_dist
        return nearest, index_of
    
    def collision(self, current, next):
        return
    
    def rrt(self):
        # local variables
        p, a, solved = 0, 0, False
        grid = np.copy(self.map_grid)
        max_pts, max_att, max_step = self.max_rrt_points, self.max_rrt_attempts, self.rrt_step_size
        points = []
        
        # head of tree
        points[0] = [(self.crnt_pose[0], self.crnt_pose[1]), None]
        
        # rrt loop
        while p < max_pts and a < max_att and not solved:
            # loop variables
            sample, nearest, near_idx, new = None, None, None, None
            
            # set sample point with goal bias
            if r.random() < self.rrt_goal_bias:
                for j in range(0, 2): sample[j] = self.goal_pose[j]
                sample = (self.goal_pose[0], self.goal_pose[1])
            else: sample = (r.randint(0, self.map_width), r.randint(0, self.map_height))
            
            # find the nearest existing point
            nearest, near_idx = self.nearest_pt(points, sample) # TODO implement a ?more efficient? f'n to find nearest point
            
            # calculate the new point
            if self.dist_to(sample, nearest) <= max_step:
                new = sample
            else:
                angle = self.angle_to(sample, nearest)
                new = (nearest[0]+round(max_step*np.cos(angle)), nearest[1]+round(max_step*np.sin(angle)))
            
            # check for collisions on path segment, if ok then add point to tree. increment loop counts as appropriate
            if collision(nearest, new): # TODO implement collision checking f'n
                continue
            else:
                points.append([new, near_idx])
                p += 1
            a += 1
            
            # check if latest point is close enough to the goal pose, if yes then trace the path back to the head of the list
            if self.dist_to(self.goal_pose, new) <= self.goal_distance_th:
                path = []
                parent_ptr = len(points)-1
                while not parent_ptr == None:
                    path.insert(0, points[parent_ptr][0])
                    parent_ptr = points[parent_ptr][1]
                self.path = path
                solved = True
                
        # TODO actual error handling
        if self.path == None: print("No path found in {} attempts with {} points".format(a, p))


    def map_cb(self, msg): # nav_msgs/msg/OccupancyGrid.msg
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        npgrid = np.empty((self.map_width, self.map_height), np.int8)
        for i, occ_val in enumerate(msg.data):
            yx = divmod(i, self.map_width)
            y, x = yx[0], yx[1]
            npgrid[x][y] = occ_val
        self.map_grid = npgrid # TODO implement f'n to add clearance around occupied map locations

    def pose_cb(self, pose): # geometry_msgs/Pose.msg
        self.crnt_pose = (pose.position.x, pose.position.y, 2*np.arccos(pose.orientation.w)-np.pi)

    def goal_cb(self, msg): # geometry_msgs/msg/PoseStamped
        self.goal_pose =  (msg.pose.position.x, msg.pose.position.y, 2*np.arccos(msg.pose.orientation.w)-np.pi)

    def plan_path(self, start_point, end_point, map):
        #TODO implement RRT*
        
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()
        

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
