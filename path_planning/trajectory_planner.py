import rclpy
from rclpy.node import Node
import numpy as np
import tf_transformations as tf

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

        self.map = None
        self.map_resolution = None
        self.map_origin_orientation = None
        self.map_origin_poistion = None
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

    def pose_cb(self, pose):
        if self.map_resolution == None:
            return
        self.get_logger().info("pose callback")
        start_x, start_y = self.map_to_pixel(pose.pose.pose.position.x, pose.pose.pose.position.y)
        self.start_point = (start_x, start_y)
        self.plan_path(self.start_point, self.end_point, self.map)

    def goal_cb(self, msg):
        if self.map_resolution == None:
            return
        self.get_logger().info("goal callback")
        end_x, end_y = self.map_to_pixel(msg.pose.position.x, msg.pose.position.y)
        self.end_point = (end_x, end_y)
        self.plan_path(self.start_point, self.end_point, self.map)

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
