import rclpy
from rclpy.node import Node

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

        self.current_pose = None
        self.goal_pose = None
        self.map_input = None
        # self.RRT_tree = dict({self.current_pose: None})

    
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
        #listens to current pose?
        self.get_logger().info('pose callback 1')
        self.get_logger().info(f"pose cb is {pose.pose.pose}")
        # pose.position 
        # pose.orientation
        xyth = self.pose_to_xyth(pose.pose)
        self.start_coord = self.xy_to_pixel(xyth, self.map_input)

        self.get_logger().info(f"start coord is {self.start_coord}")
        
        # self.current_pose = pose
        self.get_logger().info('pose callback 2')

    def goal_cb(self, msg):
        #listens to goal pose and use it to plan current
        self.get_logger().info('goal callback')
        self.get_logger().info(f"goal cb msg is {msg.pose}")

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
        # for i in range(n): 
        #     x_rand = self.sample()
        #     x_nearest = self.nearest(, x_rand)
        #     x_new = Steer(x_nearest, x_rand)
        #     if Obst

        self.get_logger().info('planning path')
        path = self.rrt(start_point, end_point, 2)[0]
        # self.get_logger().info('finished self.trajectory.points = [path planning')
        # self.get_logger().info(f"path is {path}")
        self.trajectory.points = path
        if self.trajectory.points == None:
            self.get_logger().info("couldn't find path :(")
            pass
        else:
            self.trajectory.points = [self.pixel_to_xy(point, map) for point in path]
            self.get_logger().info('found traj points')
    
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()

    def extend_path(self, start, end, max_diff=1.0): 
        """
        Step 1, construct a path that is a linear interpolation between `start` and `end`.
        At each step, the movement along x and along y should be both smaller than or equal to `max_diff`.

        Step 2: check for collisions.
        You should iterate over the configurations in the path generated in step 1,
        if a configuration triggers a collision with an obstacle,
        your algorithm should return the longest prefix path that does not have any collision.
        We have implemented this test for you. But make sure you read the code and understand how it works.
            
        """
        path = [start]
        print('extend path-------')
        x0, y0 = start[0], start[1]
        x1, y1 = end[0], end[1]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        # max_diff = 4
        # max_diff = 1
        max_diff = max_diff
        nx = math.ceil(dx/max_diff)
        ny = math.ceil(dy/max_diff)
        n = max(nx, ny)

        x_list = np.linspace(x0, x1, num = n+1)
        y_list = np.linspace(y0, y1, num = n+1)
        path = [RobotConfig(x_list[i], y_list[i]) for i in range(n+1)]
        
        assert path[-1] == end

        # Step 2: Collision checking.
        for i, config in enumerate(path):
            if self.collide(config):
                print('false, theres a collision -------')
                return False, path[:i]
        print('no collision!')
        return True, path


    def rrt(self, start, end,
        max_diff: float = 1.0,
        nr_iterations: int = 300,
        goal_sample_prob: float = 0.3):
        """Implement RRT algorithm to compute a path from the initial configuration
        to the target configuration. Your implementation should returns a list of
        robot configurations, starting from `problem.initial` and terminating at `proble.goal`.

        Similar to the definition in extend_path, at each step, the movement along x and along y should be both smaller than or equal to max_diff.
        You should use the extend_path utility that you just implemented to check whether there is a path from one node to another.

        Args:
        problem: an RobotPlanningProblem instance, used for configuration sampling and collision checking.
            `problem.initial` should be the first element of your return list.
            `problem.goal` should be the last element of your return list.
        max_diff: the maximum movement in x or y.
        nr_iterations: the number of iterations in RRT.
        goal_sample_prob: at each iteration, with prob p, you should try to construct a path from a node to the goal configuration.
            Otherwise, you should sample a new configuration and try to constrcut a path from a node to it.

        Returns:
        path: a list of RobotConfig as the trajectory. Return None if we can't find such a path after the specified number of iterations.
        rrt: an RRT object. The root should be the intial_config. There must also exist another node for goal_config.
        """
        # # finalpath = [start]
        # print('start rrt-------')
        # rrt_problem = RRT(RRTNode(start))
        # for k in range(nr_iterations):
        #     if random.random() < goal_sample_prob:
        #         q_rand = end
        #     else:
        #         q_rand = self.sample()

        # q_near = rrt_problem.nearest(q_rand)
        # success, path = self.extend_path(q_near.config, q_rand, max_diff)
        
        # print('start path checking_____')
        # print(success)
        # print(path)
        # child = q_near
        # for i in range(len(path) - 1):
        #     child = rrt_problem.extend(child, path[i+1])
        # if success:
        #     if q_rand == end:
        #         out = child.backtrace()
        #         return [(point.x, point.y) for point in out], None
        #         # return out, rrt_problem
        # return None, rrt_problem
        
        # return [(point.x, point.y) for point in path], None
        # finalpath = [start]
        print('start rrt-------')
        rrt_problem = RRT(RRTNode(start))
        
        #try direct path first
        success, path = self.extend_path(start, end, 100)
        if success:
            return [(point.x, point.y) for point in path], None
            # child = start
            # for i in range(len(path) - 1):
            #     child = rrt_problem.extend(child, path[i+1])
            # out = child.backtrace()
            # return out, None

        for k in range(nr_iterations):
            print(k)
            if random.random() < goal_sample_prob:
                q_rand = end
            else:
                q_rand = self.sample()
            q_near = rrt_problem.nearest(q_rand)
            success, path = self.extend_path(q_near.config, q_rand, max_diff)

            print('start path checking_____')
            child = q_near
            for i in range(len(path) - 1):
                child = rrt_problem.extend(child, path[i+1])
            if success:
                print('success at some point')
                if q_rand == end:
                    print('child backtracing')
                    out = child.backtrace()
                    # return [(point.x, point.y) for point in out], None
                    # return out, rrt_problem
                    return out, None
        return None, rrt_problem
        


class Obstacle:

    @abstractmethod
    def collide(self, config):
        ...


# @dataclasses.dataclass
# class Circle(Obstacle):
#     x: float
#     y: float
#     r: float

#     def collide(self, config):
#         return euclidean_distance(config, self) < self.r


RobotConfig = collections.namedtuple('RobotConfig', ['x', 'y'])
ExtentType = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclasses.dataclass(frozen=False, eq=False)
class RRTNode:
    """Node of a rapidly-exploring random tree.

    Each node is associated with a robot configuration.
    It also keeps track of
      - the parent in the tree.
      - a list of children nodes in the tree.

    If the node itself is the node for an RRT, node.parent will be set to None.
    If the node is a leaf node in an RRT, node.children will be an empty list.
    """

    config: RobotConfig
    children: List['RRTNode'] = dataclasses.field(default_factory=list)
    parent: 'RRTNode' = None

    def add_child(self, other: 'RRTNode') -> 'RRTNode':
        """Register another node as the child of the current node.

        Args:
          other: the other RRTNode.

        Returns:
          self
        """
        self.children.append(other)
        return self

    def attach_to(self, parent: 'RRTNode') -> 'RRTNode':
        """Attach the current node to another node. This function will:

          - set the parent of self to parent.
          - add self to parent.children.

        Args:
          parent: the parent node.

        Returns:
          self
        """
        self.parent = parent
        self.parent.add_child(self)
        return self

    def backtrace(self, config: bool = True) -> List[RobotConfig]:
        """Return the path from root to the current node.
        """
        path = []

        def dfs(x):
            if x.parent is not None:
                dfs(x.parent)
            path.append(x.config if config else x)

        dfs(self)
        return path

    def __repr__(self) -> str:
        return f"RRTNode(config={self.config}, parent={self.parent})"


class RRT:
    def __init__(self, roots: Union[RRTNode, Sequence[RRTNode]]):
        """The constructor of RRT takes two parameters:

        Args:
            problem: an instance of the robot planning problem.
                It will define the distance metric in the configuration space.
            roots: a single RRTNode or a list of RRTNode instances as the root(s) of the RRT.
        """
        if isinstance(roots, RRTNode):
            roots = [roots]
        self.roots = list(roots)
        self.size = len(self.roots)
        
    def distance(self, coord1, coord2): 
        #return distance between two robot configs 
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    # def sample(self) -> RobotConfig:
    #     """Generate a random sample in the robot configuration space."""
    #     return RobotConfig(random.uniform(self.extent[0][0], self.extent[1][0]),
    #                        random.uniform(self.extent[0][1], self.extent[1][1]))


    def extend(self, parent: RRTNode, child_config: RobotConfig) -> RRTNode:
        """Create a new RRTNode from a robot configuration and set its parent.

        Basically, it creates a new RRTNode instance from child_config and attach this
        new node to the `parent` node. We recommend you to use this function
        instead of doing it yourself because it will help you keep track of the size
        of the RRT, which may be useful for your debugging purpose.

        Args:
          parent: the parent node (in the current RRT).
          child_config: the robot configuration of the new node.

        Returns:
          child: the new node.
        """
        child = RRTNode(child_config).attach_to(parent)
        self.size += 1
        return child

    def nearest(self, config: RobotConfig) -> RRTNode:
        """Return the RRTNode in the tree that is closest to the target
        configuration. We strongly suggest you to take a look at the function
        definition to see the example usage of function `traverse_rrt_bfs` and
        `problem.distance`.

        Args:
          config: the query robot config.

        Returns:
          best_node: a node in the RRT that is closest to the query configuration
            based on the distance metric in the configuration space.
        """
        best_node, best_value = None, np.inf
        # print(self.roots)
        print(len(self.roots))
        for node in self.traverse_rrt_bfs(self.roots):
            distance = self.distance(node.config, config)
            if distance < best_value:
                best_value = distance
                best_node = node
        return best_node
    
    def traverse_rrt_bfs(self, nodes: Sequence[RRTNode]) -> List[RRTNode]:
        """Run breadth-first search to return all RRTNodes that are descendants of
        the input list of RRTNodes.

        Args:
        nodes: a list of RRTNodes as the seed nodes.

        Returns:
        a list of RRTNodes that are descendants of the input lists (including the input list).
        """
        queue = list(nodes)
        results = []
        while queue:
            x = queue.pop(0)
            results.append(x)
            queue.extend(x.children)

        return results
        


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
