import rclpy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
import tf_transformations as tf
from std_msgs.msg import Header

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        # self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        # self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.odom_topic = "/odom"
        self.drive_topic = "/drive"
        self.focal_point = "/focal_point"

        self.lookahead = 1  # FILL IN #
        self.speed = 1.0  # FILL IN #
        self.wheelbase_length = 0.381  # FILL IN : 15in ish??#

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)

        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                                    self.drive_topic,
                                                    1)
        self.odom_sub = self.create_subscription(Odometry, 
                                                 self.odom_topic, 
                                                 self.pose_callback, 
                                                 1)
        
        self.focal_point_pub = self.create_publisher(Marker,
                                                    self.focal_point,
                                                    1)

        self.segments = None
        
    def find_min_distance_for_segment(self, row):
        """
        Helper function for Step 1: Find minimum distance for EACH segment
        """
        pt1_x, pt1_y, pt2_x, pt2_y, robot_x, robot_y = row
        pt1 = np.array([[pt1_x, pt1_y]])
        pt2 = np.array([[pt2_x, pt2_y]])
        robot = np.array([[robot_x, robot_y]])

        segment_len = np.linalg.norm(pt1 - pt2)
        if segment_len == 0:
            return np.linalg.norm(pt1 - robot) # any of the points works because segment has no len

        t = max(0, min(1, np.dot(robot.squeeze()-pt1.squeeze(), pt2.squeeze()-pt1.squeeze())/segment_len))
        projection = pt1 + t*(pt2-pt1)
        return np.linalg.norm(projection - robot)

    def find_min_distance_robot_to_segment(self, pose):
        """
        Step 1: Find segment index WITH the minimum distance, return the index
        """
        traj_pts = self.trajectory.points
        position = pose.x, pose.y
        robot_pts_array = np.vstack([np.array(position)] * (len(traj_pts) - 1))
        segments = np.hstack((np.array(traj_pts[:-1]), np.array(traj_pts[1:]), robot_pts_array))

        self.segments = segments

        distances = np.apply_along_axis(self.find_min_distance_for_segment, 1, segments)
        min_distance_index = np.argmin(distances)

        return min_distance_index

    def segment_iteration(self, pose, min_distance_index):
        """
        Step 2 of function: Iterate through every segment starting at min_distance_index
        and see if we can find a valid goal point for each one of them
        """
        self.get_logger().info(f"min_distance_index: {min_distance_index}")
        for i in range(min_distance_index, min(min_distance_index + 5, len(self.segments))):
            segment = self.segments[i]
            soln = self.compute_math_for_segment(pose, segment)
            if soln is not None:
                self.get_logger().info(f"soln found segment # {i}")
                return soln
        # self.get_logger().info(f"No solution found for segment {segment}")

    
    def check_angle(self, robot_angle, check_point, robot_point):
        """Helper function for step 2"""
        v = check_point-robot_point

        rotation_matrix = np.array([[np.cos(-robot_angle), -np.sin(-robot_angle)],
                                    [np.sin(-robot_angle), np.cos(-robot_angle)]])
        # self.get_logger().info(f"Value of rotation_matrix: {rotation_matrix}")
        # self.get_logger().info(f"Shape of rotation_matrix: {rotation_matrix.shape}")
        # self.get_logger().info(f"Value of v: {v}")
        # self.get_logger().info(f"Shape of v: {v.shape}")
        w = rotation_matrix @ v.T
        return w[0][0]>0


    def compute_math_for_segment(self, pose, segment):
        """
        Helper function for Step 2
        """
        pt1_x, pt1_y, pt2_x, pt2_y, _, _ = segment
        pt1 = np.array([[pt1_x, pt1_y]])
        pt2 = np.array([[pt2_x, pt2_y]])

        robot_x, robot_y, angle = pose.x, pose.y, pose.angle

        # Calculate the slope and y-intercept for the line
        slope = (pt2_y - pt1_y) / (pt2_x - pt1_x)  # add EPSILON to avoid division by zero
        y_intercept = pt1_y - slope * pt1_x

        from sympy import symbols, Eq, solve

        x, y = symbols('x y')

        eq1 = Eq( (x - robot_x)**2 + (y - robot_y)**2, self.lookahead**2)
        eq2 = Eq(y - (slope * x + y_intercept), 0)

        solutions = solve((eq1,eq2), (x, y))
        decimal_solutions = np.array([(sol[0].evalf(), sol[1].evalf()) for sol in solutions])

        for soln in decimal_solutions:
            if soln[0].is_real and soln[1].is_real:
                if self.check_angle(angle, soln, np.array([[robot_x, robot_y]])):
                    return soln
        # self.get_logger().info(f"Pose: x={pose.x}, y={pose.y}, theta={pose.angle}")
        # self.get_logger().info(f"Segment: {segment}")
        # self.get_logger().info(f"Equations: {eq1}, {eq2}")
        return None

    # def drive_angle(self, pose, goal_point):
    #     """
    #     Step 3: given a goal point and a pose, drive to it
    #     """
    #     dx = goal_point[0] - pose.x
    #     dy = goal_point[1] - pose.y
    #     # self.get_logger().info(f"Value of dx: {dx}")
    #     # self.get_logger().info(f"Value of dy: {dy}")
    #     # self.get_logger().info(f"{type(dx)}")
    #     return np.arctan(float(dy)/float(dx))

    def drive_angle(self, pose, goal_point):
        dx = float(goal_point[0] - pose.x)
        dy = float(goal_point[1] - pose.y)
        robot_angle = pose.angle
        rotation_matrix = np.array([[np.cos(-robot_angle), -np.sin(-robot_angle)],
                                    [np.sin(-robot_angle), np.cos(-robot_angle)]])
        # self.get_logger().info(f"Value of rotation_matrix: {rotation_matrix}")
        # self.get_logger().info(f"Shape of rotation_matrix: {rotation_matrix.shape}")
        # self.get_logger().info(f"Value of v: {v}")
        # self.get_logger().info(f"Shape of v: {v.shape}")
        w = rotation_matrix @ np.array([[dx], [dy]])

        th = np.arctan(w[0][0]/w[1][0])
        return th
        # lookahead_dist = np.sqrt(dx**2 + dy**2)
        # return np.arctan(2*self.wheelbase_length*np.sin(th)/lookahead_dist)
    
    def publish_marker(self, point, duration=0.0, scale=0.1):
        should_publish = True
        # self.get_logger().info("Before Publishing start point")
        # self.get_logger().info("Publishing goal point")
        marker = Marker()
        marker.header = self.make_header("/map")
        marker.ns = "/goal_point"
        marker.id = 0
        marker.type = 2  # sphere
        marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
        if should_publish:
            marker.action = 0
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            # delete marker
            marker.action = 2

        self.focal_point_pub.publish(marker)

    def make_header(self, frame_id, stamp=None):
        if stamp == None:
            stamp = self.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header


    def pose_callback(self, odom_msg):
        odom_euler = tf.euler_from_quaternion((odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w))
        
        class Pose():
            def __init__(self, x, y, angle):
                self.x = x
                self.y = y
                self.angle = angle
        pose = Pose(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_euler[2])

        if len(self.trajectory.points) == 0:
            return
        
        ### STEP 1
        closest_segment_index = self.find_min_distance_robot_to_segment(pose)

        ### STEP 2
        goal_point = self.segment_iteration(pose, closest_segment_index)

        
        ### STEP 3
        if goal_point is not None:
            self.publish_marker(goal_point)
            driving_angle = self.drive_angle(pose, goal_point)

            ### STEP 4
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = self.speed
            drive_msg.drive.steering_angle = driving_angle
            self.get_logger().info(f"Driving angle is: {driving_angle}")
            self.drive_pub.publish(drive_msg)


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        # converting trajectory to LineTrajectory format???
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        # flag, but where used??
        self.initialized_traj = True

    ### STEP 1: Get current position of vehicle

def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()