import rclpy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 1  # FILL IN #
        self.speed = 1  # FILL IN #
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

        self.segments = None
        
    def find_min_distance_for_segment(self, row):
        """
        Helper function for Step 1: Find minimum distance for EACH segment
        """
        pt1_x, pt1_y, pt2_x, pt2_y, robot_x, robot_y = row
        pt1 = np.array([[pt1_x, pt1_y]])
        pt2 = np.array([[pt2_x, pt2_y]])
        robot = np.array([[robot_x, robot_y]])

        segment_len = np.linalg.norm(pt1, pt2)
        if segment_len == 0:
            return np.linalg.norm(pt1, robot) # any of the points works because segment has no len
        t = max(0, min(1, np.dot(robot-pt1, pt2-pt1)/segment_len))
        projection = pt1 + t*(pt2-pt1)
        return np.linalg.norm(robot, projection)

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
        for i in range(min_distance_index, min_distance_index + 5):
            segment = self.segments[i]
            soln = self.compute_math_for_segment(pose, segment)
            if soln:
                return soln
        self.get_logger().info(f"No solution found for segment {segment}")

    
    def check_angle(self, robot_angle, check_point, robot_point):
        """Helper function for step 2"""
        v = check_point-robot_point

        rotation_matrix = np.array([[np.cos(-robot_angle), -np.sin(-robot_angle)],
                                    [np.sin(-robot_angle), np.cos(-robot_angle)]])
        w = rotation_matrix @ v
        return w>0


    def compute_math_for_segment(self, pose, segment):
        """
        Helper function for Step 2
        """
        pt1_x, pt1_y, pt2_x, pt2_y = segment
        pt1 = np.array([[pt1_x, pt1_y]])
        pt2 = np.array([[pt2_x, pt2_y]])

        robot_x, robot_y, angle = pose.x, pose.y, pose.orientation

        from scipy import optimize

        # Calculate the slope and y-intercept for the line
        slope = (pt2_y - pt1_y) / (pt2_x - pt1_x)  # add EPSILON to avoid division by zero
        y_intercept = pt1_y - slope * pt1_x

        from sympy import symbols, Eq, solve

        x, y = symbols('x y')

        eq1 = Eq( (x - robot_x)**2 + (y - robot_y)**2, )
        eq2 = Eq(y - (slope * x + y_intercept), 0)

        solutions = solve((eq1,eq2), (x, y))
        print(solutions)
        decimal_solutions = np.array([(sol[0].evalf(), sol[1].evalf()) for sol in solutions])

        for soln in decimal_solutions:
            if self.check_angle(angle, soln, np.array([[robot_x, robot_y]])):
                return soln
        return False

    def drive_angle(self, pose, goal_point):
        """
        Step 3: given a goal point and a pose, drive to it
        """
        dx = goal_point[0][0] - pose.x
        dy = goal_point[0][1] - pose.y
        return np.arctan(dy/dx)

    def pose_callback(self, odom_msg):
        pose = odom_msg.pose.pose
        position = pose.position
        orientation = pose.orientation

        if len(self.trajectory.points) == 0:
            return
        
        ### STEP 1
        closest_segment_index = self.find_min_distance_robot_to_segment(position)

        ### STEP 2
        goal_point = self.segment_iteration(pose, closest_segment_index)
        
        ### STEP 3
        driving_angle = self.drive_angle(pose, goal_point)

        ### STEP 4
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = driving_angle
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