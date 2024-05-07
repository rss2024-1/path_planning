import rclpy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray, PoseStamped, Point, Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
import tf_transformations as tf
from std_msgs.msg import Header, Bool
import time

from .utils import LineTrajectory

class Pose():
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")
        self.declare_parameter('segment_resolution', 0.1)
        
        
        self.seg_res = self.get_parameter('segment_resolution').value
        # self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        # self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        print("hello")
        self.odom_topic = "/odom"
        self.drive_topic = "/drive"
        self.focal_point = "/focal_point"
        self.debug_robot_pose = "/debug/pose"
        self.debug_eta_angle = "/debug/eta"
        # self.debug_point = "/debug_point"

        self.lookahead = 1.0  # FILL IN #
        self.speed = 4.0  # FILL IN #
        # self.speed = 0.2  # FILL IN #
        self.wheelbase_length = 0.381  # FILL IN : 15in ish??#

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(
            PoseArray,
            "/trajectory/current",
            self.trajectory_callback,
            1
            )
        
        self.update_status_sub = self.create_subscription(
            Bool,
            "/trajectory/updating",
            self.status_callback,
            1
            )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.drive_topic,
            1
            )
        
        self.odom_sub = self.create_subscription(
            Odometry, 
            self.odom_topic, 
            self.pose_callback, 
            1
            )
        
        self.focal_point_pub = self.create_publisher(
            Marker,
            self.focal_point,
            1
            )
        
        # self.debug_pub = self.create_publisher(Marker,
        #                                        self.debug_point,
        #                                        1)
        
        # self.line_pub = self.create_publisher()
        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.debug_robot_pose,
            1
            )
        
        self.eta_pub = self.create_publisher(
            PoseStamped,
            self.debug_eta_angle,
            1
            )

        self.initialized_traj = False
        self.planner_updating = False
        self.segments = None
        self.position = None
        
        
    def status_callback(self, msg):
        self.planner_updating = msg.data
        
    def find_min_distance_for_segment(self, row):
        """
        Helper function for Step 1: Find minimum distance for EACH segment
        """
        pt1_x, pt1_y, pt2_x, pt2_y, segment_len, _ = row
        pt1 = np.array([[pt1_x, pt1_y]])
        pt2 = np.array([[pt2_x, pt2_y]])
        robot = np.array([[self.position]])

        if segment_len <= 0.001:
            return np.linalg.norm(pt1 - robot) # any of the points works because segment has no len

        t = max(0, min(1, np.dot(robot.squeeze()-pt1.squeeze(), pt2.squeeze()-pt1.squeeze())/segment_len))
        projection = pt1 + t*(pt2-pt1)
        return np.linalg.norm(projection - robot)

    def find_min_distance_robot_to_segment(self, pose):
        """
        Step 1: Find segment index WITH the minimum distance, return the index
        """
        self.position = np.array((pose.x, pose.y))

        distances = np.apply_along_axis(self.find_min_distance_for_segment, 1, self.segments)
        min_distance_index = np.argmin(distances)
        return min_distance_index

    def segment_iteration(self, pose, min_distance_index):
        """
        Step 2 of function: Iterate through every segment starting at min_distance_index
        and see if we can find a valid goal point for each one of them
        """
        # self.get_logger().info(f"min_distance_index: {min_distance_index}")
        # self.get_logger().info("starting segment iteration")
        for i in range(min_distance_index, len(self.trajectory.points)):
            dist_ahead = np.linalg.norm(self.trajectory.points[i]-self.trajectory.points[min_distance_index])
            if dist_ahead >= self.lookahead or i == len(self.trajectory.points)-1:
                return self.trajectory.points[i]
        self.get_logger().info(f"No solution found for segment {min_distance_index}")

    
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

    def drive_angle(self, pose, goal_point):
        dx = float(goal_point[0] - pose.x)
        dy = float(goal_point[1] - pose.y)
        goal_angle = np.arctan2(dy, dx)
        robot_angle = pose.angle
        # self.get_logger().info(f"goal angle is {goal_angle}, robot angle is {robot_angle}")

        
        eta = goal_angle - robot_angle
        if abs(eta) <= np.pi: pass # self.get_logger().info(f"eta is {eta}")
        elif eta > np.pi:
            # old_eta = eta
            eta -= 2*np.pi
            # self.get_logger().info(f"eta was overvalued. old eta was {old_eta}, new eta is {eta}")
        else:
            # old_eta = eta
            eta += 2*np.pi
            # self.get_logger().info(f"eta was undervalued. old eta was {old_eta}, new eta is {eta}")
            
        eta_marker = eta + robot_angle
        eta_cap = np.clip(eta, -np.pi/2, np.pi/2)
        # self.get_logger().info(f"eta is {eta}, clipped eta is {eta_cap}")
        
        theta = np.arctan((2*self.wheelbase_length*np.sin(eta_cap))/(self.lookahead))
        # self.get_logger().info(f"theta is {theta}")
        return theta, eta_marker
    
    def debug_marker():
        # self.get_logger().info("Before Publishing start point")
        # self.get_logger().info("Publishing goal point")
        marker = Marker()
        marker.header = self.make_header("/map")
        marker.ns = "/debug_point"
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
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            # delete marker
            marker.action = 2
        
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
        if not self.initialized_traj: return
        
        odom_euler = tf.euler_from_quaternion((odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
                                               odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w))
        pose = Pose(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_euler[2])
        
        ### STEP 1
        closest_segment_index = self.find_min_distance_robot_to_segment(pose)

        ### STEP 2
        goal_point = self.segment_iteration(pose, closest_segment_index)
        
        ### STEP 3
        if goal_point is not None:
            self.publish_marker(goal_point)
            driving_angle, eta = self.drive_angle(pose, goal_point)
            if False:
                debug_pose = PoseStamped()
                debug_pose.pose = odom_msg.pose.pose
                debug_pose.header = odom_msg.header
                self.pose_pub.publish(debug_pose)
                debug_eta = PoseStamped()
                qe = Quaternion()
                qe.x, qe.y, qe.z, qe.w = tf.quaternion_from_euler(0.0, 0.0, eta)
                debug_eta.pose.position.x = pose.x
                debug_eta.pose.position.y = pose.y
                debug_eta.pose.position.z = 0.0
                debug_eta.pose.orientation = qe
                debug_eta.header = self.make_header("/map")
                self.eta_pub.publish(debug_eta)
            # self.get_logger().info(str("Pose {}, Goal Point {}".foramt(pose, goal_point)))

            ### STEP 4
            drive_msg = AckermannDriveStamped()
            new_speed = self.speed
            goal_dist = np.linalg.norm(goal_point - np.array((pose.x, pose.y)))
            # self.get_logger().info(f"Current dist to goal is {goal_dist:.2f}")
            if goal_dist < 0.9*self.lookahead:
                new_speed = 2*goal_dist
            if self.planner_updating or goal_dist < 0.1: new_speed = 0.0
            drive_msg.drive.speed = new_speed
            drive_msg.drive.steering_angle = driving_angle
            # self.get_logger().info(f"Driving angle is: {driving_angle}")
            self.drive_pub.publish(drive_msg)
            
            # time.sleep(0.5)


    def trajectory_callback(self, msg):
        # self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")
        self.initialized_traj = False
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        if len(self.trajectory.points) == 0:
            return
        
        ### PRE-STEP 1: Preprocess trajectory.points:
        init_point, scnd_point, last_point = np.array(self.trajectory.points[0]), np.array(self.trajectory.points[1]), np.array(self.trajectory.points[-1])
        init_vector = scnd_point - init_point
        init_angle = np.arctan2(init_vector[1], init_vector[0])
        filtered_points = np.asarray([init_point])
        # self.get_logger().info(str("filtered points so far: {}".format(filtered_points)))
        
        prev_point, prev_angle = scnd_point, init_angle
        
        for point in self.trajectory.points[2:]:
            this_point = np.array(point)
            segment_vect = this_point - prev_point
            segment_angle = np.arctan2(segment_vect[1], segment_vect[0])
            if abs(segment_angle - prev_angle) >= 0.000001:
                filtered_points = np.concatenate((filtered_points, [prev_point]), axis=0)
                prev_angle = segment_angle
            prev_point = this_point
        filtered_points = np.concatenate((filtered_points, [last_point]), axis=0)
        # self.get_logger().info(str("filtered points so far: {}".format(filtered_points)))
        
        reconstructed_traj, resolution, prev_point = np.asarray([init_point]), self.seg_res, filtered_points[0]
        for point in filtered_points[1:]:
            segment_length = np.linalg.norm(point - prev_point)
            segment_divs = int(round(segment_length/resolution))
            xlist, ylist = np.linspace(prev_point[0], point[0], segment_divs), np.linspace(prev_point[1], point[1], segment_divs+1)
            for i in range(1, len(xlist)):
                reconstructed_traj = np.concatenate((reconstructed_traj, [[xlist[i], ylist[i]]]), axis=0)
            prev_point = point
        # self.get_logger().info(str("filtered points so far: {}".format(reconstructed_traj)))

        self.trajectory.points = reconstructed_traj
        starts, ends = np.array(reconstructed_traj[:-1]), np.array(reconstructed_traj[1:])
        seg_lengths = np.vstack([ends - starts])
        self.segments = np.hstack((np.array(reconstructed_traj[:-1]), np.array(reconstructed_traj[1:]), seg_lengths))
        trajectory_length = len(self.trajectory.points)
        self.initialized_traj = True
        # self.get_logger().info(f"Length of the trajectory is: {trajectory_length}")

def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()