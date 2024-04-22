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
        self.follower_sub = self.create_subscription(Odometry, 
                                                     self.odom_topic, 
                                                     self.pose_callback, 
                                                     1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
    
    def closest_point(self, current_position):
        # mentioned something about segments... not sure why we'd need segments...
        # casting literally everything to np just in case
        current_position = np.array([current_position.x, current_position.y])
        # pts in the trajectory class rep as List[Tuple[float, float]], need to cast ig
        trajectory_pts = np.array([np.asarray(point) for point in self.trajectory.points])
    
        distances = np.linalg.norm(trajectory_pts - current_position, axis=1)
        # get direct /euclidean distance btwn current pos and all trajectory pts
        closest_idx = np.argmin(distances)
        closest_dist = distances[closest_idx] #hmm do we need this??? just keep for now :p
        closest_point = self.trajectory.points[closest_idx]
        return closest_dist, closest_point, closest_idx

    def is_progressing_along_path(self, pos, point):
        ## not really sure if this is what they meant by making sure that 
        ## we're going forwards in the path... may need to check this out...
        pos_orientation = np.arctan2(pos.y, pos.x)
        pt_orientation = np.arctan2(point[1]-pos.y, point[0]-pos.x)
        return np.abs(pt_orientation-pos_orientation) < np.pi/2
    
    def calculate_curvature(p1, p2, p3):
        # to get curve at p2 from 3 sample pts, courtesy of chatGPT :D
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Calculate the lengths of the sides of the triangle formed by the points
        a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

        # Calculate the radius of the circumcircle of the triangle
        # If the points are collinear, return a large value to indicate a straight line
        if a * b * c == 0:
            return 1e6
        else:
            radius = (a * b * c) / np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))

        # Calculate the curvature as the reciprocal of the radius
        curvature = 1.0 / radius
        return curvature


    def lookahead_point(self, current_position):
        closest_dist, closest_point, closest_idx = self.closest_point(current_position)
        # create a circle around the car
        circle_center = np.array([current_position.x, current_position.y])
        circle_radius = self.lookahead # lookahead distance, change upstairs

        ## adjusting radius based on path curvature intensity:
        if 0 < closest_idx < len(self.trajectory.points)-1:
            p1 = self.trajectory.points[closest_idx-1]
            p2 = self.trajectory.points[closest_idx]
            p3 = self.trajectory.points[closest_idx+1] 
            # look at very small step from the current closest point and see how extreme the curve is
            curvature = self.calculate_curvature(p1, p2, p3)
            if curvature < 0.1: # pretty much straight...
                circle_radius = self.lookahead*2
            else: # terrible curving, need to scale down
                circle_radius = self.lookahead/2

        # look for the lookahead pt, intersection between circle and piecewise linear
        # direction vector of the circle lookahead
        circle_direction = np.diff(np.array(self.trajectory.points), axis=0)
        circle_dir_norm = np.linalg.norm(circle_direction, axis=1)
        circle_dir_unit = circle_direction / np.column_stack([circle_dir_norm, circle_dir_norm])
        
        # vector from center of circle to traj pts
        d = np.array(self.trajectory.points) - circle_center

        # projection onto circle direction to find closest pt on circle's path
        projection_lens = np.sum(d*circle_dir_unit, axis=1)
        projection_pts = np.column_stack([circle_center[0] + projection_lens * circle_dir_unit[:, 0],
                                         circle_center[1] + projection_lens * circle_dir_unit[:, 1]])
        
        projection_dists = np.linalg.norm(projection_pts-circle_center, axis=1)

        intersection_indices = np.where(projection_dists <= circle_radius)[0]
        if len(intersection_indices)>0:
            for i in intersection_indices:
                if self.is_progressing_along_path(current_position, self.trajectory.points[i]):
                    return self.trajectory.points[i]
        return self.trajectory.points[-1] 
        # fallback only if no intersections, just return last point in trajectory...
        # really bad, but only way to follow trajectory if no lookahead wrt car otherwise it stops everything
        # maybe find another fallback implementation??
    
    def calc_steering_angle(self, current_position, lookahead_point):
        # calculate angle to lookahead
        dx = lookahead_point[0] - current_position.x
        dy = lookahead_point[1] - current_position.y
        th = np.arctan2(dy,dx)
        lookahead_dist = np.sqrt(dx**2 + dy**2)
        return np.arctan(2*self.wheelbase_length*np.sin(th)/lookahead_dist)
    
    def pose_callback(self, odometry_msg):
        my_pose = odometry_msg.pose.pose
        my_position = my_pose.position
        my_orientation = my_pose.orientation # dunno if we really need this
        # maybe useful to check that we're going forward along path?? idk
        lookahead_point = self.lookahead_point(my_position)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = self.calc_steering_angle(my_position, lookahead_point)
        self.drive_pub.publish(drive_msg)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        # converting trajectory to LineTrajectory format???
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        # flag, but where used??
        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()