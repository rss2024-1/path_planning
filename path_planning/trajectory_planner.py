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
        
        self.lane_trajectory = [(-19.99921417236328, 1.3358267545700073), (-18.433984756469727, 7.590575218200684), (-15.413466453552246, 10.617328643798828), (-6.186201572418213, 21.114534378051758), (-5.5363922119140625, 25.662315368652344), (-19.717021942138672, 25.677358627319336), (-20.30797004699707, 26.20694923400879), (-20.441822052001953, 33.974945068359375), (-55.0716438293457, 34.07769775390625), (-55.30067825317383, 1.4463690519332886)]
        
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
        self.lane_trajectory_viz = LineTrajectory(node=self, viz_namespace="/lane_trajectory")
        self.map = None
        self.map_resolution = None
        self.map_origin_orientation = None
        self.map_origin_poistion = None
        self.start_point = None
        self.end_point = None

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

    def map_cb(self, msg):
        self.get_logger().info("map callback")
        self.map = np.array(msg.data).reshape(msg.info.height, msg.info.width)

        #MAIN QUESTION: HOW TO GET ALL X,Y PIXEL VALUES FROM TRAJECTORY POINTS
        #ALSO, SHOULD I TAKE CROSSWALKS INTO ACCOUNT? 
        
        self.get_logger().info(f"np.array(msg.data): {np.array(msg.data)}")
        self.get_logger().info(f"Dimensions of np.array(msg.data): {np.array(msg.data).shape}")
        self.get_logger().info(f"self.map: {self.map}")
        self.get_logger().info(f"Dimensions of self.map: {self.map.shape if self.map is not None else None}")
        self.get_logger().info(f"Summary statistics of self.map: {np.unique(self.map, return_counts=True) if self.map is not None else None}")
        
        from scipy.ndimage import binary_dilation

        # Create a binary mask where 100 (walls) are True
        wall_mask = self.map == 100

        # Dilate the mask by 5 pixels
        dilated_wall_mask = binary_dilation(wall_mask, iterations=15)

        # Apply the dilated mask back to the map, setting these pixels to 100
        self.map[dilated_wall_mask] = 100

        # Post dilation
        self.get_logger().info(f"np.array(msg.data): {np.array(msg.data)}")
        self.get_logger().info(f"Dimensions of np.array(msg.data): {np.array(msg.data).shape}")
        self.get_logger().info(f"self.map: {self.map}")
        self.get_logger().info(f"Dimensions of self.map: {self.map.shape if self.map is not None else None}")
        self.get_logger().info(f"Summary statistics of self.map: {np.unique(self.map, return_counts=True) if self.map is not None else None}")

        self.map_resolution = msg.info.resolution
        self.map_origin_orientation = msg.info.origin.orientation
        self.map_origin_poistion = msg.info.origin.position
        #self.get_logger().info("map: {}".format(' '.join(map(str, self.map))))

        #IDEA: BUILD A MASK THAT CORRESPONDS TO LINE IN MIDDLE LINE FOLDER
        #AND MAKE THESE OCCUPANCY GRID MAP VALUES 100
        lane_traj_pixels = [self.map_to_pixel(point[0], point[1]) for point in self.lane_trajectory]
        lane_pixels = [] 
        for ix in range(len(lane_traj_pixels)):
            if ix < len(lane_traj_pixels) - 1: 
                point1 = lane_traj_pixels[ix]
                point2 = lane_traj_pixels[ix+1]
                lane_segment_pixels = self.bresenham_line_supercover(point1[0], point1[1], point2[0], point2[1])
                lane_pixels = lane_pixels + lane_segment_pixels

        self.get_logger().info(f"width: {len(self.map[0])}, height: {len(self.map)}")
        self.get_logger().info(str(lane_pixels))
        for lane_pixel in lane_pixels:
            self.map[lane_pixel[1], lane_pixel[0]] = 100
        self.lane_trajectory_viz.clear()
        for ix, lane_pixel in enumerate(lane_pixels):
            map_lane_cell = self.pixel_to_map(lane_pixel[0], lane_pixel[1])
            #if ix % 1 == 0:
                #self.get_logger().info(f'lane x: {map_lane_cell[0]}, y: {map_lane_cell[1]}')
            self.lane_trajectory_viz.addPoint(map_lane_cell)
        self.lane_trajectory_viz.publish_viz()
        self.get_logger().info(str(self.lane_trajectory_viz.points))
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

        #Processing to limit number of points
        if map_goal_path:
            reference_point = map_goal_path[0]
            filtered_points = [reference_point]
        self.trajectory.clear()
        for point in map_goal_path[1:]:
            if np.linalg.norm(np.array(point) - np.array(reference_point)) >= 1:
                filtered_points.append(point)
                self.trajectory.addPoint(point)
                reference_point = point

        #self.trajectory.points = filtered_points

        # self.trajectory.clear()
        # for point in map_goal_path:
        #     self.trajectory.addPoint(point)
        #self.trajectory.points = map_goal_path
        
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
