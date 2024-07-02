from numpy import ndarray
from regelum.simulator import Simulator, CasADi
from regelum.system import ComposedSystem, System, ThreeWheeledRobotDynamic

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist

import transformations as tftr 
import threading
import math
import numpy as np
import time
import subprocess
from pathlib import Path


class RosTurtlebot(CasADi):
    def __init__(self, system: System | ComposedSystem,
                 state_goal: ndarray, 
                 state_init: ndarray | None = None, 
                 action_init: ndarray | None = None, 
                 time_final: float | None = 1, 
                 max_step: float | None = 0.001, 
                 first_step: float | None = 0.000001, 
                 atol: float | None = 0.00001, 
                 rtol: float | None = 0.001,
                 ros_ctrl_rate: int = 100
                 ):
        self.state_goal = state_goal
        self.rotation_counter = 0
        self.prev_theta = 0
        self.new_state = state_init
    
        # Topics
        rospy.init_node('ros_preset_node', log_level=rospy.INFO)
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=False)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odometry_callback)

        # ROS 
        self.RATE = rospy.get_param('/rate', int(1/max_step))
        self.lock = threading.Lock()

        while not hasattr(self, "new_state"):
            time.sleep(.1)

        super().__init__(system, self.new_state, action_init, time_final, max_step, first_step, atol, rtol)
        self._action = np.expand_dims(self.initialize_init_action(), axis=0)

        self.reset()

    def get_velocity(self, msg):
        self.linear_velocity = msg.twist.twist.linear.x
        self.angular_velocity = msg.twist.twist.angular.z

    def odometry_callback(self, msg):
        self.lock.acquire()
        self.get_velocity(msg)
        # Read current robot state
        x = msg.pose.pose.position.x

        # Complete for y and orientation
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
          
        # Transform quat2euler using tf transformations: complete!
        current_rpy = tftr.euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Extract Theta (orientation about Z)
        theta = current_rpy[0]
        
        # Make transform matrix from 'robot body' frame to 'goal' frame
        theta_goal = self.state_goal[2]
        
        # Complete rotation matrix
        rotation_matrix = np.array([
            [np.cos(theta_goal), -np.sin(theta_goal), 0],
            [np.sin(theta_goal), np.cos(theta_goal), 0],
            [0, 0, 1]
        ])
            
        state_matrix = np.vstack([self.state_goal[0], self.state_goal[1], 0])  # [x, y, 0] -- column   
        
        # Compute Transformation matrix 
        self.t_matrix = np.block([
            [rotation_matrix, state_matrix],
            [np.array([0, 0, 0, 1])]
            ])
        
        # Complete rotation counter for turtlebot3
        ''' Your solution goes here (rotation_counter) '''        
        if math.copysign(1, self.prev_theta) != math.copysign(1, theta) and \
            abs(self.prev_theta) > np.pi:
            if math.copysign(1, self.prev_theta) == -1:
                self.rotation_counter -= 1
            
            else:
                self.rotation_counter += 1
        
        self.prev_theta = theta
        theta = theta + 2 * np.pi * self.rotation_counter
        
        # Orientation transform
        new_theta = theta - theta_goal
        
        # Do position transform
        temp = np.array([x, y , 0, 1])
        self.new_state = np.linalg.inv(self.t_matrix) @ temp.T
        self.new_state = [self.new_state[0], self.new_state[1], new_theta]

        if isinstance(self.system, ThreeWheeledRobotDynamic):
            self.new_state += [self._action[0, 0], self._action[0, 1]] # if hasattr(self, "_action") else [0, 0]

        self.new_state = np.expand_dims(self.new_state, axis=0)
        self.lock.release()

    # Override this reset
    def reset(self):
        if self.system.system_type == "diff_eqn":
            self.ODE_solver = self.initialize_ode_solver()
            self.time = 0.0
            self.state = self.new_state
            self.observation = self.get_observation(
                time=self.time, state=self.new_state, inputs=self.action_init
            )
        else:
            self.time = 0.0
            self.observation = self.get_observation(
                time=self.time, state=self.new_state, inputs=self.system.inputs
            )
        
        self.total_timestamp = int(self.time_final/self.max_step)
        self.rate = rospy.Rate(self.RATE)
        self.episode_start = None
        subprocess.check_output(["python3.10", f"{Path(__file__).parent.resolve()}/reset_ros.py"])
        self.receive_action(np.zeros_like(self._action))

    # Publish action to gazebo
    def receive_action(self, action):
        try:
            self.system.receive_action(action)
            self._action = action
            velocity = Twist()

            # Generate ROSmsg from action
            velocity.linear.x = action[0, 0]
            velocity.angular.z = action[0, 1]
            self.pub_cmd_vel.publish(velocity)
        except Exception as err:
            print(err)

    # Stop condition
    # update time, new_state
    def do_sim_step(self):
        '''
        Return: -1: episode ended
                otherwise: episode continues
        '''
        stop_signal = False

        stop_signal |= rospy.is_shutdown()

        self.total_timestamp -= 1
        stop_signal |= self.total_timestamp < 0

        self.rate.sleep()

        if stop_signal:
            return -1

        if self.episode_start is None:
            self.episode_start = rospy.get_time()
        self.time = rospy.get_time() - self.episode_start
        self.observation = self.get_observation(
                time=self.time, state=self.new_state, inputs=self.system.inputs
            )
        self.state = self.new_state
        pass
