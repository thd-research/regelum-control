import rospy
from std_srvs.srv import Empty

reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

reset_world()