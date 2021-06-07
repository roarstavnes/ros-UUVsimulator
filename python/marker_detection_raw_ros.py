
import rospy
import cv2
import numpy as np
import math

# Bridge between ROS and OpenCV
from cv_bridge import CvBridge, CvBridgeError

# Ros messages
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose 

from Kinematics import *
import Init as init


# Instantiate CvBridge
bridge = CvBridge()

# Commonly used matrices and constants
pi = math.pi

# Homogeneous transformation from the BODY frame to the camera frame
euler = np.array([[pi/2 - 0.6, 0 , pi/2]]).T
R_body2camera = Rzyx(euler)
o_body2camera = np.array([ [1.15, 0, -0.4]]).T
H_body2camera = H(R_body2camera,o_body2camera)
H_camera2body = (R_body2camera,o_body2camera)


class Aruco:
    def __init__(self, id, marker_length, x, y, z, phi, theta, psi):
        self.id             = id
        self.length         = marker_length
        self.position       = np.array([[x, y, z]]).T
        self.orientation    = np.array([[phi, theta, psi]]).T
        self.quat           = euler2q(self.orientation)
        self.R_ned2marker   = Rzyx(self.orientation)
        self.H_ned2marker   = H(self.R_ned2marker, self.position)
        self.H_marker2ned   = Hinv(self.R_ned2marker, self.position)        


# Create a list for all markers
markers = []
markers.append( Aruco(7, 0.3, 0, 2, 2, -pi/2, 0, pi))
markers.append( Aruco(2, 0.3, -1, 2, 2.3, -pi/2, 0, pi))



    
class markerDetectionRaw:

    def __init__(self):

        # Camera measurement
        self.image                      = None
        self.dt                         = 0.2

        self.marker_pose_raw_publisher = rospy.Publisher("/rexrov2/marker/poses_raw", PoseArray,queue_size=1)
        self.detectedMarker_image_publisher = rospy.Publisher("/rexrov2/camera/detected_markers", Image, queue_size=1)

        rospy.Subscriber("/rexrov2/rexrov2/camera/camera_image", Image, self.image_callback)        

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.run)
            
    def run(self,timer):
        """
        run() computes the global position of the vehicle by detection and computation of relative position of Aruco markers with known position and orientation.
        """
        if self.image is None:
            return

        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
            
        aruco_dict      = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        aruco_params    = cv2.aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,aruco_dict,parameters=aruco_params)
        
        if ids is None:
            return

        msg = PoseArray()

        rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners,0.3,cameraMatrix=init.camera_matrix,distCoeffs=init.distortion_coefficients,rvecs=None,tvecs=None)

        for i in range(0,len(ids)):
            for marker in markers:
                if ids[i] == marker.id:

                    t_x = tvecs[i][0][0]
                    t_y = tvecs[i][0][1]
                    t_z = tvecs[i][0][2]
                                

                    r_x = rvecs[i][0][0] 
                    r_y = rvecs[i][0][1] 
                    r_z = rvecs[i][0][2]
                    r   = np.array([[r_x, r_y, r_z]]).T
                    quat = euler2q(r)
                                

                    # Publish marker pose estimate
                    msg_camera = Pose()
                    msg_camera.position.x    = t_x
                    msg_camera.position.y    = t_y
                    msg_camera.position.z    = t_z
                    msg_camera.orientation.w = quat[0][0]
                    msg_camera.orientation.x = quat[1][0]
                    msg_camera.orientation.y = quat[2][0]
                    msg_camera.orientation.z = quat[3][0]

                    msg_marker = Pose()
                    msg_marker.position.x     = float(marker.position[0][0])
                    msg_marker.position.y     = float(marker.position[1][0])
                    msg_marker.position.z     = float(marker.position[2][0])
                    msg_marker.orientation.w  = marker.quat[0][0]
                    msg_marker.orientation.x  = marker.quat[1][0]
                    msg_marker.orientation.y  = marker.quat[2][0]
                    msg_marker.orientation.z  = marker.quat[3][0]

                    msg.poses.append(msg_camera)
                    msg.poses.append(msg_marker)                   
        
        
        msg.header.stamp = rospy.get_rostime()
        self.marker_pose_raw_publisher.publish(msg)

        cv2.aruco.drawDetectedMarkers(self.image,corners,ids)
        msg = Image()
        msg = bridge.cv2_to_imgmsg(self.image,"bgr8")
        self.detectedMarker_image_publisher.publish(msg)

        self.image = None
    
    def image_callback(self,msg):
        self.image = bridge.imgmsg_to_cv2(msg, "bgr8")


if __name__ == '__main__':

    rospy.init_node('listener')

    node = markerDetectionRaw()

    rospy.spin()
