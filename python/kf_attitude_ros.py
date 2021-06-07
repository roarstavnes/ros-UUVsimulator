import rospy
import cv2
import numpy as np
import math

# Ros messages
#from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu, MagneticField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray

from Kinematics import *

class Observer:
    def __init__(self):
        # Measurements
        self.f_imu  = np.array([[0.0, 0.0, -9.8]]).T
        self.w_imu  = np.array([[0.0, 0.0, 0.0]]).T
        self.m_imu  = np.array([[0.0, -32.5, 56.3]]).T
        self.poses  = None
        self.receivedCameraMeasurement = False

        self.counter = 0.0
        self.runtime = 0.0

        # Magnetometer reference vector
        self.m_ref  = np.array([[0, -32.5, 56.3]]).T

        # Transformation matrix from body to camera
        pi = math.pi
        self.d_b2c  = np.array([[1.15, 0, -0.4]]).T
        euler       = np.array([[pi/2 -0.6, 0, pi/2]]).T
        self.R_b2c  = Rzyx(euler)


        # Kalman filter parameters
        self.p_ins      = np.zeros((3,1))
        self.v_ins      = np.zeros((3,1))
        self.b_acc_ins  = np.zeros((3,1))
        self.quat_ins   = np.zeros((4,1))
        self.quat_ins[0][0] = 1
        self.b_ars_ins  = np.zeros((3,1))


        self.Q_d    = np.diag([0.001, 0.001, 0.001, 0.000001, 0.000001, 0.000001, 0.0005, 0.0005, 0.0005, 0.000001, 0.000001, 0.000001])*0.0001
        self.Rd     = np.array([0.007, 0.007, 0.007, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001]) *0.001

        self.P_prd = np.identity(15)
        self.P_hat  = np.zeros((15,15))

        self.T_acc = 1000
        self.T_ars = 500

        self.g_n = np.array([[0, 0, 9.8]]).T

        self.dt = 0.01
        
        self.pose_ins_publisher = rospy.Publisher("/rexrov2/observer/odom", Odometry,queue_size=50)

        # Subscribe to topics
        rospy.Subscriber("/rexrov2/marker/poses_raw",PoseArray, self.poses_callback)
        rospy.Subscriber("/rexrov2/imu",Imu, self.imu_callback)
        rospy.Subscriber("/rexrov2/magnetometer", MagneticField, self.mag_callback)

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.run)


    def run(self,timer):
        
        Z_3 = np.zeros((3,3))
        I_3 = np.identity(3)

        # Jacobian matrices
        R = Rquat(self.quat_ins)
        T = Tquat(self.quat_ins)

        # Bias compensated IMU measurements
        f_ins = self.f_imu - self.b_acc_ins
        w_ins = self.w_imu - self.b_ars_ins

        # Normalized gravity vectors
        v10 = np.array([[0, 0, 1]]).T
        v1  = - f_ins
        v1  = v1/np.linalg.norm(v1)

        # Normalized magnetic field vectors
        v20 = self.m_ref/np.linalg.norm(self.m_ref)
        v2  = self.m_imu/np.linalg.norm(self.m_imu)


        # Define state space matrices
        A = np.concatenate((np.concatenate((Z_3,    I_3,    Z_3,                    Z_3,                    Z_3),axis=1),
                            np.concatenate((Z_3,    Z_3,   -1*R,                   -1*R.dot(S(f_ins)),      Z_3),axis=1),
                            np.concatenate((Z_3,    Z_3,   -(1/self.T_acc)*I_3,     Z_3,                    Z_3),axis=1),
                            np.concatenate((Z_3,    Z_3,    Z_3,                   -1*S(w_ins),            -I_3),axis=1),
                            np.concatenate((Z_3,    Z_3,    Z_3,                    Z_3,                   -(1/self.T_ars)*I_3),axis=1)),axis=0)
                            

        Ad = np.identity(15) + self.dt*A

        Cd = np.concatenate((np.concatenate((I_3,   Z_3,    Z_3,    Z_3,                Z_3),axis=1),
                             np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v10)),    Z_3),axis=1),
                             np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v20)),    Z_3),axis=1)),axis=0)

        Ed = self.dt*np.concatenate((np.concatenate((Z_3,     Z_3,    Z_3,      Z_3),axis=1),   
                                     np.concatenate((-1*R,    Z_3,    Z_3,      Z_3),axis=1),
                                     np.concatenate((Z_3,     I_3,    Z_3,      Z_3),axis=1),
                                     np.concatenate((Z_3,     Z_3,    -1*I_3,   Z_3),axis=1),
                                     np.concatenate((Z_3,     Z_3,    Z_3,      I_3),axis=1)),axis=0)
  
        # Check if aiding measurement is available
        if self.receivedCameraMeasurement == False and self.counter <= 0.1:
            self.P_hat = self.P_prd
        else:
            self.counter = 0
            eps_g       = v1 - R.T.dot(v10)
            eps_mag     = v2 - R.T.dot(v20)

            if self.receivedCameraMeasurement == True and self.runtime >= 4:

                self.receivedCameraMeasurement == False
                Cd = np.concatenate((np.concatenate((I_3,   Z_3,    Z_3,    Z_3,                Z_3),axis=1),
                                     np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v10)),    Z_3),axis=1),
                                     np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v20)),    Z_3),axis=1)),axis=0)   
                R_d = np.diag(self.Rd)

                # Take the weighted discounted average for the translation between camera and the marker(s)
                weight  = 0
                y_pos   = np.zeros((3,1)) 
                
                for i in range(0,int(len(self.poses)/2)):
                    t_x = self.poses[2*i].position.x
                    t_y = self.poses[2*i].position.y
                    t_z = self.poses[2*i].position.z 
                    t   = np.array([[t_x, t_y, t_z]]).T

                    x_i = self.poses[2*i+1].position.x
                    y_i = self.poses[2*i+1].position.y
                    z_i = self.poses[2*i+1].position.z
                    p_i = np.array([[x_i, y_i, z_i]]).T

                    y_i = self.R_b2c.dot(t) + self.d_b2c

                    norm_inv = 1/np.sqrt(y_i[0][0]**2 + y_i[1][0]**2 + y_i[2][0]**2)

                    if norm_inv > 1:
                        norm_inv = 1

                    weight += norm_inv
                    
                    y_pos += norm_inv*(p_i - R.dot(y_i))

                y_pos   = y_pos/weight
                
                eps_pos = y_pos - self.p_ins
                #self.get_logger().info('avg_pos: %f, %f, %f' %(eps_pos[0][0], eps_pos[1][0], eps_pos[2][0]))
                eps     = np.concatenate((eps_pos, eps_g, eps_mag), axis=0)
            
            else:
                Cd = np.concatenate((np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v10)),    Z_3),axis=1),
                                     np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v20)),    Z_3),axis=1)),axis=0)   
                eps = np.concatenate((eps_g,eps_mag),axis = 0)
                R_d  = np.diag(self.Rd[3:9])

            # KF gain: K[k]
            K   = self.P_prd.dot(Cd.T).dot(np.linalg.inv(Cd.dot(self.P_prd).dot(Cd.T) + R_d))
            IKC = np.identity(15) - K.dot(Cd)


            # Corrector 
            delta_x_hat = K.dot(eps)
            self.P_hat  = IKC.dot(self.P_prd).dot(IKC.T) + K.dot(R_d).dot(K.T)

            # Error quaternion
            delta_a     = delta_x_hat[9:12]
            delta_quat_hat = 1/np.sqrt(4 + delta_a.T.dot(delta_a)) * np.array([[2, delta_a[0][0], delta_a[1][0], delta_a[2][0]]]).T

            # INS reset: x_ins[k]
            self.p_ins      += delta_x_hat[:3]
            self.v_ins      += delta_x_hat[3:6]
            self.b_acc_ins  += delta_x_hat[6:9]
            self.b_ars_ins  += delta_x_hat[12:15]
            self.quat_ins   = quatprod(self.quat_ins,delta_quat_hat)
            self.quat_ins   = self.quat_ins/np.linalg.norm(self.quat_ins)


        # Predictor: P_prd[k+1]
        self.P_prd = Ad.dot(self.P_hat).dot(Ad.T) + Ed.dot(self.Q_d).dot(Ed.T)

        # INS propagation: x_ins[k+1]
        self.p_ins      += self.dt*self.v_ins
        self.v_ins      += self.dt*(R.dot(f_ins) + self.g_n)
        self.quat_ins   += self.dt*T.dot(w_ins)
        self.quat_ins    = self.quat_ins/np.linalg.norm(self.quat_ins)
        self.counter    += self.dt
        self.runtime    += self.dt

        # Publish estimate
        msg = Odometry()
        msg.header.stamp = rospy.get_rostime()
        msg.pose.pose.position.x = self.p_ins[0][0]
        msg.pose.pose.position.y = self.p_ins[1][0]
        msg.pose.pose.position.z = self.p_ins[2][0]
        msg.pose.pose.orientation.w = self.quat_ins[0][0]
        msg.pose.pose.orientation.x = self.quat_ins[1][0]
        msg.pose.pose.orientation.y = self.quat_ins[2][0]
        msg.pose.pose.orientation.z = self.quat_ins[3][0]
        msg.twist.twist.linear.x = self.v_ins[0][0]
        msg.twist.twist.linear.y = self.v_ins[1][0]
        msg.twist.twist.linear.z = self.v_ins[2][0]
        self.pose_ins_publisher.publish(msg)

    def poses_callback(self,msg):
        self.receivedCameraMeasurement = True
        self.poses = msg.poses

    def imu_callback(self,msg):
        self.f_imu[0][0] = msg.linear_acceleration.x
        self.f_imu[1][0] = msg.linear_acceleration.y
        self.f_imu[2][0] = msg.linear_acceleration.z
        self.w_imu[0][0] = msg.angular_velocity.x
        self.w_imu[1][0] = msg.angular_velocity.y 
        self.w_imu[2][0] = msg.angular_velocity.z
    
    def mag_callback(self,msg):
        self.m_imu[0][0] = msg.magnetic_field.x # bug in UUV simulator
        self.m_imu[1][0] = -msg.magnetic_field.y # bug in UUV simulator
        self.m_imu[2][0] = -msg.magnetic_field.z 

if __name__ == '__main__':

    rospy.init_node('listenerObserver')

    node = Observer()

    rospy.spin()
