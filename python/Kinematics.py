
import numpy as np
import math

from nav_msgs.msg import Odometry

import numpy as np
import math


def Rzyx(euler):
    """
    Rzyx(euler) computes the rotation matrix, R in SO(3), using the zyx convention and Euler angle representation.
    """
    phi     = euler[0][0]
    theta   = euler[1][0]
    psi     = euler[2][0]

    R = np.array([[math.cos(psi)*math.cos(theta), -math.sin(psi)*math.cos(phi) + math.cos(psi)*math.sin(theta)*math.sin(phi), math.sin(psi)*math.sin(phi) + math.cos(psi)*math.cos(phi)*math.sin(theta)],
                  [math.sin(psi)*math.cos(theta), math.cos(psi)*math.cos(phi) + math.sin(phi)*math.sin(theta)*math.sin(psi), -math.cos(psi)*math.sin(phi) + math.sin(theta)*math.sin(psi)*math.cos(phi)],
                  [-math.sin(theta), math.cos(theta)*math.sin(phi), math.cos(theta)*math.cos(phi)]])
    return R

def Rquat(quat):
    """
    Rquat(quat) computes the rotation matrix, R in SO(3), using the zyx convention and unit-quaternion attitude representation.
    """
    w = quat[0][0]
    x = quat[1][0]
    y = quat[2][0]
    z = quat[3][0]

    R = np.array([[1-2*(y**2 + z**2),2*(x*y-z*w),2*(x*z+y*w)],
                  [2*(x*y + z*w),1-2*(x**2 + z**2),2*(y*z-x*w)],
                  [2*(x*z - y*w),2*(y*z + x*w),1-2*(x**2 + y**2)]])
    return R

def Tzyx(euler):
    """
    Tzyx(euler) computes the angular velocity transformation using the zyx convention and Euler angle attitude representation.
    """
    phi     = euler[0][0]
    theta   = euler[1][0]
    psi     = euler[2][0]

    T = np.array([[1, math.sin(phi)*math.tan(theta), math.cos(phi)*math.tan(theta)],
                  [0, math.cos(phi),-math.sin(phi)],
                  [0, math.sin(phi)/math.cos(theta), math.cos(phi)/math.cos(theta)]])
    return T

def Tquat(quat):
    """
    Tquat(quat) computes the angular velocity transformation using the zyx convention and unit quaternion attitude representation.
    """
    w = quat[0][0]
    x = quat[1][0]
    y = quat[2][0]
    z = quat[3][0]

    T = np.array([ [-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])*0.5
    return T

def H(R,o):
    """
    H(R,o) computes the homogeneous transformation given a relative rotation R and translation o
    """
    b       = np.array([[0, 0, 0, 1]])
    temp    = np.concatenate((R,o),axis=1)
    H       = np.concatenate((temp,b),axis=0)
    return H

def Hinv(R,o):
    """
    H(R,o) computes the inverse homogeneous transformation given a relative rotation R and translation o
    """
    b       = np.array([[0, 0, 0, 1]])
    temp    = np.concatenate((R.T,-R.T.dot(o)),axis=1)
    Hinv    = np.concatenate((temp,b),axis=0)
    return Hinv

def q2euler(quat):
    """
    q2euler(quat) converts the unit-quaternion representation, quat, to the Euler angle representation (phi, theta, psi)
    """
    w = quat[0][0]
    x = quat[1][0]
    y = quat[2][0]
    z = quat[3][0]

    phi     = math.atan2(2*(w*x + y*z),1-2*(x**2 + y**2))
    theta   = math.asin(2*(w*y - z*x))
    psi     = math.atan2(2*(w*z + x*y),1-2*(y**2 + z**2))
    return np.array([[phi, theta, psi]]).T

def euler2q(euler):
    """
    euler2q(euler) converts the euler angle representation to the unit-quaternion representation
    """
    phi     = euler[0][0]
    theta   = euler[1][0]
    psi     = euler[2][0]
    
    w = math.cos(0.5*psi)*math.cos(0.5*theta)*math.cos(0.5*phi) + math.sin(0.5*psi)*math.sin(0.5*theta)*math.sin(0.5*phi)
    x = math.cos(0.5*psi)*math.cos(0.5*theta)*math.sin(0.5*phi) - math.sin(0.5*psi)*math.sin(0.5*theta)*math.cos(0.5*phi)
    y = math.sin(0.5*psi)*math.cos(0.5*theta)*math.sin(0.5*phi) + math.cos(0.5*psi)*math.sin(0.5*theta)*math.cos(0.5*phi)
    z = math.sin(0.5*psi)*math.cos(0.5*theta)*math.cos(0.5*phi) - math.cos(0.5*psi)*math.sin(0.5*theta)*math.sin(0.5*phi)

    return np.array([[w, x, y, z]]).T

def H2o(H):
    """
    H2o(H) returns the translation vector, o, given a homogeneous transformation H
    """
    o_x = H[0][3]
    o_y = H[1][3]
    o_z = H[2][3]
    return np.array([[o_x, o_y, o_z]]).T

def H2quat(H):
    """
    H2quat(H) computes the unit-quaternion given the homogeneous transformation martix H
    """
    w = np.sqrt( (1 + H[0][0] + H[1][1] + H[2][2])/2 )
    x = (H[2][1] - H[1][2])/(4*w)
    y = (H[0][2] - H[2][0])/(4*w)
    z = (H[1][0] - H[0][1])/(4*w)
    return np.array([[w, x, y, z]]).T


def R2quat(R):
    """
    R2quat(R) computes the unit-quaternion given the rotation matrix
    """
    w = np.sqrt( (1 + R[0][0] + R[1][1] + R[2][2])/2 )
    x = (R[2][1] - R[1][2])/(4*w)
    y = (R[0][2] - R[2][0])/(4*w)
    z = (R[1][0] - R[0][1])/(4*w)
    return np.array([[w, x, y, z]]).T

def S(w):
    """
    S(w) computes the 3x3 vector skew-symmetric matrix S(w) = -S(w).T giver the angular velocities  w = [p, q, r]^T
    """
    p = w[0][0]
    q = w[1][0]
    r = w[2][0]
    Skew = np.array([   [0,     -r,     q],
                        [r,     0,      -p],
                        [-q,    p,      0]])
    return Skew

def ssa(angle):
    """
    ssa(angle) returns the smallest signed angle
    """
    return (angle + math.pi)%(2*math.pi) - math.pi

def c_psi(quat):
    """
    c_psi(quat) calculates the linearization of the compass measurement
    """
    w = quat[0][0]
    x = quat[1][0]
    y = quat[2][0]
    z = quat[3][0]

    a = (2/x)*np.array([[y, z, w]]).T
    a1 = a[0][0]
    a2 = a[1][0]
    a3 = a[2][0]
    u  = 2*(a1*a2 + 2*a3)/(4 + a1**2 - a2**2 - a3**2)
    du = 1/(1+u**2)
    cpsi = du * (1/(4 + a1**2 - a2**2 -a3**2)**2) * np.array([[-2*((a1**2+a3**2-4)*a2 + a2**3 + 4*a1*a3)], [2*((a1**2-a3**2+4)*a1 + a1**3 + 4*a2*a3)], [4*((a3**2+a1*a2*a3 + a1**2 -a2**2 +4))]])
    print(cpsi)
    return u, cpsi

def quatprod(quat1,quat2):
    """
    quatprod(quat1,quat2) computes the quaternion product
    """
    w1 = quat1[0:1]
    imag1 = quat1[1:4]
    w2 = quat2[0:1]
    imag2 = quat2[1:4]

    prod1 = w1*w2 - imag1.T.dot(imag2)
    prod2 = w2*imag1 + w1*imag2 + np.cross(imag1,imag2, axis=0)
    prod  = np.concatenate((prod1,prod2),axis=0)
    return prod
