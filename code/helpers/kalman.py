# This file contains a Kalman filter class.
# The implementation assume 2D motion, and no acceleration.
# The code is largely copied from https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/, with some modifications to make it fit to the salmon tracking task.

import numpy as np
class KalmanFilter(object):
        def __init__(self, dt, x_0, y_0, std_acc, std_meas):
                # Sampling time
                self.dt = dt

                # Intial state
                self.x = np.matrix([[x_0], [y_0], [0], [0]])

                # State transition matrix A
                self.A = np.matrix([    [1, 0, self.dt, 0],
                                        [0, 1, 0, self.dt],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
                
                # Measurement mapping matrix
                self.H = np.matrix([    [1, 0, 0, 0],
                                        [0, 1, 0, 0]])
                
                # Model uncertainty
                self.Q = np.matrix([    [(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                                        [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                                        [(self.dt**3)/2, 0, self.dt**2, 0],
                                        [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
                
                # Measurement uncertainty
                self.R = np.matrix([    [std_meas**2,0],
                                        [0, std_meas**2]])
                
                # Initial Covariance matrix
                self.P = np.eye(self.A.shape[1])

        def predict(self):
                self.x = np.dot(self.A, self.x) 
                self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
                return self.x[0:2]

        def update(self, z):
                S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
                K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 
                self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
                I = np.eye(self.H.shape[1])
                self.P = (I - (K * self.H)) * self.P
                return self.x[0:2]