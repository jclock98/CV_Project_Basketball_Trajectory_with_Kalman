import cv2
import numpy as np
import numpy.linalg as la


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    # H
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    # A
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

    def predict(self, coord_x, coord_y):
        """ This function estimates the position of the object"""
        measured = np.array([[np.float32(coord_x)], [np.float32(coord_y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()  # prediction
        statePre = self.kf.statePre  # x_k-1
        statePost = self.kf.statePost  # x_k

        x, y = int(predicted[0]), int(predicted[1])
        return (x, y), statePre.T[0], statePost.T[0]

    def kal(self, x_k_minus_1, P, B, u, z):
        """ This function update the prediction (?),
            B = control matrix
            P = errorCov
        """
        A = self.kf.transitionMatrix
        Q = self.kf.processNoiseCov
        H = self.kf.measurementMatrix
        R = self.kf.measurementNoiseCov

        x_pred = A @ x_k_minus_1 + B @ u
        P_pred = A @ P @ A.T + Q / 4

        if z is None:
            return x_pred, P_pred

        zp = H @ x_pred
        epsilon = z - zp

        k = P_pred @ H.T @ la.inv(H @ P_pred @ H.T + R)  # Kalman Gain

        x_esti = x_pred + k @ epsilon
        P = (np.eye(len(P)) - k @ H) @ P_pred
        return x_esti, P
