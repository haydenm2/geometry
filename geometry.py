import numpy as np

class Geometry:
    def __init__(self):
        pass

    def Rx_a(self, ang):
        return np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])

    def Ry_a(self, ang):
        return np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])

    def Rz_a(self, ang):
        return np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])

    def Rx_p(self, ang):
        return self.Rx_a(ang).transpose()

    def Ry_p(self, ang):
        return self.Ry_a(ang).transpose()

    def Rz_p(self, ang):
        return self.Rz_a(ang).transpose()

    def R_to_axis_angle(self, R):
        ang = np.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
        x = (R[2, 1] - R[1, 2]) / np.sqrt((R[2, 1] - R[1, 2])**2 + (R[0, 2] - R[2, 0])**2 + (R[1, 0] - R[0, 1])**2)
        y = (R[0, 2] - R[2, 0]) / np.sqrt((R[2, 1] - R[1, 2])**2 + (R[0, 2] - R[2, 0])**2 + (R[1, 0] - R[0, 1])**2)
        z = (R[1, 0] - R[0, 1]) / np.sqrt((R[2, 1] - R[1, 2])**2 + (R[0, 2] - R[2, 0])**2 + (R[1, 0] - R[0, 1])**2)
        ax = np.vstack((x, y, z))
        return [ax, ang]

    def R_to_quat(self, R):
        w = np.sqrt(1 + R[0, 0] + R[1, 1], R[2, 2]) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
        return np.vstack((w, x, y, z))

    def axis_angle_to_R(self, ax, ang):
        R = np.array([[np.cos(ang) + ax[0]**2 * (1 - np.cos(ang)), ax[0]*ax[1]*(1-np.cos(ang)) - ax[2]*np.sin(ang), ax[0]*ax[2]*(1-np.cos(ang)) + ax[1]*np.sin(ang)], \
                      [ax[0]*ax[1]*(1-np.cos(ang)) + ax[2]*np.sin(ang), np.cos(ang) + ax[1]**2 * (1 - np.cos(ang)), ax[1]*ax[2]*(1-np.cos(ang)) - ax[0]*np.sin(ang)], \
                      [ax[0]*ax[2]*(1-np.cos(ang)) - ax[1]*np.sin(ang), ax[1]*ax[2]*(1-np.cos(ang)) + ax[0]*np.sin(ang), np.cos(ang) + ax[2]**2 * (1 - np.cos(ang))]])
        return R

    def axis_angle_to_quat(self, ax, ang):
        x = ax[0] * np.sin(ang/2)
        y = ax[1] * np.sin(ang/2)
        z = ax[2] * np.sin(ang/2)
        w = np.cos(ang/2)
        return np.vstack((w, x, y, z))

    def quat_to_R(self, q):
        # assumes normalized quaternion
        R = np.array([[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]], \
                      [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]], \
                      [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]])
        return R

    def quat_to_axis_angle(self, q):
        ang = 2 * np.acos(q[0])
        x = q[1] / np.sqrt(1 - q[0] * q[0])
        y = q[2] / np.sqrt(1 - q[0] * q[0])
        z = q[3] / np.sqrt(1 - q[0] * q[0])
        ax = np.vstack((x, y, z))
        return [ax, ang]

    def rotate_axis_angle(self, v, v_axis, ang):
        # assumes v and v_axis are column vectors
        v_rot = v * np.cos(ang) + np.cross(v_axis, v, axis=0) * np.sin(ang) + np.tensordot(v_axis, v) * v_axis * (1 - np.cos(ang))
        return v_rot

    def rotate_quaternion(self, q, v):
        v_comb = np.vstack((0, v))
        v_rot = q.transpose() @ v_comb @ q
        return v_rot



