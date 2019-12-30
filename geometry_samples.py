import numpy as np
from geometry import Geometry

if __name__ == "__main__":
    g = Geometry()
    Rx1 = g.Rx_p(np.deg2rad(10))
    Ry1 = g.Ry_p(np.deg2rad(10))
    Rz1 = g.Rz_p(np.deg2rad(10))
    Rx2 = g.Rx_p(np.deg2rad(5))
    Ry2 = g.Ry_p(np.deg2rad(5))
    Rz2 = g.Rz_p(np.deg2rad(5))
    e1 = np.array([[1], [0], [0]])
    e2 = np.array([[0], [1], [0]])
    e3 = np.array([[0], [0], [1]])
    p = e3
    # v = g.rotate_axis_angle(e2, e1, np.deg2rad(90))

    v = Rx2 @ Ry2 @ Rz2 @ e2

    test1 = g.rotate_axis_angle(p, v, np.deg2rad(10))
    Ry1 = g.Ry_p(np.deg2rad(10)+np.deg2rad(10))
    test2 = Rx2 @ Ry2 @ Rz2 @ Rz1 @ Ry1 @ Rx1 @ e2
    print(test1)
    print(test2)



