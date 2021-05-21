import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_fundamental(x1, x2):
    # build matrix for equations
    equations = []
    for i, (p1, p2) in enumerate(zip(x1, x2)):
        u_t, v_t = p1
        u, v = p2
        equations.append([u_t * u, u_t * v, u_t, v_t * u, v_t * v, v_t, u, v, 1])

    equations_matrix = np.stack(equations)

    l, v, r = np.linalg.svd(equations_matrix)

    return r[-1].reshape(3, 3).transpose()


def compute_fundamental_with_least_squares(x1, x2):
    # build matrix for equations
    equations = []
    for i, (p1, p2) in enumerate(zip(x1, x2)):
        u_t, v_t = p1
        u, v = p2
        equations.append([u_t * u, u_t * v, u_t, v_t * u, v_t * v, v_t, u, v, 1])

    equations.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
    equations_matrix = np.stack(equations)

    b = np.zeros(equations_matrix.shape[0])
    b[-1] = 1
    sol = np.linalg.lstsq(equations_matrix, b, rcond=None)[0]
    sol = sol * 1 / sol[8]  # normalize

    return sol.reshape(3, 3).transpose()


def load_file(name):
    lst = []
    f = open(name, 'r')
    for line in f:
        lst.append(line.strip().split())
    f.close()
    return np.array(lst, dtype=np.float32)

pts_a = load_file('input/sceneA/sceneA-pts2d-1.txt')
pts_b = load_file('input/sceneA/sceneA-pts2d-2.txt')

F_sceneA = compute_fundamental_with_least_squares(pts_a, pts_b)

U,S,V = np.linalg.svd(F_sceneA)
S[-1] = 0
F_sceneA_rank_2 = U.dot(np.diag(S).dot(V))


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def plot_epipole_lines(pts1, pts2, I1, I2, F):
    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax2.imshow(I2)

    plt.sca(ax1)

    sy_1, sx_1, _ = I1.shape
    sy_2, sx_2, _ = I2.shape

    # calculate plot for right image
    for point in pts1:
        xc, yc = int(point[0]), int(point[1])
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        l = l / s

        if l[0] != 0:
            ye = sy_2 - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx_2 - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        epiline = line([xs, ys], [xe, ye])

        left_boundary = line([0, 0], [0, 1])
        right_boundary = line([sx_2 - 1, 0], [sx_2 - 1, 1])

        upper_boundary = line([0, sy_2 - 1], [1, sy_2 - 1])
        bottom_boundary = line([0, 0], [1, 0])

        left_boundary_point = intersection(epiline, left_boundary)
        right_boundary_point = intersection(epiline, right_boundary)
        upper_boundary_point = intersection(epiline, upper_boundary)
        bottom_boundary_point = intersection(epiline, bottom_boundary)
        intersection_points = []

        if 0 < left_boundary_point[1] < I2.shape[1]:
            intersection_points.append(left_boundary_point)

        if 0 < right_boundary_point[1] < I2.shape[1]:
            intersection_points.append(right_boundary_point)

        if 0 < upper_boundary_point[0] < I2.shape[1]:
            intersection_points.append(upper_boundary_point)

        if 0 < bottom_boundary_point[0] < I2.shape[1]:
            intersection_points.append(bottom_boundary_point)

        assert len(intersection_points) == 2

        ax2.plot([intersection_points[0][0], intersection_points[1][0]],
                 [intersection_points[0][1], intersection_points[1][1]], linewidth=2)

    # calculate plot for left image
    for point in pts2:
        xc, yc = int(point[0]), int(point[1])
        v = np.array([xc, yc, 1])
        l = v.dot(F)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        l = l / s

        if l[0] != 0:
            ye = sy_1 - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx_1 - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        epiline = line([xs, ys], [xe, ye])

        left_boundary = line([0, 0], [0, 1])
        right_boundary = line([sx_1 - 1, 0], [sx_1 - 1, 1])

        upper_boundary = line([0, sy_1 - 1], [1, sy_1 - 1])
        bottom_boundary = line([0, 0], [1, 0])

        left_boundary_point = intersection(epiline, left_boundary)
        right_boundary_point = intersection(epiline, right_boundary)
        upper_boundary_point = intersection(epiline, upper_boundary)
        bottom_boundary_point = intersection(epiline, bottom_boundary)
        intersection_points = []

        if 0 < left_boundary_point[1] < I1.shape[1]:
            intersection_points.append(left_boundary_point)

        if 0 < right_boundary_point[1] < I1.shape[1]:
            intersection_points.append(right_boundary_point)

        if 0 < upper_boundary_point[0] < I1.shape[1]:
            intersection_points.append(upper_boundary_point)

        if 0 < bottom_boundary_point[0] < I1.shape[1]:
            intersection_points.append(bottom_boundary_point)

        assert len(intersection_points) == 2

        ax1.plot([intersection_points[0][0], intersection_points[1][0]],
                 [intersection_points[0][1], intersection_points[1][1]], linewidth=2)

    plt.draw()
    plt.show()


sceneA_pts_a = load_file('input/sceneA/sceneA-pts2d-1.txt')
sceneA_pts_b = load_file('input/sceneA/sceneA-pts2d-2.txt')
sceneA_im1 = cv2.imread("input/sceneA/sceneA-im-1.png")
sceneA_im1 = cv2.cvtColor(sceneA_im1, cv2.COLOR_BGR2RGB)
sceneA_im2 = cv2.imread("input/sceneA/sceneA-im-2.png")
sceneA_im2 = cv2.cvtColor(sceneA_im2, cv2.COLOR_BGR2RGB)

plot_epipole_lines(sceneA_pts_a, sceneA_pts_b, sceneA_im1, sceneA_im2, F_sceneA_rank_2)

print('done')
