from ray_tracer.scene_entities import Scene, Camera
import numpy as np


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


# Rotate matrix from world coord to view coord
def set_rotate_mat(toward_vec, camera):
    [a,b,c] = toward_vec
    Sx = -b
    Cx = np.sqrt(1 - Sx * Sx)
    Sy = -a / Cx
    Cy = c / Cx
    M = np.zeros((3,3))
    M[0] = [Cy, 0, Sy]
    M[1] = [-Sx * Sy, Cx, Sx * Cy]
    M[2] = [-Cx * Cy, -Sx, Cx * Cy]
    return M

def rotate_to_view_coord(M, toward_vec):
    Vx = np.matmul(np.array([1,0,0]), M)
    Vy = np.matmul(np.array([0,1,0]), M)
    Vz = np.matmul(np.array([0,0,1]), M)

    assert np.linalg.norm(toward_vec - Vz) <= 0.0001
    return Vx, Vy, Vz

def ray_casting(scene: Scene, width=500, height=500):
    camera = scene.camera
    # Tomer: towards_vector = normalize_vector(camera.look_at_3d - camera.pos_3d)
    # https://www.cs.tau.ac.il/~dcor/Graphics/cg-slides/view04.pdf (slide 3)
    # (P0 - P) / |P0 - P| where P0 is the eye/camera point and P is the look-at point
    towards_vector = normalize_vector(camera.pos_3d - camera.look_at_3d)
    screen_center_point = camera.pos_3d + camera.sc_dist * towards_vector

    right_vector = np.cross(towards_vector, camera.up_3d) # the U vector 
    # init RGB screen
    screen = np.zeros((height, width, 3))
    # define matrix M
    M = set_rotate_mat(camera, toward_vector)
    # rotate from world coord to view coord
    Vx, Vy, _ = rotate_to_view_coord(M, toward_vector)
    # set screen original point
    screen_orig_point = screen_center_point - width * Vx - height * Vy
     
    P0 = screen_orig_point
    camera_position = np.array([camera.pos_x, camera.pos_y, camera.pos_z])

    for i in range(height):
        p = P0
        for j in range(width):
            pixel = np.array([i, j, 0])
            ray = camera_position, p - camera_position
            # ray = create_ray(camera, i, j)
            # intersection = find_intersection(ray, scene)
            # color = get_color(intersection)
            # screen[i][j][0] = color[0]
            # screen[i][j][1] = color[1]
            # screen[i][j][2] = color[2]
            p += Vx
        P0 += Vy

env_path = r"C:\dev\graphics\ray_tracer\scenes\Pool.txt"
out_path = r"C:\dev\graphics\ray_tracer\scenes\Pool_test.png"
scene = Scene(env_path, out_path)
ray_casting(scene)
