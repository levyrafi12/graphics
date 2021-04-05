from ray_tracer.scene_entities import Scene, Camera
import numpy as np


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def ray_casting(scene: Scene, width=500, height=500):
    camera = scene.camera
    towards_vector = normalize_vector(camera.look_at_3d - camera.pos_3d)
    screen_center_point = camera.pos_3d + camera.sc_dist * towards_vector

    right_vector = np.cross(towards_vector, camera.up_3d)
    # init RGB screen
    screen = np.zeros((height, width, 3))
    for i in range(width):
        for j in range(height):
            pixel = np.array([i, j, 0])
            camera_position = np.array([camera.pos_x, camera.pos_y, camera.pos_z])

            # ray = create_ray(camera, i, j)
            # intersection = find_intersection(ray, scene)
            # color = get_color(intersection)
            # screen[i][j][0] = color[0]
            # screen[i][j][1] = color[1]
            # screen[i][j][2] = color[2]


env_path = r"C:\dev\graphics\ray_tracer\scenes\Pool.txt"
out_path = r"C:\dev\graphics\ray_tracer\scenes\Pool_test.png"
scene = Scene(env_path, out_path)
ray_casting(scene)
