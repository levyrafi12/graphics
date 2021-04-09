import math

import matplotlib.pyplot as plt
import numpy as np

from ray_tracer.scene_entities import Scene


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def find_intersections(ray_origin, ray_direction, scene: Scene):
    intersections = []

    for box in scene.boxes:
        pass

    for plane in scene.planes:
        denom = np.dot(plane.normal_vector, ray_direction)

        plane_point = plane.normal_vector * plane.offset

        vec = plane_point - ray_origin
        dist = np.dot(vec, plane.normal_vector) / denom

        if dist < 0:
            continue

        intersections.append([dist, plane])

    for sphere in scene.spheres:
        L = sphere.center_3d - ray_origin
        t_ca = np.dot(L, ray_direction)
        if t_ca < 0:
            continue  # no intersection

        d_power2 = np.dot(L, L) - t_ca ** 2

        r_power2 = sphere.radius ** 2
        if d_power2 > r_power2:
            continue  # the intersection is outside of the sphere

        t_hc = math.sqrt(r_power2 - d_power2)
        intersection_distance = min(t_ca - t_hc, t_ca + t_hc)

        intersections.append([intersection_distance, sphere])

    return intersections


def get_color(intersections, scene):
    if len(intersections) == 0:
        return np.array([0, 0, 0])  # return black

    intersection_object = min(intersections, key=lambda t: t[0])
    material = intersection_object[1].get_material(scene)

    return material.difuse_color


def ray_casting(scene: Scene, image_width=500, image_height=500):
    camera = scene.camera
    Vz = normalize_vector(camera.look_at_3d - camera.pos_3d)

    # set screen original point
    screen_center_point = camera.pos_3d + camera.sc_dist * Vz

    screen_aspect_ratio = image_width / image_height
    screen_width = camera.sc_width
    screen_height = screen_width / screen_aspect_ratio

    Vx = (normalize_vector(np.cross(camera.up_3d, Vz)) * screen_width) / image_width
    Vy = (normalize_vector(np.cross(Vx, Vz)) * screen_height) / image_height

    screen_orig_point = screen_center_point - (image_width / 2) * Vx - (image_height / 2) * Vy
     
    P0 = np.copy(screen_orig_point)
    screen = np.zeros((image_height, image_width, 3))

    for i in range(image_height):
        p = np.copy(P0)
        for j in range(image_width):
            ray_direction = normalize_vector(p - camera.pos_3d)
            intersections = find_intersections(camera.pos_3d, ray_direction, scene)
            color = get_color(intersections, scene)
            screen[i][j] = color
            p += Vx
        P0 += Vy

    plt.imshow(screen)
    plt.show()


def main():
    env_path = r"C:\dev\graphics\ray_tracer\scenes\Pool.txt"
    out_path = r"C:\dev\graphics\ray_tracer\scenes\Pool_test.png"
    scene = Scene(env_path, out_path)
    ray_casting(scene)


if __name__ == "__main__":
    main()
