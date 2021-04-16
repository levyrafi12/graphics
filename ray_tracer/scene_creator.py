import math

import matplotlib.pyplot as plt
import numpy as np
import random
import time

from ray_tracer.scene_entities import Scene


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def find_nearest_object(intersections):
    if len(intersections) == 0:
        return None

    return min(intersections, key=lambda t: t[0])


def arbitrary_vector_in_plane(normal, D, xyz):
    V = np.zeros(3)
    for i in range(3):
        if normal[i] != 0:
            V[i] = -D / normal[i]
            break
    return normalize_vector(xyz - V)


def soft_shadow(intersect_object, scene):
    intersect_point = intersect_object[1]

    N = scene.sett.shadow_rays
    light_intensity = 0
    for light in scene.lights:
        L = normalize_vector(intersect_point - light.pos_3d)
        # coefficients of perpendicular plane to the ray
        # from light to intersection point
        D = -np.dot(light.pos_3d, L)

        V1 = arbitrary_vector_in_plane(L, D, light.pos_3d) / light.width
        V2 = np.cross(V1, L) / light.width
        # square origin point
        P0 = light.pos_3d - (light.width / 2) * V1 - (light.width / 2) * V2
        num_hits = 0
        cell_edge = light.width / N

        for i in range(N):
            for j in range(N):
                P = P0 + V2 * (i * cell_edge + random.uniform(0, cell_edge))
                P = P + V1 * (j * cell_edge + random.uniform(0, cell_edge))
                ray = normalize_vector(intersect_point - P)
                intersections = find_intersections(P, ray, scene)
                object = find_nearest_object(intersections)
                if object is None:
                    continue
                if object[2] == intersect_object[2]:
                    num_hits += 1

        hit_ratio = num_hits / (N * N)
        opaque_ratio = (1 - intersect_object[2].get_material(scene).trans)
        light_intensity += ((1 - light.shadow) + light.shadow * hit_ratio) * opaque_ratio

        return light_intensity


def get_plane_intersection(ray_origin, ray_direction, plane):
    N = plane.normal_3d
    d = -plane.offset
    t = -(np.dot(ray_origin, N) + d) / np.dot(ray_direction, N)
    intersection_point = ray_origin + t * ray_direction
    return t, intersection_point, plane


def find_intersections(ray_origin, ray_direction, scene: Scene):
    intersections = []

    for box in scene.boxes:
        pass

    for plane in scene.planes:
        intersections.append(get_plane_intersection(ray_origin, ray_direction, plane))

    for sphere in scene.spheres:
        # geometric method
        L = sphere.center_3d - ray_origin
        t_ca = np.dot(L, ray_direction)
        if t_ca < 0:
            continue  # no intersection

        d_power2 = np.dot(L, L) - t_ca ** 2

        r_power2 = sphere.radius ** 2
        if d_power2 > r_power2:
            continue  # the intersection is outside of the sphere

        t_hc = math.sqrt(r_power2 - d_power2)
        t = min(t_ca - t_hc, t_ca + t_hc)  # distance
        intersection_point = ray_origin + t * ray_direction
        intersections.append((t, intersection_point, sphere))

    return intersections


def get_color(intersections, scene):

    intersection_object = find_nearest_object(intersections)
    if intersection_object is None:
        return np.array([0, 0, 0])  # return black

    material = intersection_object[2].get_material(scene)

    light_intensity = 1
    light_intensity = soft_shadow(intersection_object, scene)

    return material.difuse_color * light_intensity


def rotation_vector_around_axis(rotation_radians, vec, rotation_axis):
    from scipy.spatial.transform import Rotation as R

    rotation_vector = rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vec)
    return rotated_vec


def ray_casting(scene: Scene, image_width=500, image_height=500):
    # print(time.ctime())
    camera = scene.camera
    Vz = normalize_vector(camera.look_at_3d - camera.pos_3d)  # towards

    # set screen original point
    screen_center_point = camera.pos_3d + camera.sc_dist * Vz

    screen_aspect_ratio = image_width / image_height
    screen_width = camera.sc_width
    screen_height = screen_width / screen_aspect_ratio

    Vx = (normalize_vector(np.cross(camera.up_3d, Vz)) * screen_width) / image_width  # right
    Vy = (normalize_vector(np.cross(Vx, Vz)) * screen_height) / image_height

    screen_orig_point = screen_center_point - (image_width / 2) * Vx - (image_height / 2) * Vy
     
    P0 = np.copy(screen_orig_point)
    screen = np.zeros((image_height, image_width, 3))

    for i in range(image_height):
        p = np.copy(P0)
        for j in range(image_width):
            ray_direction_straight = normalize_vector(p - camera.pos_3d)

            if camera.fisheye:
                radius = np.linalg.norm(p - camera.pos_3d) ** 2 - camera.sc_dist ** 2
                if radius > 0:
                    radius = np.sqrt(radius)

                    f = camera.sc_dist
                    k = camera.k_val

                    theta = np.arctan((k * radius) / f) / k

                    # check degrees
                    if np.abs(theta * 180 / np.pi) < 90:
                        ray_direction_fish = rotation_vector_around_axis(theta, Vz, np.cross(ray_direction_straight, Vz))

                        intersections = find_intersections(camera.pos_3d, ray_direction_fish, scene)
                        color = get_color(intersections, scene)
                        screen[i][j] = np.clip(color, 0, 1)

            else:  # not fisheye
                intersections = find_intersections(camera.pos_3d, ray_direction_straight, scene)
                color = get_color(intersections, scene)
                screen[i][j] = np.clip(color, 0, 1)

            p += Vx
        P0 += Vy

    print(time.ctime())
    plt.imshow(screen)
    plt.show()


def main():
    env_path = r"scenes\Pool.txt"
    out_path = r"scenes\Pool_test.png"
    scene = Scene(env_path, out_path)
    ray_casting(scene)


if __name__ == "__main__":
    main()
