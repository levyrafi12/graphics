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


def get_diff_spec_color(intersect_object, scene):
    soft_shadow_flag = True

    # list of point light intensity (Ip)
    if soft_shadow_flag:
        light_intensity_list = soft_shadow(intersect_object, scene)
    else:
         light_intensity_list = [1] * len(scene.lights)

    N = intersect_object[3] # surface normal
    intersect_point = intersect_object[1]

    color = np.zeros(3)
    Kd = intersect_object[2].get_material(scene).difuse_color
    Ks = intersect_object[2].get_material(scene).spec_color
    V = normalize_vector(scene.camera.pos_3d - intersect_point)
    n = intersect_object[2].get_material(scene).phong

    for i, light in enumerate(scene.lights):
        # Idiff = Kd * Ip * dot(N,P)
        L = -normalize_vector(intersect_point - light.pos_3d)
        cos_theta = np.dot(N, L)
        if cos_theta > 0:
            color += Kd * cos_theta * light_intensity_list[i] * light.color_3d
        # Ispec = Ks * Ip * dot(H,N) ** n
        H = normalize_vector(V + L)
        cos_phi = np.dot(H, N)
        if cos_phi > 0:
            color += Ks * light_intensity_list[i] * np.power(cos_phi, n) * light.color_3d
    return color


def soft_shadow(intersect_object, scene):
    intersect_point = intersect_object[1]
    light_intensity_list = []
    eps = 1e-10

    N = scene.sett.shadow_rays
    for light in scene.lights:
        light_intensity = 0
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
                li_intersect_obj = find_nearest_object(intersections)
                if object is None:
                    continue
                if li_intersect_obj[2] == intersect_object[2]:
                    if np.linalg.norm(intersect_point - li_intersect_obj[1]) < eps:
                        num_hits += 1

        hit_ratio = num_hits / (N * N)
        transparency = intersect_object[2].get_material(scene).trans
        light_intensity = ((1 - light.shadow) + light.shadow * hit_ratio) * (1 - transparency)
        light_intensity_list.append(light_intensity)

    return light_intensity_list


def get_plane_intersection(ray_origin, ray_direction, plane):
    N = plane.normal_3d
    d = -plane.offset
    t = -(np.dot(ray_origin, N) + d) / np.dot(ray_direction, N)
    intersection_point = ray_origin + t * ray_direction
    return t, intersection_point, plane, plane.normal_3d


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
        N = normalize_vector(intersection_point - sphere.center_3d)
        intersections.append((t, intersection_point, sphere, N))

    return intersections


def get_color(intersections, scene):

    intersection_object = find_nearest_object(intersections)
    if intersection_object is None:
        return np.array([0, 0, 0])  # return black

    return get_diff_spec_color(intersection_object, scene)


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
