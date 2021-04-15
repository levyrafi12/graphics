import math

import matplotlib.pyplot as plt
from scene_entities import Scene, Camera
import numpy as np
import random
import time

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

def find_nearest_object(intersections):
    if len(intersections) == 0:
        return None

    return min(intersections, key=lambda t: t[0])
   
def arbitrary_vector_in_plane(normal, D, intersect_point):
    V = np.zeros(3)
    for i in range(3):
        if normal[i] != 0:
            V[i] = -normal[i] / D
    return normalize_vector(V - intersect_point)

def soft_shadow(intersect_object, scene):
    intersect_point = intersect_object[1]

    N = scene.sett.shadow_rays
    light_intensity = 0
    for light in scene.lights:
        L = normalize_vector(intersect_point - light.pos_3d)
        # coefficients of perpendicular plane to the ray 
        # from light to intersection point
        D = -np.dot(light.pos_3d, L)
        A,B,C = L[0], L[1], L[2]

        V1 = arbitrary_vector_in_plane(L, D, intersect_point) / light.width
        V2 = np.cross(V1, L) / light.width
        square_origin_point = light.pos_3d - (light.width / 2) * V1 - (light.width / 2) * V2
        num_hits = 0
        cell_edge = light.width / N
        P0 = np.copy(square_origin_point)

        for i in range(N):
            P = np.copy(P0)
            P += V2 * random.uniform(0, cell_edge)
            for j in range(N):
                P += V1 * random.uniform(0, cell_edge)
                # P[2] = (-A * P[0]  -B * P[1] - D) / C
                ray = normalize_vector(intersect_point - P)
                intersections = find_intersections(P, ray, scene)
                object = find_nearest_object(intersections)
                if object == None:
                    continue
                if object[2] == intersect_object[2]:
                    num_hits += 1
                P += cell_edge * V1
            P0 += cell_edge * V2

        hit_ratio = num_hits / (N * N)
        opaque_ratio = (1 - intersect_object[2].get_material(scene).trans)
        light_intensity += ((1 - light.shadow) + light.shadow * hit_ratio) * opaque_ratio

        return light_intensity

def find_plane_intersections(ray_origin, ray_direction, scene: Scene):
    intersections = []

    for plane in scene.planes:
        N = plane.normal_3d
        d = -plane.offset
        t = -(np.dot(ray_origin, N) + d) / np.dot(ray_direction, N)
        intersection_point = ray_origin + t * ray_direction
        intersections.append((t, intersection_point, plane))

    return intersections

def find_intersections2(ray_origin, ray_direction, scene: Scene):
    intersections = []

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
        t = min(t_ca - t_hc, t_ca + t_hc) # distance
        intersection_point = ray_origin + t * ray_direction
        intersections.append((t, intersection_point, sphere))

    plane_intersects = find_plane_intersections(ray_origin, ray_direction, scene)

    return intersections + plane_intersects

# ray_origin + t * ray_direction (P = P0 + t * V)
def find_intersections(ray_origin, ray_direction, scene: Scene):
    intersections = []

    for sphere in scene.spheres:
        # algebric method
        L = ray_origin - sphere.center_3d
        b = 2 * np.dot(ray_direction, L)

        c = np.linalg.norm(L) ** 2 - sphere.radius ** 2

        delta = b ** 2 - 4 * c
        if delta > 0:
            delta_sqrt = np.sqrt(delta)
            t1 = (-b + delta_sqrt) / 2
            t2 = (-b - delta_sqrt) / 2
            if t1 > 0 and t2 > 0:
                t = min(t1, t2) # distance
                intersection_point = ray_origin + t * ray_direction
                intersections.append((t, intersection_point, sphere))

    plane_intersects = find_plane_intersections(ray_origin, ray_direction, scene)

    return intersections + plane_intersects

def get_color(intersections, scene):

    intersection_object = find_nearest_object(intersections)
    if intersection_object == None:
        return np.array([0, 0, 0])  # return black

    material = intersection_object[2].get_material(scene)

    light_intensity = 1
    light_intensity = soft_shadow(intersection_object, scene)

    return material.difuse_color * light_intensity

# Rotate matrix from world coord to view coord
def set_rotate_mat(toward_vec):
    [a, b, c] = toward_vec
    Sx = -b
    Cx = np.sqrt(1 - Sx * Sx)
    Sy = -a / Cx
    Cy = c / Cx
    M = np.array([[Cy, 0, Sy],
                  [-Sx * Sy, Cx, Sx * Cy],
                  [-Cx * Cy, -Sx, Cx * Cy]])
    return M


def rotate_to_view_coord(M):
    Vx = np.matmul(np.array([1,0,0]), M)
    Vy = np.matmul(np.array([0,1,0]), M)
    Vz = np.matmul(np.array([0,0,1]), M)

    return Vx, Vy, Vz

def ray_casting(scene: Scene, image_width=500, image_height=500):
    # print(time.ctime())
    camera = scene.camera
    Vz = normalize_vector(camera.look_at_3d - camera.pos_3d) # towards

    # set screen original point
    screen_center_point = camera.pos_3d + camera.sc_dist * Vz

    screen_aspect_ratio = image_width / image_height
    screen_width = camera.sc_width
    screen_height = screen_width / screen_aspect_ratio

    Vx = (normalize_vector(np.cross(camera.up_3d, Vz)) * screen_width) / image_width # right
    Vy = (normalize_vector(np.cross(Vx, Vz)) * screen_height) / image_height

    screen_orig_point = screen_center_point - (image_width / 2) * Vx - (image_height / 2) * Vy
     
    P0 = np.copy(screen_orig_point)
    screen = np.zeros((image_height, image_width, 3))

    for i in range(image_height):
        if i % 50 == 0:
             print(time.ctime())
        p = np.copy(P0)
        for j in range(image_width):
            ray_direction = normalize_vector(p - camera.pos_3d)
            # ray = create_ray(camera, i, j)
            intersections = find_intersections(camera.pos_3d, ray_direction, scene)
            color = get_color(intersections, scene)
            screen[i][j] = np.clip(color, 0, 1)
            p += Vx
        P0 += Vy

    # plt.imshow(np.ones((500, 500, 3)))
    # plt.show()

    print(time.ctime())
    plt.imshow(screen)
    plt.show()

env_path = r"scenes\Pool.txt"
out_path = r"scenes\Pool_test.png"
scene = Scene(env_path, out_path)
ray_casting(scene)


