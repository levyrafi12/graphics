import math

import matplotlib.pyplot as plt
import numpy as np
import random
import time

# from ray_tracer.scene_entities import Scene
from scene_entities import Scene

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def arbitrary_vector_in_plane(normal, D, xyz):
    V = np.zeros(3)
    for i in range(3):
        if normal[i] != 0:
            V[i] = -D / normal[i]
            break
    return normalize_vector(xyz - V)


def reflected_vector(V, N):
    return V - 2 * np.dot(V, N) * N


def get_transparency_color(trace_ray, intersections, scene, rec_depth):
    bg = scene.sett.background_color_3d
    intersect_object = intersections[0]   
    trans = intersect_object[2].get_material(scene).trans
    if rec_depth <= 0 or trans <= 0:
        return bg

    behind_intersections = intersections[1:]
    return get_color(trace_ray, behind_intersections, scene, rec_depth - 1)

def get_reflection_color(V, intersect_object, scene, rec_depth):
    intersect_surface = intersect_object[2]
    ref_color = intersect_surface.get_material(scene).reflection_color
    bg = scene.sett.background_color_3d
    if rec_depth <= 0:
        return bg * ref_color

    # V = normalize_vector(camera.pos_3d - intersect_point)
    N = intersect_object[3]
    R = reflected_vector(V, N)

    intersect_point = intersect_object[1]
    # shifted_point = intersect_point + 1e-5 * N
    intersections = find_intersections(intersect_point, R, scene)
    if intersections == []:
        return ref_color * bg

    nearest_object = intersections[0] 
    if nearest_object[2] == intersect_surface: # maybe a bug
        return np.zeros(3)
    
    return ref_color * get_color(R, intersections, scene, rec_depth - 1)

def get_diff_spec_color(intersect_object, scene):
    soft_shadow_flag = True

    # list of point light intensity (Ip)
    if soft_shadow_flag: 
        lig_intensity_list = soft_shadow(intersect_object, scene)
    else:
         lig_intensity_list = [1] * len(scene.lights) 

    N = intersect_object[3] # surface normal
    intersect_point = intersect_object[1]

    color = np.zeros(3)
    Kd = intersect_object[2].get_material(scene).difuse_color
    Ks = intersect_object[2].get_material(scene).spec_color
    n = intersect_object[2].get_material(scene).phong

    for i, light in enumerate(scene.lights):
        # Idiff = Kd * Ip * dot(N,P)
        L = normalize_vector(light.pos_3d - intersect_point)
        cos_theta = np.dot(N, L)
        if cos_theta > 0:
            color += Kd * cos_theta * lig_intensity_list[i] * light.color_3d
        # Ispec = Ks * Ip * dot(H,N) ** n
        V = normalize_vector(scene.camera.pos_3d - intersect_point)
        H = normalize_vector(V + L)
        cos_phi = np.dot(H, N)
        if cos_phi > 0:
            color += Ks * lig_intensity_list[i] * np.power(cos_phi, n) * light.color_3d * light.spec

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
                if intersections == []:
                    continue
                li_intersect_obj = intersections[0]
                if li_intersect_obj[2] == intersect_object[2]: 
                    if np.linalg.norm(intersect_point - li_intersect_obj[1]) < eps:
                        num_hits += 1

        hit_ratio = num_hits / (N * N)
        light_intensity = ((1 - light.shadow) + light.shadow * hit_ratio)
        light_intensity_list.append(light_intensity)

    return light_intensity_list


def get_plane_intersection(ray_origin, ray_direction, plane):
    N = plane.normal_3d
    d = -plane.offset
    t = -(np.dot(ray_origin, N) + d) / np.dot(ray_direction, N)
    if t <= 1e-4:
        return None

    intersection_point = ray_origin + t * ray_direction
    return t, intersection_point, plane, plane.normal_3d 


def find_intersections(ray_origin, ray_direction, scene: Scene):
    intersections = []

    for box in scene.boxes:
        pass

    for plane in scene.planes:
        intersect_obj = get_plane_intersection(ray_origin, ray_direction, plane)
        if intersect_obj != None:
            intersections.append(intersect_obj)

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

    return sorted(intersections, key=lambda t : t[0])


def get_color(trace_ray, intersections, scene, rec_depth):
    if intersections == []:
        return np.array([0, 0, 0])

    intersect_object = intersections[0]
    trans = intersect_object[2].get_material(scene).trans
    trans_color = get_transparency_color(trace_ray, intersections, scene, rec_depth)
    diff_spec = get_diff_spec_color(intersect_object, scene)
    ref_color = get_reflection_color(trace_ray, intersect_object, scene, rec_depth)
   
    return  trans * trans_color + (1 - trans) * diff_spec + ref_color

def trace_ray_from_camera(intersections, scene):
    bg = scene.sett.background_color_3d
    if intersections == []:
        return bg

    V = normalize_vector(intersections[0][1] - scene.camera.pos_3d)
    return get_color(V, intersections, scene, scene.sett.rec_max)

def ray_casting(scene: Scene, image_width=500, image_height=500):
    print(time.ctime())
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
        p = np.copy(P0)
        for j in range(image_width):
            ray_direction = normalize_vector(p - camera.pos_3d)
            intersections = find_intersections(camera.pos_3d, ray_direction, scene)
            color = trace_ray_from_camera(intersections, scene)
            screen[i][j] = np.clip(color, 0, 1)
            p += Vx
        P0 += Vy
        if i > 0 and i % 50 == 0:
            print(time.ctime())

    plt.imshow(screen)
    plt.show()


def main():
    env_path = r"scenes\Spheres.txt"
    out_path = r"scenes\Pool_test.png"
    scene = Scene(env_path, out_path)
    ray_casting(scene)


if __name__ == "__main__":
    main()
