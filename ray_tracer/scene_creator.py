import math
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from scene_entities import Scene

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def length_vector(vector):
    return np.linalg.norm(vector)

def dot_product(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def arbitrary_vector_in_plane(normal, D, xyz):
    V = np.zeros(3)
    for i in range(3):
        if normal[i] != 0:
            V[i] = -D / normal[i]
            break
    return normalize_vector(xyz - V)


def reflected_vector(V, N):
    return V - 2 * dot_product(V, N) * N

def get_reflection_color(V, intersect_object, scene, rec_depth):
    intersect_surface = intersect_object[2]
    ref_color = intersect_surface.get_material(scene).reflection_color
    bg = scene.sett.background_color_3d
    eps = 1e-5
    if np.sum(ref_color) < eps:
        return np.zeros(3)

    N = intersect_object[3]
    R = reflected_vector(V, N)

    intersect_point = intersect_object[1]
    # shifted_point = intersect_point + 1e-5 * N
    # intersections = find_intersections(intersect_point, R, scene)
    intersections = find_intersections(intersect_point, R, scene)
    if intersections == []:
        return ref_color * bg

    nearest_object = intersections[0]
    assert(nearest_object[2] != intersect_surface)

    return ref_color * get_color(R, intersections, scene, rec_depth - 1)

def get_diff_spec_color(V, intersect_object, scene):
    soft_shadow_flag = True

    # list of point light intensity (Ip)
    if soft_shadow_flag: 
        lig_intensity_list = soft_shadow(intersect_object, scene)
    else:
         lig_intensity_list = [1] * len(scene.lights)

    N = intersect_object[3]  # surface normal
    intersect_point = intersect_object[1]

    color = np.zeros(3)
    Kd = intersect_object[2].get_material(scene).difuse_color
    Ks = intersect_object[2].get_material(scene).spec_color
    n = intersect_object[2].get_material(scene).phong

    for i, light in enumerate(scene.lights):
        # Idiff = Kd * Ip * dot(N,P)
        L = normalize_vector(light.pos_3d - intersect_point)
        cos_theta = dot_product(N, L)
        if cos_theta > 0:
            color += Kd * cos_theta * lig_intensity_list[i] * light.color_3d
        # Ispec = Ks * Ip * dot(H,N) ** n
        R = 2 * dot_product(L, N) * N - L
        cos_phi = dot_product(-V, R)
        if cos_phi > 0:
            color += Ks * lig_intensity_list[i] * np.power(cos_phi, n) * light.color_3d * light.spec

    return color

# If the point is reached immediately, return 1
# If an opaque object is on the way, return 0
# If point is reached through transparent objects return a value between 0 and 1
# that is equal to the trans coeff multiplication of all the transparent objects 
# being crossed
def shadow_reach_via_trans_objs(li_intersections, intersect_obj, scene):
    intersect_point = intersect_obj[1]
    intersect_surface = intersect_obj[2]
    eps = 1e-5
    trans_val = 1

    for i, li_intersect_obj in enumerate(li_intersections):
        li_intersect_obj = li_intersections[i]
        if li_intersect_obj[2] != intersect_surface: 
            if li_intersect_obj[2].get_material(scene).trans <= 0:
                return 0
        else:        
            if length_vector(intersect_point - li_intersect_obj[1]) < eps:
                return trans_val
            else:
                return 0
        trans_val *= li_intersect_obj[2].get_material(scene).trans
       
    return trans_val

def hard_shadow(intersect_obj, light, scene):
    eps = 1e-5
    intersect_point = intersect_obj[1]
    ray = normalize_vector(intersect_point - light.pos_3d)
    intersections = find_intersections(light.pos_3d, ray, scene)
    assert intersections != []

    nearest_object = intersections[0]
    if nearest_object[2] == intersect_obj[2]: 
        if length_vector(intersect_point - nearest_object[1]) < eps:
            return (1 - light.shadow) * 1 + light.shadow * 1


    trans_value = shadow_reach_via_trans_objs(intersections, intersect_obj, scene)

    return 1 - light.shadow + trans_value


def soft_shadow(intersect_object, scene):
    intersect_point = intersect_object[1]
    light_intensity_list = []
    bonus_flag = True
   

    N = scene.sett.shadow_rays
    for light in scene.lights:
        L = normalize_vector(intersect_point - light.pos_3d)
        # coefficients of perpendicular plane to the ray
        # from light to intersection point
        D = -dot_product(light.pos_3d, L)

        if light.width <= 0:
            light_intensity = hard_shadow(intersect_object, light, scene)
            light_intensity_list.append(light_intensity)
            continue

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

                if bonus_flag:
                    num_hits += shadow_reach_via_trans_objs(intersections, intersect_object, scene)
                    continue
                li_intersect_obj = intersections[0]
                if li_intersect_obj[2] == intersect_object[2]: 
                    if length_vector(intersect_point - li_intersect_obj[1]) < eps:
                        num_hits += 1

        hit_ratio = num_hits / (N * N)
        light_intensity = ((1 - light.shadow) + light.shadow * hit_ratio)
        light_intensity_list.append(light_intensity)

    return light_intensity_list


def get_plane_intersection(ray_origin, ray_direction, plane):
    N = plane.normal_3d
    d = -plane.offset
    t = -(dot_product(ray_origin, N) + d) / dot_product(ray_direction, N)
    if t <= 1e-4:
        return None

    intersection_point = ray_origin + t * ray_direction
    return t, intersection_point, plane, plane.normal_3d 


def is_intersected(box, ray_origin, ray_direction):
    invdir = 1 / ray_direction

    sign = (invdir < 0).astype(np.int)

    bounds = [box.min_bound, box.max_bound]
    tmin = (bounds[sign[0]][0] - ray_origin[0]) * invdir[0]
    tmax = (bounds[1 - sign[0]][0] - ray_origin[0]) * invdir[0]
    tymin = (bounds[sign[1]][1] - ray_origin[1]) * invdir[1]
    tymax = (bounds[1 - sign[1]][1] - ray_origin[1]) * invdir[1]

    if ((tmin > tymax) or (tymin > tmax)):
        return False, 0

    if (tymin > tmin):
        tmin = tymin
    if (tymax < tmax):
        tmax = tymax

    tzmin = (bounds[sign[2]][2] - ray_origin[2]) * invdir[2]
    tzmax = (bounds[1 - sign[2]][2] - ray_origin[2]) * invdir[2]

    if ((tmin > tzmax) or (tzmin > tmax)):
        return False, 0

    if (tzmin > tmin):
        tmin = tzmin
    if (tzmax < tmax):
        tmax = tzmax

    t = tmin

    if (t < 0):
        t = tmax
        if (t < 0):
            return False, 0

    if t <= 1e-4:
        return False, 0

    return True, t


def find_intersections(ray_origin, ray_direction, scene: Scene):
    intersections = []

    for box in scene.boxes:
        return_value = is_intersected(box, ray_origin, ray_direction)
        if return_value[0]:
            intersection_point = ray_origin + return_value[1] * ray_direction
            for plane in box.planes:
                intersect_obj = get_plane_intersection(ray_origin, ray_direction, plane)
                if intersect_obj is not None:
                    if np.allclose(intersect_obj[1], intersection_point):
                        intersections.append(intersect_obj)

    for plane in scene.planes:
        intersect_obj = get_plane_intersection(ray_origin, ray_direction, plane)
        if intersect_obj is not None:
            intersections.append(intersect_obj)

    for sphere in scene.spheres:
        # geometric method
        L = sphere.center_3d - ray_origin
        t_ca = dot_product(L, ray_direction)
        if t_ca < 0:
            continue  # no intersection

        d_power2 = dot_product(L, L) - t_ca ** 2

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
    if intersections == [] or rec_depth <= 0:
        return scene.sett.background_color_3d

    intersect_object = intersections[0]
    trans = intersect_object[2].get_material(scene).trans
    trans_color = np.zeros(3)
    if trans > 0:
        trans_color = get_color(trace_ray, intersections[1:], scene, rec_depth - 1)
    diff_spec = get_diff_spec_color(trace_ray, intersect_object, scene)
    ref_color = get_reflection_color(trace_ray, intersect_object, scene, rec_depth)

    return trans * trans_color + (1 - trans) * diff_spec + ref_color


def trace_ray_from_camera(intersections, scene):
    if intersections == []:
        return scene.sett.background_color_3d

    V = normalize_vector(intersections[0][1] - scene.camera.pos_3d)
    return get_color(V, intersections, scene, scene.sett.rec_max)

def ray_casting(scene: Scene, image_width=500, image_height=500):
    print(time.ctime())
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
            if camera.fisheye:
                sensor_radius = length_vector(p - screen_center_point)

                f = camera.sc_dist
                k = camera.k_val
                if 0 < k <= 1:
                    theta = np.arctan((k * sensor_radius) / f) / k
                elif k == 0:
                    theta = sensor_radius / f
                elif -1 <= k < 0:
                    theta = np.arcsin((k * sensor_radius) / f) / k
                else:
                    raise Exception("not supported k")

                # check degrees
                if theta < np.pi / 2:
                    image_radius = math.tan(theta) * camera.sc_dist

                    mid_to_point = normalize_vector(p - screen_center_point)

                    new_point = screen_center_point + mid_to_point * image_radius

                    ray_direction_fish = normalize_vector(new_point - camera.pos_3d)

                    intersections = find_intersections(camera.pos_3d, ray_direction_fish, scene)
                    color = trace_ray_from_camera(intersections, scene)
                    screen[i][j] = np.clip(color, 0, 1)

            else:  # not fisheye
                ray_direction_straight = normalize_vector(p - camera.pos_3d)
                intersections = find_intersections(camera.pos_3d, ray_direction_straight, scene)
                color = trace_ray_from_camera(intersections, scene)
                screen[i][j] = np.clip(color, 0, 1)
            p += Vx
        P0 += Vy
        if i > 0 and i % 50 == 0:
            print(time.ctime())

    return screen


def main():
    scene_file_path = sys.argv[1]
    out_path = sys.argv[2]
    if len(sys.argv) > 3:
        image_width = int(sys.argv[3])
    else:
        image_width = 500
    if len(sys.argv) > 4:
        image_height = int(sys.argv[4])
    else:
        image_height = 500

    scene = Scene(scene_file_path)
    screen = ray_casting(scene, image_width, image_height)

    plt.imsave(out_path, screen)

if __name__ == "__main__":
    main()
