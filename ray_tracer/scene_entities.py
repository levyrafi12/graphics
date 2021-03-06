from typing import Optional

import numpy as np


# General setting for a scene
class Sett:
	def __init__(self, params):
		self.bg_r = float(params[0])  # background colors (r,g,b)
		self.bg_g = float(params[1])
		self.bg_b = float(params[2])
		self.shadow_rays = int(params[3])  # (N = root number of shadow rays)
		self.rec_max = int(params[4])  # max number of recursions

	@property
	def background_color_3d(self):
		return np.array([self.bg_r, self.bg_g, self.bg_b])


class Camera:
	def __init__(self, params):
		self.pos_x = float(params[0])  # camera position (x,y,z)
		self.pos_y = float(params[1])
		self.pos_z = float(params[2])
		self.look_x = float(params[3])  # look at position (lx,ly,lz)
		self.look_y = float(params[4])
		self.look_z = float(params[5])
		self.up_x = float(params[6])  # up vector (ux,uy,uz)
		self.up_y = float(params[7])
		self.up_z = float(params[8]) 
		self.sc_dist = float(params[9])
		self.sc_width = float(params[10])
		# use fisheye if true o.w. use pinhole (optional)
		self.fisheye = params[11] == 'true' if len(params) > 11 else False 
		self.k_val = float(params[12]) if len(params) > 12 else 0.5  # optional

		self.pos_3d = np.array([self.pos_x, self.pos_y, self.pos_z])
		self.look_at_3d = np.array([self.look_x, self.look_y, self.look_z])
		self.up_3d = np.array([self.up_x, self.up_y, self.up_z])


class Plane:
	@staticmethod
	def create_from_params(params):
		return Plane(float(params[0]), float(params[1]), float(params[2]), float(params[3]), int(params[4]) - 1)

	def __init__(self, nx, ny, nz, offset, mat_ind):
		self.nx = nx  # normal vector (nx,ny,nz)
		self.ny = ny
		self.nz = nz
		self.offset = offset
		self.mat_ind = mat_ind

	def get_material(self, scene):
		return scene.materials[self.mat_ind]

	@property
	def normal_3d(self):
		return np.array([self.nx, self.ny, self.nz])


class Sphere:
	def __init__(self, params):
		self.cx = float(params[0])  # center point (cx,cy,cz)
		self.cy = float(params[1])
		self.cz = float(params[2])
		self.radius = float(params[3])
		self.mat_ind = int(params[4]) - 1

	@property
	def center_3d(self):
		return np.array([self.cx, self.cy, self.cz])

	def get_material(self, scene):
		return scene.materials[self.mat_ind]


class Box:
	def __init__(self, params):
		self.pos_x = float(params[0])  # position of center point (cx,cy,cz)
		self.pos_y = float(params[1])
		self.pos_z = float(params[2])
		self.edge_length = float(params[3])
		self.mat_ind = int(params[4]) - 1
		self.planes = []

		self.min_bound = self.center_3d - (self.edge_length / 2)
		self.max_bound = self.center_3d + (self.edge_length / 2)

		plane_point = self.center_3d + np.array([(self.edge_length / 2), 0, 0])
		plane_normal = np.sign(np.array([plane_point[0], 0, 0]))
		d = np.dot(plane_normal, plane_point)
		self.planes.append(Plane(plane_normal[0], plane_normal[1], plane_normal[2], d, self.mat_ind))

		plane_point = self.center_3d + np.array([0, (self.edge_length / 2), 0])
		plane_normal = np.sign(np.array([0, plane_point[1], 0]))
		d = np.dot(plane_normal, plane_point)
		self.planes.append(Plane(plane_normal[0], plane_normal[1], plane_normal[2], d, self.mat_ind))

		plane_point = self.center_3d + np.array([0, 0, (self.edge_length / 2)])
		plane_normal = np.sign(np.array([0, 0, plane_point[2]]))
		d = np.dot(plane_normal, plane_point)
		self.planes.append(Plane(plane_normal[0], plane_normal[1], plane_normal[2], d, self.mat_ind))

		plane_point = self.center_3d - np.array([(self.edge_length / 2), 0, 0])
		plane_normal = np.sign(np.array([plane_point[0], 0, 0]))
		d = np.dot(plane_normal, plane_point)
		self.planes.append(Plane(plane_normal[0], plane_normal[1], plane_normal[2], d, self.mat_ind))

		plane_point = self.center_3d - np.array([0, (self.edge_length / 2), 0])
		plane_normal = np.sign(np.array([0, plane_point[1], 0]))
		d = np.dot(plane_normal, plane_point)
		self.planes.append(Plane(plane_normal[0], plane_normal[1], plane_normal[2], d, self.mat_ind))

		plane_point = self.center_3d - np.array([0, 0, (self.edge_length / 2)])
		plane_normal = np.sign(np.array([0, 0, plane_point[2]]))
		d = np.dot(plane_normal, plane_point)
		self.planes.append(Plane(plane_normal[0], plane_normal[1], plane_normal[2], d, self.mat_ind))

	@property
	def center_3d(self):
		return np.array([self.pos_x, self.pos_y, self.pos_z])

	def get_material(self, scene):
		return scene.materials[self.mat_ind]


class Light:
	def __init__(self, params):
		self.pos_x = float(params[0])
		self.pos_y = float(params[1])
		self.pos_z = float(params[2])
		self.color_r = float(params[3])
		self.color_g = float(params[4])
		self.color_b = float(params[5])
		self.spec = float(params[6])  # specular intensity
		self.shadow = float(params[7])  # shadow intensity
		self.width = float(params[8])  # light width or radius (used for soft shadow)

	@property
	def pos_3d(self):
		return np.array([self.pos_x, self.pos_y, self.pos_z])

	@property
	def color_3d(self):
		return np.array([self.color_r, self.color_g, self.color_b])


class Material:
	def __init__(self, params):
		self.dr = float(params[0])  # diffuse color (r,g,b)
		self.dg = float(params[1])
		self.db = float(params[2])
		self.sr = float(params[3])  # specular color (r,g,b)
		self.sg = float(params[4])
		self.sb = float(params[5])
		self.rr = float(params[6])  # reflection color (r,g,b)
		self.rg = float(params[7])
		self.rb = float(params[8])
		self.phong = int(params[9])  # phong specularity coefficient (shininess)
		self.trans = float(params[10])  # transparency value between 0 and 1

	@property
	def difuse_color(self):
		return np.array([self.dr, self.dg, self.db])

	@property
	def spec_color(self):
		return np.array([self.sr, self.sg, self.sb])

	@property
	def reflection_color(self):
		return np.array([self.rr, self.rg, self.rb])


class Scene:
	def __init__(self, scene_file):
		self.planes = []
		self.boxes = []
		self.spheres = []
		self.lights = []
		self.materials = []
		self.camera = None  # type: Optional[Camera]
		self.sett = None
		self.parse(scene_file)

	def parse(self, scene_file):
		f = open(scene_file, "r")
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			if len(line) == 0 or line[0] == '#':
				continue
			params = line.split()
			if params[0] == 'cam':
				self.camera = Camera(params[1:])
			elif params[0] == 'set':
				self.sett = Sett(params[1:])
			elif params[0] == 'mtl':
				self.materials.append(Material(params[1:]))
			elif params[0] == 'pln':
				self.planes.append(Plane.create_from_params(params[1:]))
			elif params[0] == 'sph':
				self.spheres.append(Sphere(params[1:]))
			elif params[0] == 'box':
				self.boxes.append(Box(params[1:]))
			elif params[0] == 'lgt':
				self.lights.append(Light(params[1:]))
			else:
				print("Bad param {}".format(params[0]))


if __name__ == "__main__":
	env_path = r"scenes\Pool.txt"
	out_path = r"Pool_test.png"
	scene_object = Scene(env_path)
	print('done')
