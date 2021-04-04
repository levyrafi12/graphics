
# General setting for a scene
class Sett:
	def __init__(self, params):
		self.bg_r = float(params[0]) # background colors (r,g,b)
		self.bg_g = float(params[1])
		self.bg_b = float(params[2])
		self.shadow_rays = float(params[3]) # (N = root number of shadow rays)
		self.rec_max = float(params[4]) # max number of recursions

class Cam:
	def __init__(self, params):
		self.pos_x = float(params[0]) # camera position (x,y,z)
		self.pos_y= float(params[1])
		self.pos_z = float(params[2])
		self.look_x = float(params[3]) # look at position (lx,ly,lz)
		self.look_y = float(params[4])
		self.look_z = float(params[5])
		self.up_x = float(params[6]) # up vector (ux,uy,uz)
		self.up_y = float(params[7])
		self.up_z = float(params[8]) 
		self.sc_dist = float(params[9])
		self.sc_width = float(params[10])
		self.fisheye = params[11] == '1' # use fisheye if true o.w. use pinhole
		self.k_val = float(params[12]) if len(params) > 12 else 0.5 # optional

class Plane:
	def __init__(self, params):
		self.nx = float(params[0]) # normal vector (nx,ny,nz)
		self.ny = float(params[1])
		self.nz = float(params[2])
		self.offset = float(params[3])
		self.mat_ind = int(params[4])

class Sphere:
	def __init__(self, params):
		self.cx = float(params[0]) # center point (cx,cy,cz)
		self.cy = float(params[1])
		self.cz = float(params[2])
		self.radius = float(params[3])
		self.mat_ind = int(params[4])

	def get_material(self, scene):
		return scene.materials[self.mat_ind]

class Box:
	def __init__(self, params):
		self.pos_x = float(params[0]) # position of center point (cx,cy,cz)
		self.pos_y = float(params[1])
		self.pos_z = float(params[2])
		self.edge = float(params[3])
		self.mat_ind = int(params[4])

class Light:
	def __init__(self, params):
		self.pos_x = float(params[0])
		self.pos_y = float(params[1])
		self.pos_z = float(params[2])
		self.color_r = int(params[3]) 
		self.color_g = int(params[4])
		self.color_b = int(arams[5])
		self.spec = float(arams[6]) # specular intensity
		self.shadow = float(params[7]) # shadow intensity
		self.width = float(params[8]) # light width or radius (used for soft shadow)

def line_to_params(line):
	line = line.strip()
	return line.split(' ').remove(' ')

class Scene:
	def __init__(self, scene_file, scene_out):
		self.planes = []
		self.boxes = []
		self.spheres = []
		self.lights = []
		self.materials = []
		self.scene_out = scene_out
		self.parse(scene_file)

	def parse(self, scene_file):
		print(scene_file)
		f = open(scene_file, "r")
		lines = f.readlines()
		for line in lines:
			print(line)
			if line.strip() == "":
				continue
			params = line_to_params(line)
			if params[0] == '#':
				continue
			if params[0] == 'cam':
				self.cam = Camera(params[1:])
			elif params[0] == 'set':
				self.sett = Sett(params[1:])
			elif params[0] == 'mtl':
				self.materials.append(Material(params[1:]))
			elif params[0] == 'pln':
				self.planes.append(Plane(params[1:]))
			elif params[0] == 'sph':
				self.spheres.append(Sphere(params[1:]))
			elif params[0] == 'box':
				self.boxes.append(Box(params[1:]))
			elif params[0] == 'lgt':
				self.lights.append(Light(params[1:]))
			else:
				print("Bad param {}".format(params[0]))