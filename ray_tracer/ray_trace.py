
import sys

from scene_entities import Scene 

if __name__ == '__main__':
	print(sys.argv)
	scene_file = sys.argv[1]
	scene_out = sys.argv[2]
	print(scene_file, scene_out)
	scene = Scene(scene_file, scene_out)
	print("\"{}\"".format(scene))
