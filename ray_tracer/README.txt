the ex zip contain all the required python files: 
RayTracer.py contains the ray tracing algorithm
scene_entities.py contain all the scene objects and parsing

the scenes png files are inside scene_images folder
the scene text files are inside scene_text folder

example for running our script: (scene_file_path, image_out_path, width, height)
python RayTracer.py scenes\Spheres.txt scenes\Spheres.png 500 500
where the width and the hight parameters are 500 by default

regarding the original fish eye scene we combined the use of the boxes and the use of the fish eye
and built our names acronyms TA and RL with boxes rendered using the fisheye

we used matplotlib to save the rendered image as png and numpy