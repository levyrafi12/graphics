the ex zip contain all the required python files: 
RayTracer.py contains the ray tracing algorithm
scene_entities.py contain all the scene objects and parsing

the scenes png files are inside scene_images folder
the scene text files are inside scene_texts folder

example for running our script: (scene_file_path, image_out_path, width, height)
python RayTracer.py scenes\Spheres.txt scenes\Spheres.png 500 500
where the width and the hight parameters are 500 by default

regarding the original fish eye scene we combined the use of the boxes and the use of the fish eye
and built our names acronyms TA and RL with boxes rendered using the fisheye (under scene_texts folder)

we used matplotlib to save the rendered image as png and numpy

We have implemented the 5 points bonus - shadow ray will be considered
as a hit even if it crosses through one or more transparent objects before 
it reaches the target point (the intersection point either by the eye, 
reflection or transparency ray). So far, prior to the bonus, for a given shadow ray, 
the hit value was either 0 or 1. Now, the hit value can be a value between 0 to 1. 
Assume the shadow ray crosses two transparent objects with coeff trans1 and trans2, 
before reaching the target point, then the hit value will be trans1 * trans2. 

Tomer Amit & Rafi Levy