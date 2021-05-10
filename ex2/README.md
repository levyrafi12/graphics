# Fundamentals of Computer Graphics, Image Processing, and Vision 
# Exercise 2: Epipolar Geometry

In this exercise you will explore estimating the relationship between two camera views using the epipolar geometry concepts learned in the lectures. 
This exercise is split into two parts: 
1. Computing the fundamental matrix given two views of the same scene,
2. Computing the correspondences between the two scenes. 

Everything you will need for completing the exercise is given in the following files: 
- `graphics_epipolar_exercise.ipynb`
	- All of your code and text answers should be provided in this notebook. 
- `inputs`
	- A directory containing the following files: 
		- `sceneA`
			- `sceneA-pts2d-1.txt`
			- `sceneA-pts2d-2.txt`
			- `sceneA-im-1.png`
			- `sceneA-im-2.png`
		- `sceneB`
            - `sceneB-pts2d-1.txt`
			- `sceneB-pts2d-2.txt`
			- `sceneB-im-1.png`
			- `sceneB-im-2.png`
	- A detailed explanation of each file is provided below
- `tag_image.py`
	- A utility script explained in detailed below

In addition to the code implementation detailed below, there are several questions you are asked to answer. Your answers 
may be provided directly in the notebook or in a separate `.pdf` file.

**NOTE: the exercise should be completed in Python!**

All parts of this exercise can either be run on Google Colab or locally. If you're implementation is correct and 
efficient, each part should run in under a minute.  
As a reference, our implementation of **all** parts took less than 30 seconds. While we do not require you to be as efficient, 
this should give you a reference to understand if your implementation makes sense.

Some clarifications: 
1. You may use any built-in python libraries
2. You may use PIL/matplotlib/cv2 for reading, displaying, and saving images.
3. You may use numpy. In fact, it is **highly** recommended that you do so!
4. You may **not** use scientific libraries such as scipy.  
If you have any issues with installing these libraries, you may want to consider completing this exercise in Google Colab, which
comes with these libraries pre-installed.
In short, if you are not sure, ask.


## Part 1: Fundamental Matrix Estimation

In the first part of the exercise you will be asked to compute the Fundamental Matrix for two scenes, labeled `sceneA` and `sceneB`.   

To help you get started, we have provided you with correspondence points of the two images in `sceneA` in the files `sceneA-pts2d-1.txt` and `sceneA-pts2d-2.txt`. 
We have also provided you with the Fundamental Matrix you should obtain if you computed F correctly.  

At this point, you should be familiar enough with the code and the computation of the Fundamental Matrix. 
We therefore have provided you with an additional scene, `sceneB`. Here, you will need to pick the correspondences yourself using the utility 
script `tag_image.py`.
Running the script as demonstrated in class will generated two files in a similar format to 
the `.txt` we provided for you in `sceneA`. After computing the correspondences, compute the Fundamental Matrix for the given scene.
This time we do not provide the expected solution! 
Please note! You should **not** use the points found in the files `sceneB-pts2d-1.txt` and `sceneB-pts2d-2.txt` which are used 
only for the bonus! You should define your own correspondences using the utility script.

** Please make sure to answer the question at the end of Question 1!

After computing the Fundamental Matrix, we are ready to visualize the epipolar lines between the two images. 
In Question 3, you will be asked to draw the epipolar lines on both `sceneA` and `sceneB`. For each scene, you should 
display the two images side-by-side with the epipolar lines drawn on top of the images, like the following:

<p align="center">
<img src="docs/epipolar_lines_vizualization.jpg" width="800px"/>
</p>

Please save the resulting images (two in total - one for each scene) to a directory labeled `outputs`.


### Bonus! (5-10 pts)
Please refer to Part 1, Question 4 for a bonus question where we analyze some of the drawbacks of the Fundamental Matrix. 
More details are provided in the notebook.   
Note! If you choose to complete the bonus, please use the points found in files `sceneB-pts2d-1.txt` and `sceneB-pts2d-2.txt` for 
computing the correspondences of `sceneB`. This will allow us to easily verify your solution!   
Please save the results to the `outputs` directory using the name `bonus.png`. As before, the epipolar lines should be visualized 
side-by-side.


## Part 2: Finding the Correspondences
In this section, we will explore a window-based approach for estimating **dense stereo correspondence** and compute the 
resulting **disparity map**. 

For this part, we will be working with the images named `corr-img-l.png` and `corr-img-r.png` found in `input/correspondences/`. 
Additional details are provided directly in the notebook.

In Question 1, you are asked to explore different values for the (1) window size and (2) the maximum disparity range. 
You should explore at least `3` different values for each (i.e., `6` combinations in total). For each, please save an 
image named `disparty_[w]_[d].png` where `[w]` is the window size and `[d]` is the maximum disparity.  

In addition to saving the images, please clearly display and label the results in the notebook with the two disparity maps shown 
side-by-side.

** Please make sure to answer the questions at the end of Question 1 and Question 2!


## Submission
There are a couple of points to keep in mind before submitting this assignment: 
1. In `graphics_epipolar_exercise.ipynb` please specify the names and IDs of the person(s) who submitted the exercise. 
2. Please submit the notebook as an `ipynb` file and `html` file after running all the cells. Saving the notebook as an `html` file 
can be done by going to `File -> Download as -> HTML`. 
3. When submitting the notebook, please remove any unnecessary/debug cells. This is important so that we will be able to easily follow your code and results.
	a. We may also submit an additional `.pdf` file will your textual answers to the questions asked in the notebook. This is not required, but will help the grader to find your responses.
