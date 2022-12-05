# semantic_robot_navigation
In this repository, I complete Semantic Robot Navigation for the HW3 of Perception and Decision Making in Intelligent Systems, NYCU, in fall 2022.

# Abstact
In the HW2, we successfully reconstructed a 3D semantic map of apartment_0.
In this job, we will show how to navigate from point A to B on the first floor of apartment_0 using the RRT algorithm.

# Quick Start
The requirement of the development environments:
- OS : ubuntu 18.04 , 20.04
- Python 3.6, 3.7 ( You can use conda to create new environment )
- opencv-contrib-python
- Open3d
- Habitat-lab and Habitat-sim<br>
following the link https://github.com/facebookresearch/habitat-lab
to until Installation step3.<br>
Note: using under code to install habitat_lab
```
/home/{user_name}/anaconda3/envs/habitat/bin/pip install -e .
```

use a semantic 3d pointcloud of apartment_0's first floor in directory "semantic_3d_pointcloud"

# Structure of directory
```
habitat-lab
  ......
hw3
  +- semantic_3d_pointcloud
    +- color01.npy
    +- color0255.npy
    +- point.npy
  +- path_output
    +- map_img
      +- cooktop.png
      +- cushion.png
      +- lamp.png
      +- rack.png
      +- refrigerator.png
      
