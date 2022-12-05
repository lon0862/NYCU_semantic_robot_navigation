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

# Task 1: 2D semantic map construction
use a semantic 3d pointcloud of apartment_0's first floor in directory "semantic_3d_pointcloud".</br>
run the following command you can convert 3d point cloud to 2d map(map.png), which remove ceiling and floor.</br>
![image](https://github.com/lon0862/semantic_robot_navigation/blob/main/map.png)
```
python 3d_np2map.py
```
# Task 2:RRT
run the following command, you can assign a target first, then click a point as start point, final you will get a RRT path.</br>
Note: Our target categories are refrigerator, rack, cushion, lamp and cooktop.</br>
if you need other target you can refrence color.csv.</br>
```
python RRT.py --target {target name}
```
# Task 3:Robot Navigation
run the following command, you can assign a target first, then you will get robot navigation in habitat with RRT path.</br>
And it will save as video.</br>
```
python robot_navigation.py --target {target name}
```

# Structure of directory
```
habitat-lab
  ......
hw3
  +- 3d_np2map.py
  +- robot_navigation.py
  +- RRT.py
  +- map.png
  +- color.csv
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
    +- RGB_img
      +- cooktop
        +- 0.png
        ......
      +- cushion
        ......
      +- lamp
        ......
      +- rack
        ......
      +- refrigerator
        ......
    +- video
      +- cooktop.mp4
      +- cushion.mp4
      +- lamp.mp4
      +- rack.mp4
      +- refrigerator.mp4
    +- word_coordinate
      +- cooktop.txt
      +- cushion.txt
      +- lamp.txt
      +- rack.txt
      +- refrigerator.txt
