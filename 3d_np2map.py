import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import csv
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import math

if __name__ == "__main__":
    points = np.load("semantic_3d_pointcloud/point.npy")
    colors = np.load("semantic_3d_pointcloud/color01.npy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # remove ceiling
    pcd = pcd.select_by_index(
        np.where(np.asarray(pcd.points)[:, 1] < 0)[0])
    # # remove floor
    pcd = pcd.select_by_index(
        np.where(np.asarray(pcd.points)[:, 1] > -0.035)[0])

    o3d.visualization.draw_geometries([pcd])
    points_dict = {}
    colors_dict = {}
    plt.figure(figsize=(5,8))

    plt.scatter(np.asarray(pcd.points)[:, 0]*10000/255, 
                -1*np.asarray(pcd.points)[:, 2]*10000/255, color=pcd.colors, s=1)
    plt.xlim(-3.5,6.5)
    plt.ylim(-10.5,5.5)
    # 使坐標軸等比例
    plt.gca().set_aspect('equal')
    # 關閉顯示座標軸
    plt.axis('off')
    # 去除坐標軸
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace= 0)
    plt.margins(0, 0)
    plt.savefig('map.png', pad_inches = 0)
    plt.show()
    print("end")
