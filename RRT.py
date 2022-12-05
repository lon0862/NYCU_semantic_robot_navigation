import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import random, sys, math, os.path
from matplotlib.pyplot import imread
import argparse
import copy

STEP_DISTANCE = 30 # Maximum distance between two vertex
MAP_IMG = './map.png' # Black and white image for a map
dpi = 50
map_offset_x = -3.5
map_offset_y = -5.5
img = imread(MAP_IMG)
min_x = img.shape[0]
max_x = 0
min_y = img.shape[1]
max_y = 0
radius = math.ceil(0.1*dpi)  # agent radius is 0.1m

def parse_args():
    parser = argparse.ArgumentParser(description='define target object')
    parser.add_argument('--target', type=str, default='refrigerator')
    args = parser.parse_args()
    return args

def rapidlyExploringRandomTree(ax, img, start, goal):
    Tree = []
    Tree.append(start)
    path = {}
    
    while(1):
        final_dist = math.sqrt((Tree[-1][0] - goal[0]) ** 2 + (Tree[-1][1] - goal[1]) ** 2)
        if(final_dist <= STEP_DISTANCE*2.5):
            ax.plot([Tree[-1][0], goal[0]], [Tree[-1][1], goal[1]], color='black') # draw black line for add_edge
            break
        
        point_random = selectRandomPoint(img)
        point_near, point_random = findNearestPoint(Tree, point_random)
        point_new = connectPoints(point_near, point_random, img)
        Tree, path = addToGraph(ax, point_new, point_near, Tree, path, img)

    # add final point before goal to path dictionary
    path[Tree[-1]] = [goal]
    final_path = [start]
    final_path, find = dfs(path, start, goal, final_path, False)

    # draw green line from start to goal
    for i in range(len(final_path)-1):
        ax.plot([final_path[i][0], final_path[i+1][0]], [final_path[i][1], final_path[i+1][1]], color='green') 
    
    return final_path

def selectRandomPoint(img):
    occupied = True
    while(occupied):
        point_random = [random.randint(min_y, max_y), random.randint(min_x, max_x)]
        if((img[point_random[1]][point_random[0]] * 255 == np.array([255, 255, 255, 255])).all()):
            occupied = checkRadius(point_random, img)
    return point_random
        
def findNearestPoint(Tree, point_random):
    min_distance = float("Inf")
    point_near = (0, 0)
    for point in Tree:
        if point == point_random:
            continue
        dist = math.sqrt((point[0] - point_random[0]) ** 2 + (point[1] - point_random[1]) ** 2)
        if (dist < min_distance and dist!=0):
            min_distance = dist
            point_near = (point[0], point[1])

    return point_near, point_random

def connectPoints(point_near, point_random, img):
    dist = math.sqrt((point_near[0] - point_random[0]) ** 2 + (point_near[1] - point_random[1]) ** 2)
    size = dist / STEP_DISTANCE
    if(size==0):
        print(point_near, point_random)
    X_direction = (point_random[0] - point_near[0])/size
    Y_direction = (point_random[1] - point_near[1])/size
    point_new = (point_near[0] + round(X_direction), point_near[1] + round(Y_direction)) 
    return point_new

def addToGraph(ax, point_new, point_near, Tree, path, img):
    dist = math.sqrt((point_new[0] - point_near[0]) ** 2 + (point_new[1] - point_near[1]) ** 2)
    X_direction = (point_new[0] - point_near[0])/dist
    Y_direction = (point_new[1] - point_near[1])/dist
    collision = False
    slide = 0
    while(1):
        if(collision==True):
            break
        
        slide += 1
        if(slide > dist):
            break
        
        point_check = point_near + np.around(np.array([X_direction, Y_direction])*slide)
        point_check = point_check.astype(int)
        if((img[point_check[1], point_check[0]]*255 == np.array([255, 255, 255, 255])).all()):
            collision = checkRadius(point_check, img)
            continue
        else:
            collision = True

    if(slide > dist):
        point_check = point_new
        if((img[point_check[1], point_check[0]]*255 == np.array([255, 255, 255, 255])).all()):
            collision = checkRadius(point_check, img)
        else:
            collision = True

    if(collision==False):
        Tree.append(point_new)
        if(point_near in path):
            path[point_near].append(point_new)
        else:
            path[point_near] = [point_new]
        ax.plot(point_new[0], point_new[1], 'm.') # draw purple point for add_point
        ax.plot([point_new[0], point_near[0]],[point_new[1], point_near[1]], color='black') # draw black line for add_edge

    return Tree, path

def dfs(path, start, goal, final_path, find):
    if(find == True):
        return final_path, find
    if(start in path):
        for point in path[start]:
            final_path.append(point)
            if(point==goal):
                find = True
                return final_path, find
            else:
                final_path, find = dfs(path, point, goal, final_path, find)
                if(find == True):
                    return final_path, find

    final_path.pop()
    return final_path, find

def checkRadius(point, img):
    occupied = True
    for i in range(radius):
        for j in range(radius):
            dist = math.sqrt(i**2 + j**2)
            if(dist<=radius):
                if((point[1]+j)<img.shape[1] and (point[0]+i)<img.shape[0]):
                    if((img[point[1]+j, point[0]+i]*255 == np.array([255, 255, 255, 255])).all()):
                        occupied = False
                    else:
                        return True
                if((point[1]+j)<img.shape[1] and (point[0]-i)>=0):
                    if((img[point[1]+j, point[0]-i]*255 == np.array([255, 255, 255, 255])).all()):
                        occupied = False
                    else:
                        return True
                if((point[1]-j)>=0 and (point[0]-i)>=0):
                    if((img[point[1]-j, point[0]-i]*255 == np.array([255, 255, 255, 255])).all()):
                        occupied = False
                    else:
                        return True
                if((point[1]-j)>=0 and (point[0]+i)<img.shape[0]):
                    if((img[point[1]-j, point[0]+i]*255 == np.array([255, 255, 255, 255])).all()):
                        occupied = False
                    else:
                        return True
    return occupied

def selectStartGoalPoints(ax, img):
    print ('Select a starting point')
    plt.title('Select a starting point')
    occupied = True
    while(occupied):
        point = plt.ginput(1, timeout=-1, show_clicks=False, mouse_pop=2)
        start = (round(point[0][0]), round(point[0][1]))
        if((img[start[1], start[0]]*255 == np.array([255, 255, 255, 255])).all()):
            occupied = checkRadius(start, img)
            if(not occupied):
                ax.plot(start[0], start[1], 'r*')
            else:
                print ('Cannot place a starting point there')
        else:
            print ('Cannot place a starting point there')
    
    return start

def pixelToWorld(point_pixel):
    tmp_u = point_pixel[0]/dpi + map_offset_x
    tmp_v = point_pixel[1]/dpi + map_offset_y
    point_world = (round(tmp_u, 3), round(tmp_v, 3))

    return point_world

if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    print ('Loading map... with file \'', MAP_IMG,'\'')

    '''
    init color_dict for target
    '''
    color_dict = {}
    color_dict["refrigerator"] = np.array([255, 0, 0, 255])
    color_dict["rack"] = np.array([0, 255, 133, 255])
    color_dict["cushion"] = np.array([255, 9, 92, 255])
    color_dict["lamp"] = np.array([160, 150, 20, 255])
    color_dict["cooktop"] = np.array([7, 255, 224, 255])

    args = parse_args()
    if args.target in color_dict:
        goal_color = color_dict[args.target]
        target = args.target
    else:
        print("target name is not right, default using refrigerator")
        goal_color = color_dict["refrigerator"]
        target = "refrigerator"

    # show img is (y,x) -> (500, 800), actually shape is (880, 500)
    print ('Map is', img.shape[1], 'x', img.shape[0])
    
    fig = plt.figure(figsize=(5.12, 5.12))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)

    target_arr = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            tmp_color = np.round(img[x, y]*255)
            if((tmp_color==goal_color).all()):
                target_arr.append([x,y])
                # ax.plot(y, x, 'k+')

            if((tmp_color == np.array([255, 255, 255, 255])).all()):
                continue
            else:
                if(x<min_x):
                    min_x = x
                if(y<min_y):
                    min_y = y
                if(x>max_x):
                    max_x = x
                if(y>max_y):
                    max_y = y

    find_goal = False
    while(not find_goal):
        target_mean = np.mean(target_arr, axis=0)
        target_var = np.var(target_arr, axis=0)
        target_dev = np.sqrt(target_var)
        dev_distance = np.linalg.norm(target_dev)

        if(dev_distance<STEP_DISTANCE*2):
            find_goal = True
        else:
            for i in range(len(target_arr)):
                dist = math.sqrt((target_arr[i][0] - target_mean[0]) ** 2 + (target_arr[i][1] - target_mean[1]) ** 2)
                if(dist>dev_distance):
                    target_arr.remove(target_arr[i])
                    find_goal = False
                    break
                else:
                    find_goal = True
        
        goal = (round(target_mean[1]), round(target_mean[0]))

    ax.plot(goal[0], goal[1], 'b*')
    start = selectStartGoalPoints(ax, img)

    final_path = rapidlyExploringRandomTree(ax, img, start, goal)
    for i in range(len(final_path)):
        final_path[i] = pixelToWorld(final_path[i])

    output_path = "path_output/world_coordinate/"+target+".txt"
    f = open(output_path, 'w+')
    f.writelines(str(final_path))
    f.close()
    print("write world coordinate to", output_path)
    img_path = "path_output/map_img/"+target+".png"
    
    plt.savefig(img_path)
    print("save map path to", img_path)
    plt.show()
    
    