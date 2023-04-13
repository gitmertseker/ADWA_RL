import math
import numpy as np
# import time
from copy import deepcopy
# from DWA import Path,Obstacle,RobotState,Robot,Costmap
import random
import zarr
import cv2
from numba import njit
# from numba.experimental import jitclass

#robot parameters
min_v = 0  # minimum translational velocity
max_v = 0.1  # maximum translational velocity
min_w = -math.pi/4    # minimum angular velocity
max_w = math.pi/4   # maximum angular velocity




def readImageMap(path,resize_constant):
    img = cv2.imread(path)
    dimensions = (int(resize_constant*img.shape[0]),int(resize_constant*img.shape[1]))
    img = cv2.resize(img,dimensions)
    # img = cv2.transpose(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img_gray_mod = deepcopy(img_gray)
    img_gray_mod_2 = deepcopy(img_gray)

    # fp = disk(5) #5 olacak!!
    # img_gray_mod_2 = erosion(img_gray_mod,fp) #obstacle size enlarged

    np.putmask(img_gray_mod_2, img_gray_mod_2<205, 0) #occupied
    np.putmask(img_gray_mod_2, img_gray_mod_2>205, 1) #free
    np.putmask(img_gray_mod_2, img_gray_mod_2==205, 1) #unknown

    # np.putmask(img_gray, img_gray<205, 0) #occupied
    # np.putmask(img_gray, img_gray>205, 1) #free
    # np.putmask(img_gray, img_gray==205, 1) #unknown
    # self.image = deepcopy(img_gray_mod_2)

    return img_gray_mod_2

@njit

def states_generation(img,min_v,max_v,min_w,max_w):

    resolution = 0.05
    # resolution = 0.02
    
    orig_px=20
    # orig_px = 200 # deneme??
    heading_cost_weight_base = 0.8
    obstacle_cost_weight_base = 0.1
    velocity_cost_weight_base = 0.1
    # goal_region = 0.3
    goal_region = 0.1
    # costmap = Costmap()
    initial_states_list = []
    costmap_list = []
    # reward_list = []
    weights_list = []
    delta_heading = 0.5
    delta_obstacle = 0.3
    delta_velocity = 0.3
    init_goal_range = 1
    n_sets = 3000000

    # img = costmap.readImageMap(path,resize_constant=1/10)


    for i in range(n_sets):

        # Generation of initial states
        while 1:
            
            low_x = (0 + orig_px) * resolution
            low_y = (0 + orig_px) * resolution
            high_x = (img.shape[0] - orig_px) * resolution
            high_y = (img.shape[1] - orig_px) * resolution
            init_x = round(random.uniform(low_x, high_x),3)
            init_y = round(random.uniform(low_y, high_y),3)

            x_range = (img.shape[0]) * resolution
            y_range = (img.shape[1]) * resolution


            if init_x + init_goal_range < x_range:
                high_lim_x = init_x + init_goal_range
            else:
                high_lim_x = x_range
            
            if init_x - init_goal_range > 0:
                low_lim_x = init_x - init_goal_range
            else:
                low_lim_x = 0


            if init_y + init_goal_range < y_range:
                high_lim_y = init_y + init_goal_range
            else:
                high_lim_y = y_range

            if init_y - init_goal_range > 0:
                low_lim_y = init_y - init_goal_range
            else:
                low_lim_y = 0


            goal_x = round(random.uniform(low_lim_x, high_lim_x),3)
            goal_y = round(random.uniform(low_lim_y, high_lim_y),3)


            if (init_x % resolution) < (resolution/2):
                init_x_pixel = math.floor(init_x / resolution)
            else:
                init_x_pixel = math.ceil(init_x / resolution)

            if (init_y % resolution) < (resolution/2):
                init_y_pixel = math.floor(init_y / resolution)
            else:
                init_y_pixel = math.ceil(init_y / resolution)

            if (goal_x % resolution) < (resolution/2):
                goal_x_pixel = math.floor(goal_x / resolution)
            else:
                goal_x_pixel = math.ceil(goal_x / resolution)

            if (goal_y % resolution) < (resolution/2):
                goal_y_pixel = math.floor(goal_y / resolution)
            else:
                goal_y_pixel = math.ceil(goal_y / resolution)


            if (not((np.sqrt((goal_x-init_x)**2 + (goal_y-init_y)**2)) < goal_region))\
                 and (not(img[init_x_pixel][init_y_pixel]<0.03)) and (not(img[goal_x_pixel][goal_y_pixel]<0.03)):
                break

        init_theta = round(random.uniform(-(2 * np.pi), (2 * np.pi)),3)
        init_v = round(random.uniform(min_v, max_v),3)
        init_w = round(random.uniform(min_w, max_w),3)

        # cm = img[init_x_pixel-orig_px:init_x_pixel+orig_px, init_y_pixel-orig_px:init_y_pixel+orig_px]
        cm_init = img[init_x_pixel-orig_px:init_x_pixel+orig_px, init_y_pixel-orig_px:init_y_pixel+orig_px]

        # obstacles = costmap.find_obstacles(cm_init)


        heading_cost_weight = round(random.uniform(heading_cost_weight_base - delta_heading, heading_cost_weight_base + delta_heading),2)
        obstacle_cost_weight = round(random.uniform(-obstacle_cost_weight_base, obstacle_cost_weight_base + delta_obstacle),3)
        velocity_cost_weight = round(random.uniform(-velocity_cost_weight_base, velocity_cost_weight_base + delta_velocity),3)


        init_state = [init_x,init_y,init_theta,goal_x,goal_y,init_v,init_w]
        weights = [heading_cost_weight, obstacle_cost_weight, velocity_cost_weight]
        initial_states_list.append(init_state)
        weights_list.append(weights)
        costmap_list.append(cm_init)
        print(i)

    return costmap_list,initial_states_list,weights_list


path = '4training.png'
resize_constant=1/10
img = readImageMap(path,resize_constant)
costmap_list,initial_states_list,weights_list = states_generation(img,min_v,max_v,min_w,max_w)

zarr.save('costmap_list.zarr', costmap_list)
zarr.save('initial_states_list.zarr', initial_states_list)
zarr.save('weights_list.zarr', weights_list)