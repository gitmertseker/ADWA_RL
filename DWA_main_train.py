import math
import numpy as np
# import time
from copy import deepcopy
from DWA import Path,Obstacle,RobotState,Robot,Costmap
import random
import zarr
import cv2
from numba import njit
from numba.experimental import jitclass

#robot parameters
min_v = 0  # minimum translational velocity
max_v = 0.1  # maximum translational velocity
min_w = -math.pi/4    # minimum angular velocity
max_w = math.pi/4   # maximum angular velocity
max_a_v = 0.05 
  # maximum translational acceleration/deceleration
max_a_w = 90 * math.pi /180  # maximum angular acceleration/deceleration
max_dec_v = max_a_v
max_dec_w = max_a_w
delta_v = 0.1/2  # increment of translational velocity # window length / interval
delta_w = np.deg2rad(18/4)  # increment of angular velocity
dt =  0.1  # time step
# dt =  0.04  # time step
n =   30      # how many time intervals



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

# @njit

def main(min_v,max_v,min_w,max_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,
    cm_list,st_list,w_list,goal_region,img):
    
    resolution = 0.05
    # resolution = 0.02
    
    orig_px=20
    # orig_px = 200 # deneme??
    # for cm,st,w in zip(cm_list,st_list,w_list):
    # r_list = np.zeros((1,len(cm_list)))
    reward_list = []

    for i in range(len(cm_list)):

        print(i)
        init_x,init_y,init_theta,goal_x,goal_y,init_v,init_w = st_list[i]
        heading_cost_weight, obstacle_cost_weight, velocity_cost_weight = w_list[i]
        cm = cm_list[i]

        # img = costmap.readImageMap(path,resize_constant=1/10)


        robot = Robot(min_v,max_v,min_w,max_w,max_a_v,max_a_w,max_dec_v,max_dec_w,delta_v,delta_w,dt,n,
                        heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,orig_px,init_x,init_y)
        state = RobotState(init_x,init_y,init_theta,init_v,init_w)

        # obs_x, obs_y = robot.obstacle_position(obstacles,state)
        # obs_x, obs_y = robot.obs_pos_trial(obstacles)

        num_cycle = 0
        num_cycle_max = 300 #deneme yanılma dogrusunu bul!!

        while 1:
            
            paths,opt_path,failFlag = robot.calc_opt_traj(goal_x,goal_y,state,goal_region,cm)

            # velocity commands
            if failFlag:
                reward_temp = -30
                reward_list.append(reward_temp)
                break 

            opt_v = opt_path.v   
            opt_w = opt_path.w 
            # print("Optimal velocities are: ({},{})".format((opt_v),(opt_w)))
            x,y,theta = state.update_state(opt_v,opt_w,dt)
            cm = state.update_costmap(img,state.x,state.y,resolution,orig_px)

            # obstacles = costmap.find_obstacles(cm)


            # x_pixel = robot.meter2pixel(state.x,state,'x')
            # y_pixel = robot.meter2pixel(state.y,state,'y')
            x_pixel = orig_px
            y_pixel = orig_px

            if cm[x_pixel][y_pixel]<0.03:
                reward_temp = -30
                reward_list.append(reward_temp)
                break

            dis_to_goal = np.sqrt((goal_x-state.x)**2 + (goal_y-state.y)**2)
            if dis_to_goal < goal_region:
                # print("Goal!!")
                # goal_Flag = True
                reward_temp = 100
                reward_list.append(reward_temp)
                break

            if num_cycle < num_cycle_max:
                num_cycle += 1
            else:
                reward_temp = 0
                reward_list.append(reward_temp)
                break

    zarr.save('D:/Python_Projects/ADWA_RL/unknown_obstacle/reward_list_200_corrected.zarr', reward_list)



# goal_region = 0.3
goal_region = 0.1

path = '4training.png'
resize_constant=1/10
# print("Start!!")
# start_time = time.perf_counter()

img = readImageMap(path,resize_constant)
cm_list = zarr.load('D:/Python_Projects/ADWA_RL/200_no_reward/costmap_list.zarr')
st_list = zarr.load('D:/Python_Projects/ADWA_RL/200_no_reward/initial_states_list.zarr')
w_list = zarr.load('D:/Python_Projects/ADWA_RL/200_no_reward/weights_list.zarr')

""
main(min_v,max_v,min_w,max_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,
        cm_list,st_list,w_list,goal_region,img)
# end_time = time.perf_counter()

# print("Run time = {} msec".format(1000*(end_time-start_time)))