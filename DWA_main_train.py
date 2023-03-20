import math
import numpy as np
import time
from copy import deepcopy
from DWA import Path,Obstacle,RobotState,Robot,Costmap
import random
import zarr


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
n =   30      # how many time intervals


filename = "local_costmap_copy.txt"



def main(filename,min_v,max_v,min_w,max_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,
    heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,goal_region = 0.02):
    
    resolution = 0.05
    orig_px=30
    costmap = Costmap()
    initial_states_list = []
    costmap_list = []
    reward_list = []
    weights_list = []
    delta_heading = 0.5
    delta_obstacle = 0.3
    delta_velocity = 0.3

    for i in range(5):
        cm=np.zeros((orig_px*2,orig_px*2))
        rand_cm = np.random.randint(low=0, high=101, size=(40,40)) #Generation of random costmap
        cm[orig_px-20:orig_px+20,orig_px-20:orig_px+20]=rand_cm
        cm_rev = costmap.cm_rev(cm)
        cm_rev2 = costmap.cm_norm(cm_rev)
        obstacles = costmap.find_obstacles(cm_rev2)

        # Generation of initial states
        while True:
            
            init_x = random.uniform(-10, 10)
            init_y = random.uniform(-10, 10)
            goal_x = random.uniform(init_x - orig_px*resolution, init_x + orig_px*resolution)
            goal_y = random.uniform(init_y - orig_px*resolution, init_y + orig_px*resolution)
            low_lim_x = init_x - orig_px*resolution
            low_lim_y = init_y - orig_px*resolution 
            if ((goal_x - low_lim_x) / resolution) % 1 < 0.5:
                goal_x_pixel = math.floor((goal_x - low_lim_x) / resolution)
            else:
                goal_x_pixel = math.ceil((goal_x - low_lim_x) / resolution)
            if ((goal_y - low_lim_y) / resolution) % 1 < 0.5:
                goal_y_pixel = math.floor((goal_y - low_lim_y) / resolution)
            else:
                goal_y_pixel = math.ceil((goal_y - low_lim_y) / resolution)
            if (not((np.sqrt((goal_x-init_x)**2 + (goal_y-init_y)**2)) < goal_region))\
                 and (not(cm_rev2[30][30]<0.03)) and (not(cm_rev2[goal_x_pixel][goal_y_pixel]<0.03)):
                break

        init_theta = random.uniform(-(2 * np.pi), (2 * np.pi)) 
        init_v = random.uniform(min_v, max_v)
        init_w = random.uniform(min_w, max_w)



        heading_cost_weight = random.uniform(heading_cost_weight_base - delta_heading, heading_cost_weight_base + delta_heading)
        obstacle_cost_weight = random.uniform(obstacle_cost_weight_base - delta_obstacle, obstacle_cost_weight_base + delta_obstacle)
        velocity_cost_weight = random.uniform(velocity_cost_weight_base - delta_velocity, velocity_cost_weight_base + delta_velocity)

        robot = Robot(cm_rev2,min_v,max_v,min_w,max_w,max_a_v,max_a_w,max_dec_v,max_dec_w,delta_v,delta_w,dt,n,
                        heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,orig_px,init_x,init_y)
        state = RobotState(init_x,init_y,init_theta,init_v,init_w)

        init_state = [init_x,init_y,init_theta,goal_x,goal_y,init_v,init_w]
        weights = [heading_cost_weight, obstacle_cost_weight, velocity_cost_weight]

        # obs_x, obs_y = robot.obstacle_position(obstacles,state)
        # obs_x, obs_y = robot.obs_pos_trial(obstacles)


        resolution = 0.05

        num_cycle = 0
        num_cycle_max = 300 #deneme yanılma dogrusunu bul!!

        while True:
            
            paths,opt_path,failFlag = robot.calc_opt_traj(goal_x,goal_y,state,obstacles,goal_region)

            # velocity commands
            if failFlag:
                reward_temp = -30
                reward_list.append(reward_temp)
                costmap_list.append(rand_cm)
                initial_states_list.append(init_state)
                weights_list.append(weights)
                break 

            opt_v = opt_path.v   
            opt_w = opt_path.w 
            # print("Optimal velocities are: ({},{})".format((opt_v),(opt_w)))
            x,y,theta = state.update_state(opt_v,opt_w,dt)


            x_pixel = robot.meter2pixel(state.x,state,'x')
            y_pixel = robot.meter2pixel(state.y,state,'y')
            if cm_rev2[x_pixel][y_pixel]<0.03:
                reward_temp = -30
                reward_list.append(reward_temp)
                costmap_list.append(rand_cm)
                initial_states_list.append(init_state)
                weights_list.append(weights)
                break

            dis_to_goal = np.sqrt((goal_x-state.x)**2 + (goal_y-state.y)**2)
            if dis_to_goal < goal_region:
                print("Goal!!")
                # goal_Flag = True
                reward_temp = 100
                reward_list.append(reward_temp)
                costmap_list.append(rand_cm)
                initial_states_list.append(init_state)
                weights_list.append(weights)
                break

            if num_cycle < num_cycle_max:
                num_cycle += 1
            else:
                reward_temp = 0
                reward_list.append(reward_temp)
                costmap_list.append(rand_cm)
                initial_states_list.append(init_state)
                weights_list.append(weights)
                break


    zarr.save('costmap_list.zarr', costmap_list)
    zarr.save('initial_states_list.zarr', initial_states_list)
    zarr.save('weights_list.zarr', weights_list)
    zarr.save('reward_list.zarr', reward_list)

    # np.savetxt('costmap_list.txt', costmap_list)
    # np.savetxt('initial_states_list.txt', initial_states_list)
    # np.savetxt('weights_list.txt', weights_list)
    # np.savetxt('reward_list.txt', reward_list)
                

heading_cost_weight_base = 0.8
obstacle_cost_weight_base = 0.1
velocity_cost_weight_base = 0.1
goal_region = 0.1

print("Start!!")
# start_time = time.perf_counter()
main(filename,min_v,max_v,min_w,max_w,max_a_v,max_a_w,delta_v,delta_w,dt,n,
        heading_cost_weight_base,obstacle_cost_weight_base,velocity_cost_weight_base,goal_region)
# end_time = time.perf_counter()

# print("Run time = {} msec".format(1000*(end_time-start_time)))