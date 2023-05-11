import math
import numpy as np
# from copy import deepcopy
# import matplotlib.pyplot as plt
import cv2
# from skimage.morphology import erosion,disk
# from numba.experimental import jitclass
# from numba import float32, boolean, int32, int8
import numba

# spec = [('x', float32),
#         ('y', float32),
#         ('theta', float32),
#         ('v', float32),
#         ('w', float32),
#         ('admissibility', boolean)]


# @jitclass(spec)
class Path:
    def __init__(self,v,w):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = v
        self.w = w
        self.admissibility = True

# spec1 = [('x', float32),
#         ('y', float32)]

# @jitclass(spec1)
class Obstacle:
    def __init__(self,x,y):
        self.x = x
        self.y = y


# @jitclass
class Costmap:

    def __init__(self):
        pass


    def readImageMap(self,path,resize_constant):
        img = cv2.imread(path)
        dimensions = (int(resize_constant*img.shape[0]),int(resize_constant*img.shape[1]))
        img = cv2.resize(img,dimensions)
        # img = cv2.transpose(img)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # img_gray_mod = deepcopy(img_gray)
        # img_gray_mod_2 = deepcopy(img_gray)

        # fp = disk(5) #5 olacak!!
        # img_gray_mod_2 = erosion(img_gray_mod,fp) #obstacle size enlarged

        # np.putmask(img_gray_mod_2, img_gray_mod_2<205, 0) #occupied
        # np.putmask(img_gray_mod_2, img_gray_mod_2>205, 1) #free
        # np.putmask(img_gray_mod_2, img_gray_mod_2==205, 1) #unknown

        np.putmask(img_gray, img_gray<205, 0) #occupied
        np.putmask(img_gray, img_gray>205, 1) #free
        np.putmask(img_gray, img_gray==205, 1) #unknown
        # self.image = deepcopy(img_gray_mod_2)

        return img_gray

    def read_costmap(self,filename):

        with open(filename, "r") as f:
            x=[]
            
            for line in f.readlines():
                x.append(line[:-1])
            y=[]
            line=""
            for i in range(0,len(x),1):
                line=line+x[i]
                if x[i][-1]==']':
                    line=line.replace('   ',' ')
                    line=line.replace('  ',' ')
                    line=line.replace('[ ','[')
                    line=line.replace(' ]',']')
                    b=line[1:-1]
                    l=b.split(' ')
                    y.append(list(map(int,l) ))
                    line=""
        return y


    def find_obstacles(self,cm):
        obstacles = []
        for z in range(len(cm[0])):
            for i in range(len(cm[0])):
                if cm[i][z] < 0.03:
                    obs_temp = Obstacle(i,z)
                    obstacles.append(obs_temp)
        return obstacles


    # def cm_rev(self,cm):
    #     cm_rev = deepcopy(cm)
    #     for i in range (0,len(cm[0])):
    #         for z in range(0,len(cm[1])):
    #             cm_rev[i][z] = 100-cm[i][z]
    #     return cm_rev


    def cm_norm(self,cm_rev):


        return [[i/100 for i in row] for row in cm_rev]



    def find_index(l, elem):
        for row, i in enumerate(l):
            try:
                column = i.index(elem)
            except ValueError:
                continue
            return row, column
        return -1

"""
spec2 = [('x', float32),
        ('y', float32),
        ('theta', float32),
        ('v', float32),
        ('w', float32),
        ('traj_x', float32[:]),
        ('traj_y', float32[:]),
        ('traj_theta', float32[:]),
        ('dt', float32)]"""

# @jitclass(spec2)
class RobotState:
    def __init__(self,init_x,init_y,init_theta,init_v,init_w):

        self.x = init_x
        self.y = init_y
        self.theta = init_theta

        self.v = init_v
        self.w = init_w

        # self.traj_x = [init_x]
        # self.traj_y = [init_y]
        # self.traj_theta = [init_theta]

    def update_state(self,v,w,dt):

        self.v = v
        self.w = w
        self.dt = dt

        next_x = (self.v * math.cos(self.theta) * self.dt) + self.x
        next_y = (self.v * math.sin(self.theta) * self.dt) + self.y
        next_theta = (self.w * self.dt) + self.theta

        # self.traj_x.append(next_x)
        # self.traj_y.append(next_y)
        # self.traj_theta.append(next_theta)

        self.x = next_x
        self.y = next_y
        self.theta = next_theta

        return self.x, self.y, self.theta


    # def traj(self):
    #     return self.traj_x,self.traj_y

    def update_costmap(self, img, x, y, resolution, orig_px):
        x_pixel = math.floor(x / resolution) if (x % resolution) < (resolution / 2) else math.ceil(x / resolution)
        y_pixel = math.floor(y / resolution) if (y % resolution) < (resolution / 2) else math.ceil(y / resolution)

        x_min = max(0, x_pixel - orig_px)
        x_max = min(img.shape[0], x_pixel + orig_px)
        y_min = max(0, y_pixel - orig_px)
        y_max = min(img.shape[1], y_pixel + orig_px)

        cm = img[x_min:x_max, y_min:y_max]

        if x_min != x_pixel - orig_px:
            padding = np.zeros(((orig_px - x_pixel + x_min), cm.shape[1]))
            cm = np.concatenate((padding, cm), axis=0)
        if x_max != x_pixel + orig_px:
            padding = np.zeros(((x_pixel + orig_px - x_max), cm.shape[1]))
            cm = np.concatenate((cm, padding), axis=0)
        if y_min != y_pixel - orig_px:
            padding = np.zeros((cm.shape[0], (orig_px - y_pixel + y_min)))
            cm = np.concatenate((padding, cm), axis=1)
        if y_max != y_pixel + orig_px:
            padding = np.zeros((cm.shape[0], (y_pixel + orig_px - y_max)))
            cm = np.concatenate((cm, padding), axis=1)

        return cm

"""
spec3 = [('init_x', float32),
        ('init_y', float32),
        ('delta_v', float32),
        ('delta_w', float32),
        ('max_dec_v', float32),
        ('max_dec_w', float32),
        ('max_a_v', float32),
        ('max_a_w', float32),
        ('max_v', float32),
        ('max_w', float32),
        ('min_v', float32),
        ('min_w', float32),
        # ('costmap', int32[:,:]),
        ('costmap', int8[:,:]),
        ('v_count', int32),
        ('w_count', int32),
        ('heading_cost_weight', float32),
        ('obstacle_cost_weight', float32),
        ('velocity_cost_weight', float32),
        ('velocity_cost_weight', float32),
        # ('traj_paths', float32[:,:]),
        ('traj_opt', float32[:]),
        ('origin_pixel', int32),
        ('n', int32),
        ('cur_v', float32),
        ('cur_w', float32),
        ('min_v_dw', float32),
        ('min_w_dw', float32),
        ('max_v_dw', float32),
        ('max_w_dw', float32),
        ('temp_cost', float32),
        ('temp_obs', float32),
        ('dt', float32)]
        """

# @jitclass(spec3)
class Robot:
    def __init__(self,min_v,max_v,min_w,max_w,max_a_v,max_a_w,max_dec_v,max_dec_w,delta_v,delta_w,dt,n,
                heading_cost_weight,obstacle_cost_weight,velocity_cost_weight,orig_px,init_x,init_y):
        #robot parameters

        self.min_v = min_v       # minimum translational velocity
        self.max_v = max_v       # maximum translational velocity
        self.min_w = min_w       # minimum angular velocity
        self.max_w = max_w       # maximum angular velocity
        self.max_a_v = max_a_v     # maximum translational acceleration/deceleration
        self.max_a_w = max_a_w     # maximum angular acceleration/deceleration
        self.max_dec_v = max_dec_v
        self.max_dec_w = max_dec_w
        self.delta_v = delta_v      #increment of velocity
        self.delta_w = delta_w     #increment of angular velocity
        self.dt = dt          # time step
        self.n = n           #how many time intervals

        self.init_x = init_x
        self.init_y = init_y

        self.v_count = 8
        self.w_count = 8

        self.heading_cost_weight = heading_cost_weight

        self.obstacle_cost_weight = obstacle_cost_weight
        self.velocity_cost_weight = velocity_cost_weight

        # self.traj_paths = []
        # self.traj_opt = []

        self.origin_pixel = orig_px


    def angle_correction(self,angle):

        if angle > math.pi:
            while angle > math.pi:
                angle -=  2 * math.pi
        elif angle < -math.pi:
            while angle < -math.pi:
                angle += 2 * math.pi

        return angle


    def calc_dw(self,state):

        self.cur_v = state.v
        self.cur_w = state.w

        Vs = [self.min_v, self.max_v, self.min_w, self.max_w]  #maximum velocity area
        Vd = [(self.cur_v-self.max_a_v*self.dt), (self.cur_v+self.max_a_v*self.dt),
                 (self.cur_w-self.max_a_w*self.dt), (self.cur_w+self.max_a_w*self.dt)]   #velocities robot can generate until next time step
        min_v = max(Vs[0], Vd[0])
        max_v = min(Vs[1], Vd[1])
        min_w = max(Vs[2], Vd[2])
        max_w = min(Vs[3], Vd[3])
        dw = [min_v, max_v, min_w, max_w]

        return dw



    def predict_state(self,v,w,x,y,theta,dt,n):

        next_x_s = []
        next_y_s = []
        next_theta_s = []

        for i in range(n):
            temp_x = (v * math.cos(theta) * dt) + x
            temp_y = (v * math.sin(theta) * dt) + y
            temp_theta = w * dt + theta

            next_x_s.append(temp_x)
            next_y_s.append(temp_y)
            next_theta_s.append(temp_theta)

            x = temp_x
            y = temp_y
            theta = temp_theta

        return next_x_s, next_y_s, next_theta_s


    def calc_opt_traj(self,goal_x,goal_y,state,goal_region,costmap):

        paths = self.make_path(state)
        # paths = self.check_path_velo(paths,obstacles)
        opt_path,failFlag = self.eval_path(paths,goal_x,goal_y,state,goal_region,costmap)
        
        # self.traj_opt.append(opt_path)

        return paths, opt_path, failFlag


    def make_path(self,state):

        dw = self.calc_dw(state)
        self.min_v_dw = dw[0]
        self.max_v_dw = dw[1]
        self.min_w_dw = dw[2]
        self.max_w_dw = dw[3]


        paths = []

        for w in np.linspace(self.min_w_dw,self.max_w_dw,self.w_count):
            for v in np.linspace(self.min_v_dw,self.max_v_dw,self.v_count):
                if not (v == 0 and w == 0):
                    path = Path(v,w)
                    # numba.literally(path)
                    next_x, next_y, next_theta = self.predict_state(v,w,state.x,state.y,state.theta,self.dt,self.n)

                    path.x = next_x
                    path.y = next_y
                    path.theta = next_theta
                    
                    paths.append(path)
                # print("path number :" + str(len(paths)-1)+" ,linear speed :" + str(v) + " ,angular speed :" +str(w))

        # self.traj_paths.append(paths)
        # numba.literally(paths)
        return paths


    def eval_path(self,paths,goal_x,goal_y,state,goal_region,costmap):
        
        failFlag = False
        score_headings_temp = []
        score_velocities_temp = []
        score_obstacles = []
        obs_idx = []
        obs_pixel_x = []
        obs_pixel_y = []

        for path in paths:

            score_headings_temp.append(self.calc_heading(path,goal_x,goal_y,state,goal_region))
            score_velocities_temp.append(self.calc_velocity(path))
            temp_score,idx,xx,yy = self.calc_clearance(path,state,costmap)
            # score_obstacles.append(self.calc_clearance(path,state)) #iceride normalize ediliyor !!
            score_obstacles.append(temp_score)
            obs_idx.append(idx)
            obs_pixel_x.append(xx)
            obs_pixel_y.append(yy)




        #normalization
        score_headings = [h/math.pi for h in score_headings_temp]  

        score_velocities = [h/self.max_v_dw for h in score_velocities_temp]  #Buraya tekrar bak !!!

        score = 0

        for k in range(len(paths)):
            # dis_to_goal = np.sqrt((goal_x-state.x)**2 + (goal_y-state.y)**2)
            # self.heading_cost_weight = self.heading_cost_weight/dis_to_goal
            # print(self.heading_cost_weight)
            temp_score = 0

            temp_score = (self.heading_cost_weight*score_headings[k]) + (self.obstacle_cost_weight*score_obstacles[k]) + (self.velocity_cost_weight*score_velocities[k])

            if temp_score > score:
                if not self.check_path_velo(paths[k],obs_idx[k],obs_pixel_x[k],obs_pixel_y[k],paths[k].v,paths[k].w,state):
                    # print("Not admissible velocity !!!")
                    paths[k].admissibility = False
                    continue
                # if paths[k].v == 0 and paths[k].w == 0:
                #     print("= ******0 hız isteği*****")
                #     continue
                opt_path = paths[k]
                score = temp_score
                # print(str(k)+ ". path is optimal for now, score :"+str(score))
        try:
            return opt_path, failFlag
        except:
            opt_path = []
            failFlag = True
            return opt_path, failFlag
            # raise("Can not calculate optimal path!")
            # opt_path.v = 0
            # opt_path.w = 0



    def check_path_velo(self,path,idx,xx,yy,v,w,state,resolution = 0.05):

        if idx == None:
            return True
        if xx == None or yy == None:
            return True
        else:
            x_0 = self.meter2pixel(path.x[idx],state,'x')
            y_0 = self.meter2pixel(path.y[idx],state,'y')
            sum = self.distance(x_0,xx,y_0,yy)*resolution
            for i in range(idx,1,-1):
                x1 = path.x[i-1]
                x2 = path.x[i]
                y1 = path.y[i-1]
                y2 = path.y[i]
                sum = (sum + self.distance(x1,x2,y1,y2))

            cond_v =  v < np.sqrt(2*sum*self.max_dec_v)
            dist_w = abs(path.theta[idx]-state.theta)
            cond_w =  w < np.sqrt(2*dist_w*self.max_dec_w) #bu conditiona gerek yok çizgisel hız conditionı durmak için yeterli!!!

            if cond_v and cond_w: #cond_w kaldırılacak makaledeki tanım mantıklı değil !!!
                return True
            else:
                return False





    def calc_heading(self,path,goal_x,goal_y,state,goal_region):
        
        dis_to_goal = np.sqrt((goal_x-state.x)**2 + (goal_y-state.y)**2)

        last_x = path.x[0]
        last_y = path.y[0]
        last_theta = path.theta[0]

        angle_to_goal = math.atan2((goal_y-last_y),(goal_x-last_x))
        score_angle = angle_to_goal-last_theta

        cost = abs(math.atan2(math.sin(score_angle), math.cos(score_angle)))

        score_cost = math.pi - cost

        return score_cost


    def calc_velocity(self,path):

        score_velocity = path.v

        return score_velocity


    def obstacle_position(self,obstacles,state):

        obs_x = []
        obs_y = []
        orig_pix = 30
        for obs in obstacles:
            # (20,20) initial robot position, resolution = 5 cm/pixel

            obs_x_temp = state.x + 0.05*(obs.x-orig_pix)
            obs_y_temp = state.y + 0.05*(obs.y-orig_pix)
            obs_x.append(obs_x_temp)
            obs_y.append(obs_y_temp)

        return obs_x,obs_y


    def obs_pos_trial(self,obstacles):

        obs_x = []
        obs_y = []

        for obs in obstacles:
            # (20,20) initial robot position, resolution = 5 cm/pixel

            obs_x_temp = 0.05*(obs.x-self.origin_pixel)
            obs_y_temp = 0.05*(obs.y-self.origin_pixel)
            obs_x.append(obs_x_temp)
            obs_y.append(obs_y_temp)

        return obs_x,obs_y



    def pixel2meter(self,pixel): #resolution eklenecek !!

            # (20,20) initial robot position, resolution = 5 cm/pixel

        return 0.05*(pixel-self.origin_pixel)



    def calculateSlope(self,x1,x2,y1,y2):
        (px,py) = (abs(x2-x1),abs(y2-y1))
        try: #slope should between 0 - 1
            if py > px:
                slope = px/py
            else:
                slope = py/px
            if py == 0 and px == 0:
                slope = 0
        except ZeroDivisionError:
            slope = 0
        # print("slope =: {}".format(slope))
        return slope


    def distance(self,x1,x2,y1,y2):
        px = (float(x1)-float(x2))**2
        py = (float(y1)-float(y2))**2
        return (px+py)**(0.5)


    def meter2pixel(self,x,state,var,resolution=0.05):

        x = round(x,3)
        #init yerine state olmalı !!
        if var == 'x':
            init = self.init_x
            # init = state.x
        elif var == 'y':
            init = self.init_y
            # init = state.y


        # if x > init:       
        #     x_pixel = (math.ceil((x-init-(resolution/2))/resolution))+self.origin_pixel
        # elif x == init:
        #     return self.origin_pixel
        # else:
        #     x_pixel = self.origin_pixel - abs(math.floor((x-init+(resolution/2))/resolution))

        if (abs(x-init) % resolution) < (resolution/2):
            if x > init:
                x_pixel = math.floor((x-init) / resolution) + self.origin_pixel
            elif x < init:
                x_pixel = self.origin_pixel - math.floor((init-x)/resolution)
            else:
                x_pixel = self.origin_pixel
        else:
            if x > init:
                x_pixel = math.ceil((x-init) / resolution) + self.origin_pixel
            elif x < init:
                x_pixel = self.origin_pixel - math.ceil((init-x)/resolution)


        # if x_pixel<0 or not(x_pixel < len(self.costmap[0])):
        #     raise IndexError

        return x_pixel


    def calc_clearance(self,path,state,costmap):

        xx = None
        yy = None
        self.temp_cost = 0
        self.temp_obs = 0
        temp_stp = 0
        cost_temp = 0

        for a in range(len(path.x)-1):
            x1 = path.x[a]
            x2 = path.x[a+1]

            y1 = path.y[a]
            y2 = path.y[a+1]
        

            x1 = self.meter2pixel(x1,state,'x')
            x2 = self.meter2pixel(x2,state,'x')
            y1 = self.meter2pixel(y1,state,'y')
            y2 = self.meter2pixel(y2,state,'y')
            if a > 0:
                x3 = self.meter2pixel(path.x[a-1],state,'x')
                y3 = self.meter2pixel(path.y[a-1],state,'y')
            pix_ctr = max(abs(x2-x1),abs(y2-y1))
            if pix_ctr == 0:
                if a > 0:
                    if not max(abs(x2-x1),abs(y2-y1),abs(y3-y2),abs(x3-x2)) == 0:
                        pix_ctr = 1
                else:
                    pix_ctr = 1
            temp_stp = pix_ctr + temp_stp


        for a in range(len(path.x)-1):
            x1_ = path.x[a]
            x2_ = path.x[a+1]

            y1_ = path.y[a]
            y2_ = path.y[a+1]        

            x1 = self.meter2pixel(x1_,state,'x')
            x2 = self.meter2pixel(x2_,state,'x')
            y1 = self.meter2pixel(y1_,state,'y')
            y2 = self.meter2pixel(y2_,state,'y')
            slope = self.calculateSlope(x1,x2,y1,y2)
            sign_x = np.sign(x2-x1)
            sign_y = np.sign(y2-y1)
            stp =  max(abs(x2-x1),abs(y2-y1))
            
            if stp == 0:
                if 0<x1<len(costmap) and 0<y1<len(costmap):
                    if a > 0:
                        x3 = self.meter2pixel(path.x[a-1],state,'x')
                        y3 = self.meter2pixel(path.y[a-1],state,'y')
                        if max(abs(x2-x1),abs(y2-y1),abs(y3-y2),abs(x3-x2)) == 0:
                            if not (a == len(path.x)-2):
                                continue
                            else:
                                return cost_temp/temp_stp,None,x1,y1
                    if costmap[x1][y1] <0.03:
                        return cost_temp/temp_stp,a,x1,y1
                    else:
                        cost_temp = cost_temp + costmap[x1][y1]
                else:
                    cost_temp = cost_temp + 1 #yeni ekledim
            else:
                for i in range (1,stp+1):
                    if abs(x2-x1) > abs(y2-y1):
                        xx = x1 + sign_x*i
                        yy = y1 + math.floor(sign_y*i*slope)
                        if xx<len(costmap) and yy<len(costmap):
                            if costmap[xx][yy] <0.03:
                                return cost_temp/temp_stp,a,xx,yy
                            else:
                                cost_temp = cost_temp + costmap[xx][yy]
                        else: cost_temp = cost_temp + 1 #yeni ekledim
                    else:
                        yy = y1 + sign_y*i
                        xx = x1 + math.floor(sign_x*i*(slope))               
                        if 0<xx<len(costmap) and 0<yy<len(costmap):
                            if costmap[xx][yy] <0.03:
                                return cost_temp/temp_stp,a,xx,yy
                            else: 
                                cost_temp = cost_temp + costmap[xx][yy]
                        else: cost_temp = cost_temp + 1 #yeni ekledim

        cost_norm = cost_temp/temp_stp

        return cost_norm,None,xx,yy