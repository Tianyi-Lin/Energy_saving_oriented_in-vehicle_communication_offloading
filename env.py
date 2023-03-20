import math
import random
import numpy as np
import constants as cn
import scipy.stats as stats
import math
import queue
from numpy import random
from gym import spaces
from sklearn import preprocessing
import paddle
# from gym.utils import seeding

##返回的一系列状态，act_shape_n分别代表了6个agent的动作的维数，以及agent的名字，
# agent action的活动空间,以及每个agent的obs_shape_n代表的是每个agent观测到的环境的个数，obs_space暂时还没懂


class Environment(object):

    def __init__(self):
        self.num_tvehicle = 5    ##用来指定task vehicle的数量，就是agent的数量           # task vehicle的用户数
        self.num_svehicle = 20     ##service vehicle的数量，可以采取随机撒点的方式，小一点应该比较好，这样才更有可能出现竞争资源的情况
        # self.tvehicles = []   ##task vehicle进行初始化
        # self.svehicles = []   ##service vehicle进行初始化
        self.segment = 300   #代表segment的长度为300m
        self.road_length = 30000     #设置道路的长度为20000m
        self.lane_num = 2   #共有两个lane road
        self.lane_width = 4    #lane road的宽度为4m
        self.dis = 20    ###RSU距离第一个街道中心的距离
        self.v_mu = 50   ###汽车速度的均值，单位是km/h
        self.v_sigma = 29.2     ##汽车速度的方差
        self.lam = 0.8   ##泊松到达的参数
        self.p_task = 0.2    ##车辆生成task的概率
        self.fai = 1000    ##每bit执行需要的CPU cycle
        self.Z = [1,2]    ##用来代表不同的task的重要性   3代表cruial task, 2代表high priority task, 1代表low priority task
        # self.Thr = [0.2,0.1]    #delay threshhold
        self.lower = self.v_mu - 0.25 * self.v_sigma  # 车辆速度为截断高斯分布
        self.upper = self.v_mu + 0.25 * self.v_sigma
        self.p_thresh = [-0.1,-0.2]    ##任务失败之后的惩罚
        self.R_h = 10      #RSU的高度
        self.v_max = self.upper      ####先假定最大速度就是截断上界，应该是合理的
        self.speed = stats.truncnorm((self.lower - self.v_mu) / self.v_sigma, (self.upper - self.v_mu) / self.v_sigma, loc=self.v_mu, scale=self.v_sigma)
        self.num_RSU = self.road_length//self.segment   #RSU的数量
        # self.RSU = []
        self.max_RSUf = 15*cn.GHZ
        self.max_vehif = 1.5 *cn.GHZ
        self.max_svehif = 10 * cn.GHZ
        # self.init_RSUs()
        self.buffer_max = 5     #假定每个汽车的buffer中最多能存储5个task
        self.time_slot = cn.slot
        self.energy_coff = math.exp(-28)
        self.power = 1
        # self.seed()
        self.pss = 2
        self.unit_r = 0.4 * cn.GHZ     #RSU的unit computing resource
        self.unit_s = 0.2 * cn.GHZ    #service vehicle的unit computing resource
        self.v2v_r = 100    #汽车能够建立连接的最大距离，可能这个参数也要调
        obs_shape = (self.num_svehicle)*6 + 1*5 +3+4        #状态信息包括所有service vehicle的速度，位置，方向，信道增益，可用计算资源以及
        # RSU的index，位置，信道增益，可用计算资源以及处理的task的三个信息，buffer以及自身的速度
        self.act_shape_n = [2 for _ in range(self.num_tvehicle)]    ##动作一共有2个,server index, offloading ratio
        self.obs_shape_n = [obs_shape for _ in range(self.num_tvehicle)]
        # self.d_r = 0
        # self.dis_ts = []
        self.action_space = []
        self.observation_space = []
        self.B_R = 20
        self.B_st =5
        for i in range(self.num_tvehicle):
            # action = spaces.Dict({
            #     "sever": spaces.Discrete(2),
            #     "offloding ratio": spaces.Box(0, 1, shape=(1,), dtype='float32'),
            #     "computing resource": spaces.Box(0, 1, shape=(1,), dtype='float32')})
            action = spaces.Dict({
                "sever": spaces.Discrete(2),
                "offloding ratio": spaces.Box(0, 1, shape=(1,), dtype='float32')})
            self.action_space.append(action)
            self.observation_space.append(spaces.Box(low=float("-inf"),high=float('inf') ,shape=(obs_shape,), dtype='float32'))
        ##目前没有对observation space 进行限制，因为感觉没有必要，后续可以再考虑考虑
        # self.init_tvehicles()      ####也就是初始化agents
        # self.init_svehicles()       ##也就是初始化service vehicle

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    #####初始化RSU的位置以及其id和其计算资源
    '''万一上一时隙的任务还没完成，就卸载到了RSU，这怎么办，后续可以关注一下，或者在这里加buffer,做优化'''
    def init_RSUs(self):
        for i in range(self.num_RSU):
            R_id = i+1
            R_x = self.segment/2 + self.segment*i
            R_y = 0
            R_h = 10                         ##RSU的高度为10m
            R_pos = [R_x,R_y]
            f = self.max_RSUf     ###RSU的计算频率，单位为GHZ
            r_ratio = np.zeros(self.num_tvehicle)
            share = False
            self.RSU.append({"id":R_id, "position":R_pos, "frequency":f,"ratio":r_ratio,"share":share})

    ##初始化task vehicle的信息，包括其初始位置,以及其他的信息等等
    def init_tvehicles(self):
        for i in range(self.num_tvehicle):
            buffer = []
            tv_x = np.random.randint(0, self.segment)     ###假设刚开始时，task vehicle都在第一个segment
            tv_y = self.dis+self.lane_width*np.random.randint(0, 2)
            tv_pos = [tv_x, tv_y]
            tv_direction = 1    ##代表是向右行驶的，-1代表向左行驶  np.random.choice([-1,1])
            t_velocity = int(self.speed.rvs(1))  ##这就是汽车行驶的速度，假设汽车一直保持匀速行驶
            tv_id = i+1
            tv_f = self.max_vehif   ###vehicle的计算频率，单位为GHZ
            r_index = tv_x//self.segment + 1   # 用来存放刚开始的时候连接的RSU的index
            o_rsu = self.RSU[r_index-1]   ##目的就是得到所处地方的RSU的状态信息
            tv_task = self.generate_tasks(t_velocity)
            buffer.append(tv_task)
            self.dis_ts = np.zeros(self.num_svehicle+1)   #第一个位置存放的是车辆到RSU的距离,其他存放的是到service vehicle的距离
            self.ch_gain = np.zeros(self.num_svehicle+1)   #第一个位置存放的是车辆到RSU的信道增益
            self.rate = np.zeros(self.num_svehicle+1)     #同理
            '''这里还需要限制buffer的最大数量'''

            self.tvehicles.append({"id": tv_id, "position": tv_pos,
                                   "direction": tv_direction,"speed":t_velocity,"frequency":tv_f,"RSU_index":r_index,
                                   "RSU":o_rsu,
                                   "buffer": buffer,
                                   "distance":self.dis_ts,
                                   "channel_gain":self.ch_gain,
                                   "trans_rate":self.rate})

    ##初始化service vehicle的信息，包括其初始位置,以及其他的信息等等
    def init_svehicles(self):
        for i in range(self.num_svehicle):
            sv_direction = np.random.choice([-1,1])  ##1代表是向右行驶的，-1代表向左行驶  np.random.choice([-1,1])
            if sv_direction == -1:      #如果汽车向左行驶，那就在最后一个segment生成
                sv_x = np.random.randint(self.road_length-self.segment,self.road_length)
                sv_y = self.dis + self.lane_width * np.random.randint(0, 2)
                sv_pos = [sv_x, sv_y]
            else:
                sv_x = np.random.randint(0,self.segment)
                sv_y = self.dis + self.lane_width * np.random.randint(0, 2)
                sv_pos = [sv_x, sv_y]
            s_velocity = int(self.speed.rvs(1))  ##这就是汽车行驶的速度，假设汽车一直保持匀速行驶
            sv_id = i + 1
            sv_f = self.max_svehif  ###vehicle的计算频率，单位为GHZ
            buffer = []
            s_ratio = np.zeros(self.num_tvehicle)
            share = False   ##代表此sever是否被公用
            self.svehicles.append({"id": sv_id, "position": sv_pos,
                                   "direction": sv_direction, "speed": s_velocity, "frequency": sv_f,
                                   "buffer": buffer,
                                   "ratio":s_ratio,
                                   "share":share})

    def generate_tasks(self,velocity):
        # task = []
        data_size =1/30*np.random.poisson(lam=3, size=None) * cn.MB
        priority = random.choice(self.Z)
        if priority == 1:      ####速率最大值v_max和sigma的取值要注意
            tao = np.random.choice([0.6,0.7,0.8])
       ### "这里的公式有可能需要改一下，不一定遵循这样的一个分布"
        # elif priority == 2:
        #     tao = self.Thr[1] * math.exp((self.v_max ** 2 - velocity ** 2) / (2 * self.v_sigma ** 2))
        else:
            tao = 0.2 * math.exp((self.v_max ** 2 - velocity ** 2) / (2 * self.v_sigma ** 2))
        tao = float(format(tao, ".1f"))    #保留一位小数
        task = [data_size,priority,tao]
        return task   ##返回的是数组只有一个元素


    def reset(self):            ###reset函数应该还得返回obs和action_dim吧
        self.tvehicles = []  ##task vehicle进行初始化
        self.svehicles = []
        self.RSU = []
        self.init_RSUs()
        self.step_count = 0
        self.init_tvehicles()
        self.init_svehicles()
        self.update_tvehicleState()
        self.done = False
        self.obs_normalization, self.obs = self.get_obs()   ##返回当前的状态信息
        return self.obs_normalization,self.obs     ##这样归一化之后有很多点的值都为0 ，这样会不会有问题

    ##这是n个agent都应该有的，每个agent的观测值都应该更新，并且obs应该更新
    ##可以先更新每个agent的观测值，再得到next_obs,所以根据每个agent的状态得到obs可以写个函数，利用reset中的语句
    def step(self, action_n):   ###我觉得这里的重点就是归一化后的action和observation ,应该怎么对他们操作
        "先得到采取这个动作后的action的reward"
        reward_n = np.zeros(self.num_tvehicle)   ##存放每个agent的reward值
        local_time = np.zeros(self.num_tvehicle)
        local_energy = np.zeros(self.num_tvehicle)
        off_time = np.zeros(self.num_tvehicle)
        total_time = np.zeros(self.num_tvehicle)
        task_done = [False for i in range(self.num_tvehicle)]   ##说明这个episode还没结束
        up_t = np.zeros(self.num_tvehicle)         ##存放offloading task的执行时间
        trans_time = np.zeros(self.num_tvehicle)
        severcom_time = np.zeros(self.num_tvehicle)
        g = np.zeros(self.num_tvehicle)    #用来记录这个task当前是否fail，fail则为1
        now_pos,rsu_index = self.save_vehicles_pos()    ##保存当前所有vehicle的位置,首先是task vehicle,再其次是service vehicle
        self.max_slot = 15        ##最大为15个时隙，如果还没执行完就放弃,而且测试了几乎所有的时延限制都小于此
        actions_n = []     ###存放真实的action
        severs = np.zeros(self.num_tvehicle)   ##存放每个task vehicle选择的sever
        delay_n = np.zeros(self.num_tvehicle)
        ave_delay = 0
        succe = 0  ###记录本次成功执行的task的个数
        for i in range(len(self.RSU)):
            self.RSU[i]["share"] = False
        for i in range(self.num_svehicle):
            self.svehicles[i]["share"] = False
        for i in range(self.num_tvehicle):
            self.tvehicles[i]["RSU"]["share"] = False
        ###先得到每个agent的真实的action值,0代表了offloading ratio,越大代表卸载的越多，1代表了卸载的server的index,在这里
        #在这里只改变了各个task vehicle处的计算资源（env中的）
        for i in range(self.num_tvehicle):
            actions_n.append((action_n[i] + 1) / 2)    #将-1~1的action映射到0~1的actions

            actions_n[i][1]=int(actions_n[i][1]*(self.num_svehicle+1))        ##server选择0~30
            if (actions_n[i][1]>self.num_svehicle):
                actions_n[i][1]= self.num_svehicle
            ###下面得到的是在本地执行需要的时间和能耗
            local_time[i]=cn.fai * (1 - actions_n[i][0]) * self.process_tasks[i][0][0]/ self.tvehicles[i]["frequency"]
            # local_energy[i]=self.energy_coff * (1 - actions_n[i][1]) * self.process_tasks[i][0][0] * cn.fai * \
            #                                             (actions_n[i][0])**2
            # self.tvehicles[i]["frequency"] = self.tvehicles[i]["frequency"]-actions_n[i][0]   #更新本地的计算频率
            ##如果刚开始选择的sever就不在覆盖范围内，则任务失败，根据不同种类数据包的重要性收到负的reward
            sever = int(actions_n[i][1])
            severs[i] = sever
            if (sever > self.num_svehicle):
                sever = self.num_svehicle    ##因为action有可能等于1，所以sever有可能等于31，所以设置一个最大值
            if ((actions_n[i][1]!=0) and (self.tvehicles[i]["distance"][sever]>self.v2v_r)):
                prior = self.process_tasks[i][0][1]
                reward_n[i] = self.p_thresh[prior-1]
                g[i] = 1
            ###下面重点处理的是卸载需要的时间和能耗，因为channal gain在每个时隙都在不停的变化
        # print(r_sever)

        ###接下来就是要判断是否有共享同一个sever的情况，在sever = 0，即卸载到RSU的时候，情况比较复杂
        records = severs   ##存放每个task vehicle选择的sever
        vals, inverse, count = np.unique(records, return_inverse=True,
                                         return_counts=True)

        idx_vals_repeated = np.where(count > 1)[0]
        vals_repeated = vals[idx_vals_repeated]

        rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
        _, inverse_rows = np.unique(rows, return_index=True)
        res = np.split(cols, inverse_rows[1:])  ###这里得到的是sever相同的task vehicle的下标
        if (len(res) == 1 and len(res[0]) == 0):
            num1 = 0      ########这个num1指的是什么呢，指的应该是有相同sever的组别的数量？
        else:
            num1 = len(res)
        for k in range(num1):
            index = res[k]      ##这里记录的是几组sever相同的task vehicle的下标
            sever = int(records[index][0])   ##这里记录的是对应的sever,从0-30
            dis = []
            prior = []
            task_size = []
            load_task = []
            for i in range(len(index)):
                v = index[i]    ##task vehicle的下标
                dis.append(self.tvehicles[v]["distance"][sever])  #这里的distance存放的也是从0-30
                prior.append(self.tvehicles[v]["buffer"][0][1])   #所以service vehicle中的buffer得应该定时清空？
                task_size.append(self.tvehicles[v]["buffer"][0][0])    ###这里得到的是task-size
                load_task.append(self.tvehicles[v]["buffer"][0][1]*self.tvehicles[v]["buffer"][0][1]*actions_n[v][0])
            # sum1 = sum(dis)
            # sum2 = sum(prior)
            sum3 = sum(load_task)
            # dis_n = [0.5 * j / sum1 for j in dis]
            # prior_n = [0.5 * j / sum2 for j in prior]
            # weight = [i + j for i, j in zip(dis_n, prior_n)]    ##这里得到的是相同sever的task vehicle所对应的权重
            weight = [j / sum3 for j in load_task]    ####这里采用的是新的权重的表达式，参考的另一篇文献
            if (sever==0):   ##虽然卸载的sever都是RSU，但是要判断是否是同一个RSU
                r = []    #存放sever为0的task vehicle所连接的RSU_index,看是否需要共享计算资源
                for i in range(len(index)):
                    v = index[i]  ##task vehicle的下标
                    r.append(self.tvehicles[v]["RSU_index"]-1)
                r = np.array(r)
                vals, inverse, count = np.unique(r, return_inverse=True,
                                                 return_counts=True)
                idx_vals_repeated = np.where(count > 1)[0]
                vals_repeated = vals[idx_vals_repeated]
                rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
                _, inverse_rows = np.unique(rows, return_index=True)
                r_res = np.split(cols, inverse_rows[1:])
                if (len(r_res) == 1 and len(r_res[0]) == 0):
                    num2 = 0
                else:
                    num2 = len(r_res)
                for m in range(num2):
                    r_index = r_res[m]  ##这里记录的是在同一个segment下的task vehicle的下标
                    rsu = r[r_index][0]  ##这里记录的是对应的rsu的下标
                    self.RSU[rsu]["share"] = True
                    weight1 = np.array(weight)
                    rsu_weight = weight1[r_index]
                    new_w = [j / sum(rsu_weight) for j in rsu_weight]  ##重新归一化之后的有关于rsu的weight
                    for n in range(len(r_index)):
                        ins = r_index[n]
                        v = index[ins]   ##task vehicle的下标
                        self.RSU[rsu]["ratio"][v] = new_w[n]  ####感觉这里得设置一个变量，说明这个step中sever被公用了
            else:   ##卸载到同一辆service vehicle的情况，这里还是比较好判断的
                self.svehicles[sever-1]["share"] = True        ####感觉这里得设置一个变量，说明这个step中sever被公用了
                for i in range(len(index)):
                    v = index[i]    ##task vehicle的下标
                    self.svehicles[sever-1]["ratio"][v] = weight[i]


        ##在sever处的计算时间,unit是1s
        for i in range(self.num_tvehicle):
            sever = int(actions_n[i][1])
            if (sever > self.num_svehicle):
                sever = self.num_svehicle  ##因为action有可能等于1，所以sever有可能等于31，所以设置一个最大值
            if (sever == 0):  ##如果选择的sever是RSU
                if (self.tvehicles[i]["RSU"]["share"] == False):
                    severcom_time[i] = cn.fai * actions_n[i][0] * \
                                        self.process_tasks[i][0][0] / self.tvehicles[i]["RSU"]["frequency"]
                else:
                    severcom_time[i] = cn.fai * actions_n[i][0] * self.process_tasks[i][0][0] / (
                                self.tvehicles[i]["RSU"]["frequency"] * self.tvehicles[i]["RSU"]["ratio"][i])
            else:
                if (self.svehicles[sever-1]["share"] == False):
                    severcom_time[i] = cn.fai * actions_n[i][0] * \
                                     self.process_tasks[i][0][0] / self.svehicles[sever-1]["frequency"]
                else:
                    severcom_time[i] = cn.fai * actions_n[i][0] * self.process_tasks[i][0][0] / (
                                self.svehicles[sever - 1]["frequency"] * self.svehicles[sever - 1]["ratio"][i])

        ##这个代表的是在sever处执行的数据量的大小
        remain_offtasksize = [action_n[i][0]*self.process_tasks[i][0][0] for i in range(self.num_tvehicle)]

        ##得到传输到sever处的传输时间，并判断任务传输时间是否超过最大时延导致任务失败，unit是0.1s
        ##在这里每个时隙，更新了所有vehicle的位置，task vehicle所能观测到的环境信息
        for i in range(self.max_slot):
            for j in range(self.num_tvehicle):
                # ###如果驶出这条道路，任务就完成了
                # if (self.tvehicles[j]["position"][0] > self.road_length):
                #     task_done[j] = True
                if (remain_offtasksize[j] > 0 and g[j]==0):  ##当这个task没有fail,才考虑传输
                    self.move_vehicles()  ##用来更新所有vehicle的位置
                    self.update_tvehicleState()  ###用来更新task vehicle的各种信息
                    # self.update_SeverComState()  ####主要是更新sever处的计算资源
                    up_t[j] = up_t[j] + cn.slot
                    remain_offtasksize[j] = remain_offtasksize[j]-self.tvehicles[j]["trans_rate"][int(actions_n[j][1])]*\
                                            cn.slot
                    if remain_offtasksize[j] <= 0 and i == (self.max_slot-1):
                        trans_time[j] = up_t[j]
                else:     ###感觉针对task fail的情况也成立，而且task fail之后我也不考虑传输时间了
                    if (-remain_offtasksize[j]>0.5*self.tvehicles[j]["trans_rate"][int(actions_n[j][1])]*cn.slot):
                        trans_time[j] = up_t[j]-cn.slot    ###这里采用的是四舍五入的办法，如果负的量很多，就假定上一个时隙就能处理完
                    else:
                        trans_time[j] = up_t[j]    #保存传输时间
                    # 如果传输时间大于时延限制，则任务失败，根据不同种类数据包的重要性收到负的reward
                    if (up_t[j]>=self.process_tasks[j][0][2]):
                        prior = self.process_tasks[j][0][1]
                        reward_n[j] = self.p_thresh[prior-1]
                        g[j] = 1
                    self.move_vehicles()  ##用来更新所有vehicle的位置
                    self.update_tvehicleState()  ###用来更新task vehicle的各种信息，离其他sever位置以及传输速率的变化
                    # self.update_SeverComState()  ####主要是更新service vehicle的计算资源

        ####判断在sever处的任务执行完之后，server是否还在传输范围之内，不在的话则任务失败，否则任务成功,但是要比较是否小于时延
        ##没有在env中改变值
        for i in range(self.num_tvehicle):
            if(g[i]==0):   ##也就是这个task没有fail
                off_time[i] = severcom_time[i] + trans_time[i]
                move_pos = self.move_vehicles_pos(off_time[i], now_pos)  ##前面是task vehicle,后面是service vehicle的位置
                t_vehicle_pos = move_pos[i]     ##这个是task vehicle的位置
                ###接下来的循环语句是判断执行结束后是否不在sever的范围内
                if(actions_n[i][1]!=0):    #sever是service vehicle
                    s_vehicle_pos = move_pos[self.num_tvehicle+int(actions_n[i][1])-1]   #service vehicle的位置
                    d = [s_vehicle_pos[j] - t_vehicle_pos[j] for j in range(len(t_vehicle_pos))]
                    d0 = [d[k] * d[k] for k in range(len(d))]
                    dis = math.sqrt(np.sum(d0))
                    if (dis > self.v2v_r):
                        prior = self.process_tasks[i][0][1]
                        reward_n[i] = self.p_thresh[prior-1]
                        g[i] = 1
                else:   ###如果出RSU的界，那么任务就失败，收到相应的惩罚
                    if(t_vehicle_pos[0]>self.segment*(rsu_index[i]+1)):
                        prior = self.process_tasks[i][0][1]
                        reward_n[i] = self.p_thresh[prior-1]
                        g[i] = 1
                ###如果在传输范围内，则进行最后的判断
                if(g[i]==0):
                    total_time[i]=max(local_time[i],off_time[i])   ##得到这个task总的执行时间
                    prior = self.process_tasks[i][0][1]
                    if(total_time[i]<=self.process_tasks[i][0][2]):  #如果执行时间小于时延限制
                        th = self.process_tasks[i][0][2]
                        reward_n[i] = th - total_time[i]
                        succe = succe + 1
                        delay_n[i] = total_time[i]
                    else:
                        reward_n[i] = self.p_thresh[prior - 1]
        if (succe != 0):
            ave_delay = sum(delay_n) / succe  ##所有agent的平均时延
        else:
            ave_delay = 10000
        ##得到了当前task的reward，就要更新状态信息，所有vehicle的位置已经得到了更新，在前前一个循环里，所以接下来
        #所以接下来要把每个agent的buffer清空，生成新的task到buffer里面，再返回得到新的observation
        self.update_SeverComState()
        for i in range(self.num_tvehicle):
            self.tvehicles[i]["buffer"].pop()    ##移除当前的task
            task= self.generate_tasks(self.tvehicles[i]["speed"])
            self.tvehicles[i]["buffer"].append(task)    ##将新的task移入buffer
        for i in range(self.num_svehicle):
            self.svehicles[i]["ratio"] = np.zeros(self.num_tvehicle)
            self.svehicles[i]["share"] = False
        for i in range(self.num_RSU):
            self.RSU[i]["ratio"] = np.zeros(self.num_tvehicle)
            self.RSU[i]["share"] = False
        self.obs_normalization, self.obs = self.get_obs()


        return self.obs_normalization,reward_n, task_done,g,ave_delay
        ###最后得到的trans_time中如果某一个为0，则说明这个agent的task没有执行完任务，那肯定早已经收到reward了



    def move_vehicles(self):  ####这一部分衡量的是vehicle的移动性，假设车辆只在一个lane移动
        for i in range(self.num_tvehicle):
            self.tvehicles[i]["position"][0]=self.tvehicles[i]["position"][0]\
                                             +self.tvehicles[i]["direction"]*self.tvehicles[i]["speed"]/3.6*cn.slot
        for i in range(self.num_svehicle):
            self.svehicles[i]["position"][0] = self.svehicles[i]["position"][0] \
                                               +self.svehicles[i]["direction"]*self.svehicles[i]["speed"]/3.6*cn.slot

    def save_vehicles_pos(self):    ##保存当前所有vehicle的位置
        now_pos = []
        rsu_index = []
        for i in range(self.num_tvehicle):
            now_pos.append([self.tvehicles[i]["position"][0],self.tvehicles[i]["position"][0]])
            rsu_index.append(self.tvehicles[i]["RSU_index"])   ##index是从1开始的
        for i in range(self.num_svehicle):
            now_pos.append([self.svehicles[i]["position"][0], self.svehicles[i]["position"][0]])
        return now_pos,rsu_index

    def move_vehicles_pos(self,o_time,now_pos):   ##得到在server处执行task以后vehicle的位置，并没有改变env中的值
        now_pos = now_pos
        move_pos = []
        for i in range(self.num_tvehicle):
            pos = now_pos[i]   ##以前车辆的位置
            move_pos.append([pos[0]+self.tvehicles[i]["direction"]*self.tvehicles[i]["speed"]/3.6*o_time,pos[1]])
        for i in range(self.num_svehicle):
            pos = now_pos[i+self.num_tvehicle]  ##以前车辆的位置
            move_pos.append(
                [pos[0] + self.svehicles[i]["direction"] * self.svehicles[i]["speed"] / 3.6 * o_time, pos[1]])
        return move_pos

    ##更新了task vehicle所能观测到的所有信息，都是在env里面改变的
    def update_tvehicleState(self):
        for i in range(self.num_tvehicle):    ###如果我不做排队的话，那感觉直接将本地的计算资源利用完不就可以了吗？所以这个优化变量是不是就可以忽略了
            r_index = int(self.tvehicles[i]["position"][0] // self.segment + 1)  # 用来存放刚现在连接的RSU的index
            # print(r_index)
            self.tvehicles[i]["RSU_index"]=r_index
            self.tvehicles[i]["RSU"]=self.RSU[r_index-1]
            ray_factor = random.rayleigh(scale=math.sqrt(2) / 2,
                                         size=(1, int(self.num_svehicle) + 1))  # 瑞丽衰落的channal fading
            pos_self = self.tvehicles[i]["position"]
            pos_r = self.tvehicles[i]["RSU"]["position"]
            d = [pos_r[k] - pos_self[k] for k in range(len(pos_r))]
            d0 = [d[k] * d[k] for k in range(len(d))]
            self.tvehicles[i]["distance"][0] = math.sqrt(np.sum(d0))  # 存放到达RSU的距离,要不要考虑加上RSU的高度呢?
            for j in range(self.num_svehicle):
                pos_s = self.svehicles[j]["position"]
                d = [pos_s[k] - pos_self[k] for k in range(len(pos_self))]
                d0 = [d[k] * d[k] for k in range(len(d))]
                if (math.sqrt(np.sum(d0)) < 1):
                    self.tvehicles[i]["distance"][j + 1] = 1.0  # 车与车之间的最短距离为1m
                else:
                    self.tvehicles[i]["distance"][j + 1] = math.sqrt(np.sum(d0))  # 存放到达所有service vehicle的距离
            self.tvehicles[i]["channel_gain"] = ray_factor[0] * 1 / (
                        pow(10, 9.81) * pow(10, -13) * pow(self.tvehicles[i]["distance"], 3)) #ray_factor[0]是一个一维list
            self.tvehicles[i]["trans_rate"][0] = self.B_R*cn.MHZ * np.log2(1 + self.tvehicles[i]["channel_gain"][0])
            for j in range(self.num_svehicle):
                self.tvehicles[i]["trans_rate"][j+1] = self.B_st * cn.MHZ * np.log2(
                    1 + self.tvehicles[i]["channel_gain"][j+1])

    #####RSU的变化幅度和service vehicle的幅度变化应该是不一样的，可以都变化
    def update_SeverComState(self):
        for i in range(self.num_RSU):
            self.RSU[i]["frequency"]=self.max_RSUf-np.random.poisson(self.pss, 1)*self.unit_r
        for i in range(self.num_svehicle):
            self.svehicles[i]["frequency"]=self.max_svehif -np.random.poisson(self.pss, 1)*self.unit_s


    def get_obs(self):   ###得到下一次的状态
        self.obs = []
        self.process_tasks = []   ###存储task vehicle的task的信息
        for i in range(self.num_tvehicle):
            process_task = self.tvehicles[i]["buffer"]
            self.process_tasks.append(process_task)
            self.state = []
            speed = []
            direction = []
            frequency = []
            pos_xy = []
            for j in range(self.num_svehicle):
                speed.append(self.svehicles[j]["speed"])
                direction.append(self.svehicles[j]["direction"])
                frequency.append(self.svehicles[j]["frequency"])
                pos_xy.append(self.svehicles[j]["position"][0])
                pos_xy.append(self.svehicles[j]["position"][1])
            chan_gain = self.tvehicles[i]["channel_gain"].tolist()
            rsu = []
            rsu.append(self.tvehicles[i]["RSU"]["id"])
            rsu.append(self.tvehicles[i]["RSU"]["position"][0])
            rsu.append(self.tvehicles[i]["RSU"]["position"][1])
            rsu.append(self.tvehicles[i]["channel_gain"][0])
            rsu.append(self.tvehicles[i]["RSU"]["frequency"])
            task_info  = process_task[0]
            own_info = []
            own_info.append(self.tvehicles[i]["speed"])
            own_info.append(self.tvehicles[i]["position"][0])
            own_info.append(self.tvehicles[i]["position"][1])
            own_info.append(self.tvehicles[i]["frequency"])
            # own_info.append(len(self.tvehicles[i]["buffer"]))
            s = speed+direction+frequency+pos_xy+chan_gain[1:len(chan_gain)]+rsu+task_info+own_info
            self.state = np.array(s, dtype=object)
            # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
            self.obs.append(self.state)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 归一化函数
        self.obs_normalization = min_max_scaler.fit_transform(self.obs)
        return self.obs_normalization, self.obs




env = Environment()
obs_normalization,obs = env.reset()
a =3



