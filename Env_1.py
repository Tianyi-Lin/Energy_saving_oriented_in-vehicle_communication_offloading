import numpy as np
import matplotlib.pyplot as plt
import math


# 蜂窝信道类 包含 车机到BS 和 DC-UE到BS 两种类型的信道
class Cellular_Channel:
    def __init__(self):
        self.h_bs = 25      # BS antenna height 基站高度
        self.h_DC_UE = 1    # DC-UE高度
        self.h_VAP = 1.5    # VAP高度
        self.Decorrelation_distance = 50  # 去相关距离 50m
        self.n_BS = 5       # 基站个数
        self.a = 166.66     # 3GPP TR 36.885中基站蜂窝小区的六边形边长
        # 基站位置 地图大小1299m * 750m
        self.BS_positions = [(self.a, 0.5 * math.sqrt(3) * self.a), (self.a, 3.5 * math.sqrt(3) * self.a),
                             (250 + self.a, 2 * math.sqrt(3) * self.a), (500 + self.a, 0.5 * math.sqrt(3) * self.a),
                             (500 + self.a, 3.5 * math.sqrt(3) * self.a)]
        self.shadow_std = 8  # Shadowing standard deviation 阴影衰落标准差
        self.R = np.sqrt(0.5 * np.ones([self.n_BS, self.n_BS]) + 0.5 * np.identity(self.n_BS))      # R矩阵
        self.N = np.random.normal(0, self.shadow_std, size=self.n_BS)                               # 随机向量N，正态分布

    # 计算某一个设备(DC-UE or VAP)到所有基站的路径损耗
    # 该设备可以使用的基站RB个数 n_available_RB = n_DCUE2BS_RB or n_VAP2BS_RB; 用于在pathloss扩展维度时使用
    # cellular_device_type = 'DC-UE' or 'VAP'
    def get_path_loss(self, veh_position, cellular_device_type, n_available_RB):
        # 暂存某设备到所有基站的路径损耗的列表
        path_loss = []
        # 轮询基站
        for BS_position in self.BS_positions:
            dx = abs(veh_position[0] - BS_position[0])
            dy = abs(veh_position[1] - BS_position[1])
            horizontal_distance = math.hypot(dx, dy)  # 输出给定坐标的欧几里得范数，水平距离
            # DC-UE和VAP的天线高度不同
            if cellular_device_type == 'DC-UE':
                vertical_distance = abs(self.h_bs - self.h_DC_UE)   # 垂直高度差
            elif cellular_device_type == 'VAP':
                vertical_distance = abs(self.h_bs - self.h_VAP)     # 垂直高度差
            else:
                print('Error: Unsupported device type!')
            d = math.sqrt(horizontal_distance ** 2 + vertical_distance ** 2) / 1000     # 两天线间距离，单位km
            path_loss.append(128.1 + 37.6 * np.log10(d))
        # path_loss的list大小(n_BS,1) 需要将其扩展维度,使其大小为(n_BS, n_available_RB)
        # 也即同一个基站上 该设备可使用的所有RB 共享相同的路径损耗
        path_loss = np.repeat(np.array(path_loss), n_available_RB, axis=0).reshape((self.n_BS, n_available_RB))
        return path_loss

    # 计算某一个设备(DC-UE or VAP)到所有基站的阴影衰落
    # 该设备可以使用的基站RB个数 n_available_RB = n_DCUE2BS_RB or n_VAP2BS_RB;
    # 输入参数：
    # (1)该设备的距离更新矩阵Dev_delta_distance 大小为(n_BS, n_BS)的对角阵;
    # (2)该设备的上一次的阴影衰落矩阵Dev_last_shadowing 大小为(n_BS, n_available_RB) 每一行的值相同，同一个基站的RB共享相同阴影衰落
    def get_shadowing(self, Dev_delta_distance, Dev_last_shadowing, n_available_RB):
        """
        delta_distance: the update distance matrix for the ith UE
        where Di(k, k) is change in distance of the ith UE to the kth eNB site from time n-1 to time n.
        Note that Di is a diagonal matrix.(对角阵)
        此处由5个BS，所以Dev_delta_distance为5*5的对角方阵，对角上的每一个元素表示Dev到5个基站距离的变化量，单位m

        Decorrelation distance = 50m

        R is a MxM matrix to generate shadowing correlation between eNB sites.(基站之间的衰落相关) M个基站
        Shadowing correlation factor = 0.5 denotes the shadowing between eNB sites
        Shadowing correlation factor = 1 denotes the shadowing between sectors of the same eNB site
        R矩阵还需要对所有元素开方

        Initial(at time 0)
        S_{eNB2UE,i}(0) = R * Ni(0)
        Ni(0)是[M*1]大小的正态分布随机向量，M为基站个数
        """
        # N随机向量, 均值0，方差8，长度为n_BS
        # 此处的N向量应该服从正态分布
        self.N = np.random.normal(0, self.shadow_std, self.n_BS)
        # ------------------- 3GPP 定义中的问题 -----------------------
        '''
        此处的Dev_last_shadowing在输入时大小为(n_BS, n_available_RB); n_available_RB = n_DCUE2BS_RB or n_VAP2BS_RB
        3GPP给出的阴影衰落迭代计算公式
        S_{eNB2UE,i}(n) = exp(-Di/D_corr) .* S_{eNB2UE,i}(n-1) + sqrt{(1-exp(-2*Di/D_corr))} .* (R * Ni (n))
        Di是大小为(n_BS, n_BS)对角阵 -> exp(-Di/D_corr)大小为(n_BS, n_BS)
        但是  S_{eNB2UE,i}(0) = R * Ni(0)     R矩阵大小(n_BS, n_BS)  Ni(0)是(n_BS*1)大小的正态分布随机向量
        所以S_{eNB2UE,i}(n-1)大小为(n_BS, 1)
        exp(-Di/D_corr) .* S_{eNB2UE,i}(n-1)    (n_BS, n_BS) .* (n_BS, 1) = (n_BS, n_BS)是允许点乘的
        但S_{eNB2UE,i}(n)大小就是(n_BS, n_BS)，与S_{eNB2UE,i}(n-1)不一致
        
        参考其他代码实现，Di应该由(n_BS, n_BS)对角阵化为(n_BS, 1)的向量
        此外，在3GPP的算法中Dev_last_shadowing大小需要为(n_BS, 1)，而此处大小为(n_BS, n_available_RB)
        故Dev_last_shadowing需要取出第一列进行降维（考虑到各列之间值相同，一个基站的RB之间共享shadowing）
        '''
        # Dev_delta_distance由对角阵转化为向量
        Dev_delta_distance = np.dot(Dev_delta_distance, np.ones(np.shape(Dev_delta_distance)[0]))
        # Dev_last_shadowing取出第一列
        Dev_last_shadowing = Dev_last_shadowing[:, 1]
        shadowing = np.multiply(np.exp(-1 * (Dev_delta_distance / self.Decorrelation_distance)), Dev_last_shadowing) + np.multiply(np.sqrt(1 - np.exp(-2 * (Dev_delta_distance / self.Decorrelation_distance))), np.dot(self.R, self.N))
        # 在返回之前 shadowing 需要升维，基站的所有RB之间共向阴影衰落值
        shadowing = np.repeat(shadowing, n_available_RB, axis=0).reshape((self.n_BS, n_available_RB))
        return shadowing


# 车辆类 描述一辆车的三个参数
class Vehicle:
    # 三个参数起始位置 起始方向 速度
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity


class Environ:
    # 环境初始化参数n_veh, n_DC_UE, n_WU
    def __init__(self, n_veh, n_DC_UE, n_WU):
        # 蜂窝信道类进行实例化
        self.Cellular_Channel = Cellular_Channel()

        # ---------- 设备数量相关信息 ----------
        self.n_veh = n_veh          # n_veh 车辆数目
        self.n_DC_UE = n_DC_UE      # 一辆车内的DC-UE数量
        self.n_WU = n_WU            # 一辆车内的WU数量
        self.n_VAP = n_veh          # VAP数量 = n_veh，一辆车配备一个VAP

        # ---------- VAP相关信息 ----------
        self.n_VAP_RB = 3 + 24          # 一个VAP可以提供的RB数量 2.4GHz 3个RB; 5GHz 24个RB
        self.VAP_RB_BW = 20e6           # VAP提供的RB带宽 20MHz
        self.h_VAP = 1.5                # 车机蜂窝天线高度
        # VAP天线增益 & 噪声
        self.vehAntGain = 3
        self.vehNoiseFigure = 9

        # ---------- 基站相关信息 ----------
        self.n_BS = 5       # 基站数量
        self.a = 166.66     # 3GPP TR 36.885中基站蜂窝小区的六边形边长
        self.BS_positions = [(self.a, 0.5 * math.sqrt(3) * self.a), (self.a, 3.5 * math.sqrt(3) * self.a),
                             (250 + self.a, 2 * math.sqrt(3) * self.a), (500 + self.a, 0.5 * math.sqrt(3) * self.a),
                             (500 + self.a, 3.5 * math.sqrt(3) * self.a)]       # 基站位置
        self.n_VAP2BS_RB = 3                                        # 每一个基站给VAP保留的RB个数
        self.n_DCUE2BS_RB = self.n_DC_UE                            # 每一个基站给DC-UE设备提供的RB个数 = DC-UE设备数
        self.n_BS_all_RB = self.n_DCUE2BS_RB + self.n_VAP2BS_RB     # 每一个基站的总RB个数
        self.BS_sbs = [15e3, 30e3, 60e3, 120e3, 240e3]              # 5G可选子载波间隔
        '''
        子载波间隔越大，一个时隙的实际长度越短。 如对于URRLC业务，可利用较大子载波间隔缩短数据传输时间，以满足时延要求;
        对于mMTc业务，可利用较小的较小的子载波间隔，增加数据传输时间，扩大覆盖范围。 
        '''
        self.BS_RB_BW = int(12 * self.BS_sbs[0])                    # 基站提供的RB带宽 = 12个子载波 * 子载波间隔
        # 基站天线增益 & 噪声
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        # 基站高度
        self.h_bs = 25  # BS antenna height 基站高度

        # ---------- 地图相关信息 ----------
        self.width = 250 * 3  # 地图宽 3个Road Grid
        self.height = 433 * 3  # 地图高 3个Road Grid
        # 地图中车道中心线位置
        # 向上行驶的车道中心线的x坐标
        self.up_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]
        # 向下行驶的车道中心线的x坐标
        self.down_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
                           750 - 3.5 / 2]
        # 向左行驶的车道中心线的y坐标
        self.left_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2,
                           866 + 3.5 + 3.5 / 2]
        # 向右行驶的车道中心线的y坐标
        self.right_lanes = [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2,
                            1299 - 3.5 - 3.5 / 2, 1299 - 3.5 / 2]

        # ---------- 车辆相关信息 ----------
        # 保存地图中所有车辆的相关信息的list
        self.vehicles = []

        # ---------- 信道相关信息 ----------
        # 对于VAP2BS和DCUE2BS 信道增益 = pathloss + shadowing + 随机快衰落
        # 所有车辆上的 所有DC-UE到 所有BS的 所有DC-UE可用RB的 信道增益
        self.DCUE2BS_Channels = np.zeros((self.n_veh, self.n_DC_UE, self.n_BS, self.n_DCUE2BS_RB))
        # 所有车辆上的 VAP到 所有BS的 所有VAP可用RB的 信道增益 self.n_VAP = self.n_veh
        self.VAP2BS_Channels = np.zeros((self.n_veh, self.n_BS, self.n_VAP2BS_RB))
        # 所有车辆上的 所有设备(DC-UE和WU)到 各自车辆上VAP 提供的所有VAP_RB的 信道增益
        self.inVeh_VAP_Channels = np.zeros((self.n_veh, self.n_WU + self.n_DC_UE, self.n_VAP_RB))

        # 所有车辆上的 所有DC-UE到 所有BS的 所有DC-UE可用RB的 路径损耗
        self.DCUE2BS_pathlosses = np.zeros((self.n_veh, self.n_DC_UE, self.n_BS, self.n_DCUE2BS_RB))
        # 所有车辆上的 VAP到 所有BS的 所有VAP可用RB的 路径损耗
        self.VAP2BS_pathlosses = np.zeros((self.n_veh, self.n_BS, self.n_VAP2BS_RB))

        # 所有车辆上的 所有DC-UE到 所有BS的 所有DC-UE可用RB的 阴影衰落
        self.DCUE2BS_shadowings = np.zeros((self.n_veh, self.n_DC_UE, self.n_BS, self.n_DCUE2BS_RB))
        # 所有车辆上的 VAP到 所有BS的 所有VAP可用RB的 阴影衰落
        self.VAP2BS_shadowings = np.zeros((self.n_veh, self.n_BS, self.n_VAP2BS_RB))

        # 所有车辆上的 所有DC-UE到 所有BS的 所有DC-UE可用RB的 慢衰落 = 路径损耗 + self.shadow_scale * 阴影衰落
        self.shadow_scale = 0.3  # 按照3GPP TR 36.885中计算的阴影衰落波动太大[-25, 25]，所以使用scale进行放缩
        self.DCUE2BS_slowfadings = np.zeros((self.n_veh, self.n_DC_UE, self.n_BS, self.n_DCUE2BS_RB))
        # 所有车辆上的 VAP到 所有BS的 所有VAP可用RB的 慢衰落 = 路径损耗 + 阴影衰落
        self.VAP2BS_slowfadings = np.zeros((self.n_veh, self.n_BS, self.n_VAP2BS_RB))

        # 所有车辆上的 所有DC-UE到 所有BS的 所有DC-UE可用RB的 随机快衰落
        self.DCUE2BS_fastfadings = np.zeros((self.n_veh, self.n_DC_UE, self.n_BS, self.n_DCUE2BS_RB))
        # 所有车辆上的 VAP到 所有BS的 所有VAP可用RB的 随机快衰落
        self.VAP2BS_fastfadings = np.zeros((self.n_veh, self.n_BS, self.n_VAP2BS_RB))

        # ---------- 距离更新矩阵相关信息 ----------
        # 所有车辆上的 所有DC-UE 到所有基站的距离更新矩阵，初始化全0
        # 第一维是车辆个数，第二维是车辆上的DC-UE数量，第三四维是某一DC-UE到所有基站的位置更新矩阵，大小为(n_BS,n_BS)的对角阵
        self.DCUE2BS_delta_distance = np.zeros((self.n_veh, self.n_DC_UE, self.n_BS, self.n_BS))
        # 所有车辆上的VAP到所有基站的位置更新矩阵，初始化全0
        self.VAP2BS_delta_distance = np.zeros((self.n_veh, self.n_BS, self.n_BS))

        # ---------- DC-UE相关信息 ----------
        # 所有车辆上的 所以DC-UE的流量需求; 10Mbps - 100Mbps
        self.DC_UEs_traffic_demand = np.random.uniform(10e6, 100e6, (self.n_veh, self.n_DC_UE))
        # DC-UE2BS可选发射功率 上行链路可选发射功率 参考 3GPP TS 38.101-1 V18.0.0 (2022-12)
        self.DC_UE_power_dB = [23, 26, 29, -100]  # -100 dBm表示不发送
        # 所有车辆上的 所以DC-UE的发射功率   初始均为23dBm
        self.DC_UEs_transmit_power = 23 * np.ones((self.n_veh, self.n_DC_UE))
        # DC-UE高度
        self.h_DC_UE = 1  # UE antenna height = 1m

        # ---------- 仿真步长相关信息 ----------
        # 快衰落/慢衰落更新时间
        self.time_fast = 0.001      # 快衰落 0.001s = 1ms
        self.time_slow = 0.1        # 慢衰落 update slow fading/vehicle position every 0.1s = 100 ms

    # 添加一辆车辆到地图上
    def add_new_vehicle(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    # 对给定数目的车辆进行初始化
    def add_new_vehicles_by_number(self, n_veh):
        for i in range(0, n_veh):
            # 随机生成车辆初始的前进方向
            direction = ['u', 'd', 'l', 'r']
            start_direction = direction[np.random.randint(0, 4, 1)[0]]
            # 随机生成车辆速度(单位m/s) 15-60 km/h; 1km/h = 0.2777m/s
            start_velocity = np.random.randint(15, 60) * 0.2777
            # 随机生成初始所处车道index
            start_lane_index = np.random.randint(0, len(self.up_lanes))
            # 随机生成初始位置
            if start_direction == 'u':
                start_position = [self.up_lanes[start_lane_index], np.random.randint(0, self.height)]
            if start_direction == 'd':
                start_position = [self.down_lanes[start_lane_index], np.random.randint(0, self.height)]
            if start_direction == 'l':
                start_position = [np.random.randint(0, self.width), self.left_lanes[start_lane_index]]
            if start_direction == 'r':
                start_position = [np.random.randint(0, self.width), self.right_lanes[start_lane_index]]
            # 添加车辆
            self.add_new_vehicle(start_position, start_direction, start_velocity)

    # 计算 标号为veh_index 的车辆上的 所有DC-UE和车机 的位置更新矩阵
    # 输入参数: veh_last_position 车辆上一次位置; veh_now_position 车辆当前位置; veh_index 车辆下标
    def renew_Vehi_Devs_delta_distance(self, veh_last_position, veh_now_position, veh_index):
        # 车辆上的DC-UE设备之间高度相同 但是与VAP高度不同
        # 且VAP水平位置 = 所有DC-UE水平位置 = 车辆水平位置
        # 所以DC-UE之间的位置更新矩阵相同 但不同于VAP的位置更新矩阵
        # 求解DC-UE到基站距离的子函数
        def DCUE2BS_distance(DCUE_x, DCUE_y, BS_x, BS_y):
            d1 = abs(DCUE_x - BS_x)
            d2 = abs(DCUE_y - BS_y)
            return math.sqrt(d1 ** 2 + d2 ** 2 + ((self.h_bs - self.h_DC_UE) ** 2))     # 单位m
        # 求解VAP到基站距离的子函数
        def VAP2BS_distance(VAP_x, VAP_y, BS_x, BS_y):
            d1 = abs(VAP_x - BS_x)
            d2 = abs(VAP_y - BS_y)
            return math.sqrt(d1 ** 2 + d2 ** 2 + ((self.h_bs - self.h_VAP) ** 2))       # 单位m

        VAP_delta_distance = []         # 车辆i的VAP到所有基站的距离变化量 长度=n_BS
        DCUE_delta_distance = []        # 车辆i的所有DC-UE到所有基站的距离变化量 长度=n_BS 假设所有DC-UE的距离变化相同
        # 遍历基站
        for BS in self.BS_positions:
            # VAP到某个基站的距离变化
            VAP_delta_distance.append(abs(VAP2BS_distance(veh_now_position[0], veh_now_position[1], BS[0], BS[1]) -
                                      VAP2BS_distance(veh_last_position[0], veh_last_position[1], BS[0], BS[1])))
            # DC-UE到某个基站的距离变化
            DCUE_delta_distance.append(abs(DCUE2BS_distance(veh_now_position[0], veh_now_position[1], BS[0], BS[1]) -
                                       DCUE2BS_distance(veh_last_position[0], veh_last_position[1], BS[0], BS[1])))

        # 更新 标号为veh_index 的车辆上VAP的距离更新矩阵，以对角阵形式构造
        diag_matrix_VAP = np.identity(self.n_BS)
        row, col = np.diag_indices_from(diag_matrix_VAP)
        diag_matrix_VAP[row, col] = np.array(VAP_delta_distance)
        self.VAP2BS_delta_distance[veh_index] = diag_matrix_VAP

        # 更新 标号为veh_index 的车辆上所有DC-UE的距离更新矩阵
        diag_matrix_DCUE = np.identity(self.n_BS)
        row, col = np.diag_indices_from(diag_matrix_DCUE)
        diag_matrix_DCUE[row, col] = np.array(DCUE_delta_distance)
        # 遍历车辆i上的所有DCUE
        for i in range(0, self.n_DC_UE):
            # 假设一辆车上所有的DC-UE位置相同，所以共享位置更新矩阵
            self.DCUE2BS_delta_distance[veh_index][i] = diag_matrix_DCUE

    # 更新所有车辆位置 并更新车内的DC-UE和车机的位置更新矩阵
    def renew_vehs_position_and_delta_distance(self):
        # 轮询车辆
        for i in range(0, self.n_veh):
            # 保存n-1时刻的车辆位置信息
            veh_last_position = (self.vehicles[i].position[0], self.vehicles[i].position[1])
            # 移动的距离
            veh_move_len = self.vehicles[i].velocity * self.time_slow
            # 是否进行了转向的标志，初始化为False
            change_direction = False
            # 判断当前车辆位置是否已进行了更新
            updated_position = False
            # 当前车辆上行中，且之前没有转向过，缺少change_direction的判断可能导致在一个十字路口车辆转向一次后，又在下一个if语句中继续转向
            if self.vehicles[i].direction == 'u' and ~change_direction and ~updated_position:
                for j in range(len(self.left_lanes)):
                    # 到达十字路口左转
                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] + veh_move_len) >= self.left_lanes[j]):
                        # 左转概率 25%
                        if np.random.uniform(0, 1) < 0.25:
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                        veh_move_len - (self.left_lanes[j] - self.vehicles[i].position[1])),
                                                         self.left_lanes[j]]
                            # self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                # 如果上次左转了，此处change_direction = True, 则不考虑右转
                # 如果没有左转
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        # 到达十字路口右转
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and (
                                (self.vehicles[i].position[1] + veh_move_len) >= self.right_lanes[j]):
                            # 右转概率 25%
                            if np.random.uniform(0, 1) < 0.25:
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                            veh_move_len - (self.right_lanes[j] - self.vehicles[i].position[1])),
                                                             self.right_lanes[j]]
                                # self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                # 未左转且未右转，保持直行
                if ~change_direction:
                    self.vehicles[i].position[1] += veh_move_len
                # 已遍历所有可能的行进方向，完成位置更新
                updated_position = True
                # 至此上行方向的车辆的三种位置更新可能：左、右、直都判断过；进入下一辆车的位置更新
            # ---------------------------------------------------------------------
            # 当前车辆下行中
            if self.vehicles[i].direction == 'd' and ~change_direction and ~updated_position:
                # 到达十字路口左转
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] - veh_move_len) <= self.left_lanes[j]):
                        # 左转概率 25%
                        if np.random.uniform(0, 1) < 0.25:
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                        veh_move_len - (self.vehicles[i].position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j]]
                            # self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                # 未左转，判断是否右转
                if ~change_direction:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (
                                (self.vehicles[i].position[1] - veh_move_len) <= self.right_lanes[j]):
                            # 右转
                            if np.random.uniform(0, 1) < 0.25:
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                            veh_move_len - (self.vehicles[i].position[1] - self.right_lanes[j])),
                                                             self.right_lanes[j]]
                                # self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                # 直行
                if ~change_direction:
                    self.vehicles[i].position[1] -= veh_move_len
                # 至此下行方向的车辆的三种位置更新可能：左、右、直都判断过；进入下一辆车的位置更新
                # 已遍历所有可能的行进方向，完成位置更新
                updated_position = True
            # ---------------------------------------------------------------------
            # 当前车辆右行
            if self.vehicles[i].direction == 'r' and ~change_direction and ~updated_position:
                # 转为向上车道
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] + veh_move_len) >= self.up_lanes[j]):
                        if np.random.uniform(0, 1) < 0.25:
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                        veh_move_len - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            # self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1]]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                # 转为向下车道
                if ~change_direction:
                    for j in range(len(self.down_lanes)):
                        # 这里对下行车道遍历，一旦满足位于十字路口的条件，就会开始判断是否向下转弯
                        # 如果没有np.random.uniform(0, 1) < 0.4: 语句
                        # j是从0开始遍历的，所以j一定是一个满足十字路口判断条件的最小值，所以车辆在向下转弯时，一定只会选择贴着建筑物的车道
                        # 也即，车辆不会选择到离建筑物较远的车道（因为其车道序号j较大）
                        # 但是由于转弯存在一定的概率，所以即便是离建筑物较远的车道也是可能被选择到的
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] + veh_move_len) >= self.down_lanes[j]):
                            if np.random.uniform(0, 1) < 0.25:
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                            veh_move_len - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                # self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1]]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                # 保持右行
                if ~change_direction:
                    self.vehicles[i].position[0] += veh_move_len
                # 至此右行方向的车辆的三种位置更新可能：上、下、直都判断过；进入下一辆车的位置更新
                # 已遍历所有可能的行进方向，完成位置更新
                updated_position = True
            # ---------------------------------------------------------------------
            # 当前车辆左行
            if self.vehicles[i].direction == 'l' and ~change_direction and ~updated_position:
                # 转为上行
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] - veh_move_len) <= self.up_lanes[j]):
                        if np.random.uniform(0, 1) < 0.25:
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                        veh_move_len - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            # self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1]]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                # 转为下行
                if ~change_direction:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] - veh_move_len) <= self.down_lanes[j]):
                            if np.random.uniform(0, 1) < 0.25:
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                            veh_move_len - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                # self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1]]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                # 保持左行
                if ~change_direction:
                    self.vehicles[i].position[0] -= veh_move_len
                # 至此左行方向的车辆的三种位置更新可能：上、下、直都判断过；进入下一辆车的位置更新
                # 已遍历所有可能的行进方向，完成位置更新
                updated_position = True
            # ---------------------------------------------------------------------

            # 车辆移动到地图边界，落出地图范围
            if self.vehicles[i].position[0] < 0 or self.vehicles[i].position[1] < 0 or self.vehicles[i].position[
                0] > self.width or self.vehicles[i].position[1] > self.height:
                if self.vehicles[i].direction == 'u':
                    # 上行转右行
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                    self.vehicles[i].direction = 'r'
                    # print('vehicle drop out up')
                else:
                    if self.vehicles[i].direction == 'd':
                        # 下行转左行
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                        self.vehicles[i].direction = 'l'
                        # print('vehicle drop out down')
                    else:
                        if self.vehicles[i].direction == 'l':
                            # 左行转上行
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                            self.vehicles[i].direction = 'u'
                            # print('vehicle drop out left')
                        else:
                            if self.vehicles[i].direction == 'r':
                                # 右行转下行
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]
                                self.vehicles[i].direction = 'd'
                                # print('vehicle drop out right')

            # ----------- 消除计算累计误差，将小车归位到车道中央 ---------------
            # 上行
            if self.vehicles[i].direction == 'u':
                # 偏离上行车道的误差
                error = abs(np.array(self.up_lanes) - self.vehicles[i].position[0])
                # 将车辆归位到车道中心线
                self.vehicles[i].position[0] = self.up_lanes[np.argmin(error)]
            # 下行
            elif self.vehicles[i].direction == 'd':
                # 偏离下行车道的误差
                error = abs(np.array(self.down_lanes) - self.vehicles[i].position[0])
                # 将车辆归位到车道中心线
                self.vehicles[i].position[0] = self.down_lanes[np.argmin(error)]
            # 右行
            elif self.vehicles[i].direction == 'r':
                # 偏离右行车道的误差
                error = abs(np.array(self.right_lanes) - self.vehicles[i].position[1])
                # 将车辆归位到车道中心线
                self.vehicles[i].position[1] = self.right_lanes[np.argmin(error)]
            # 左行
            elif self.vehicles[i].direction == 'l':
                # 偏离左行车道的误差
                error = abs(np.array(self.left_lanes) - self.vehicles[i].position[1])
                # 将车辆归位到车道中心线
                self.vehicles[i].position[1] = self.left_lanes[np.argmin(error)]
            else:
                print('Error: unsupported moving direction')

            # --------------- 计算某一辆车vehi上所有设备(DC-UE + VAP)的位置更新矩阵 --------------------
            # 车辆在时刻n的位置
            veh_now_position = (self.vehicles[i].position[0], self.vehicles[i].position[1])
            # 计算n-1时刻到n时刻 车辆i上的所有设备(DC-UE + VAP)的位置更新矩阵
            self.renew_Vehi_Devs_delta_distance(veh_last_position, veh_now_position, veh_index=i)

    # 更新所有信道的慢衰落信息 = pathloss + shadowing
    def renew_channels_slow_fading(self):
        """ Renew slow fading channel """
        # 遍历所有车辆
        for veh_index in range(0, len(self.vehicles)):
            # 对某辆车上的所有DC-UE 更新路径损耗 和 阴影衰落
            for DC_UE_index in range(0, self.n_DC_UE):
                # 写入路径损耗信息
                self.DCUE2BS_pathlosses[veh_index][DC_UE_index] = self.Cellular_Channel.get_path_loss(veh_position=self.vehicles[veh_index].position, cellular_device_type='DC-UE', n_available_RB=self.n_DCUE2BS_RB)
                # 写入阴影衰落信息
                self.DCUE2BS_shadowings[veh_index][DC_UE_index] = self.Cellular_Channel.get_shadowing(Dev_delta_distance=self.DCUE2BS_delta_distance[veh_index][DC_UE_index], Dev_last_shadowing=self.DCUE2BS_shadowings[veh_index][DC_UE_index], n_available_RB=self.n_DCUE2BS_RB)
                # 写入慢衰落信息
                self.DCUE2BS_slowfadings[veh_index][DC_UE_index] = self.DCUE2BS_pathlosses[veh_index][DC_UE_index] + np.multiply(self.shadow_scale, self.DCUE2BS_shadowings[veh_index][DC_UE_index])
            # 某辆车上的VAP到BS链路(仅一个VAP)更新路径损耗 和 阴影衰落
            # 写入路径损耗信息
            self.VAP2BS_pathlosses[veh_index] = self.Cellular_Channel.get_path_loss(veh_position=self.vehicles[veh_index].position, cellular_device_type='VAP', n_available_RB=self.n_VAP2BS_RB)
            # 写入阴影衰落信息
            self.VAP2BS_shadowings[veh_index] = self.Cellular_Channel.get_shadowing(Dev_delta_distance=self.VAP2BS_delta_distance[veh_index], Dev_last_shadowing=self.VAP2BS_shadowings[veh_index], n_available_RB=self.n_VAP2BS_RB)
            # 写入慢衰落信息
            self.VAP2BS_slowfadings[veh_index] = self.VAP2BS_pathlosses[veh_index] + np.multiply(self.shadow_scale, self.VAP2BS_shadowings[veh_index])

    # 更新所有的信道增益信息 = 快衰落 + 慢衰落
    def renew_channels_fast_slow_fading(self):
        # 遍历所有车辆
        for veh_index in range(0, len(self.vehicles)):
            # ------------- 对某辆车上的所有DC-UE -------------
            for DC_UE_index in range(0, self.n_DC_UE):
                # 生成随机快衰落
                self.DCUE2BS_fastfadings[veh_index][DC_UE_index] = np.abs(np.random.normal(0, 1, self.DCUE2BS_fastfadings[veh_index][DC_UE_index].shape) + 1j * np.random.normal(0, 1, self.DCUE2BS_fastfadings[veh_index][DC_UE_index].shape)) / math.sqrt(2)
                # 慢衰落 叠加 快衰落 生成信道增益
                self.DCUE2BS_Channels[veh_index][DC_UE_index] = self.DCUE2BS_fastfadings[veh_index][DC_UE_index] + self.DCUE2BS_slowfadings[veh_index][DC_UE_index]

            # ------------- 对某辆车上的VAP -------------
            # 生成随机快衰落
            self.VAP2BS_fastfadings[veh_index] = np.abs(np.random.normal(0, 1, self.VAP2BS_fastfadings[veh_index].shape) + 1j * np.random.normal(0, 1, self.VAP2BS_fastfadings[veh_index].shape)) / math.sqrt(2)
            # 慢衰落 叠加 快衰落 生成信道增益
            self.VAP2BS_Channels[veh_index] = self.VAP2BS_fastfadings[veh_index] + self.VAP2BS_slowfadings[veh_index]
            xxxxx = 1122

    # 初始化所有车辆到所有基站的阴影衰落
    def Initial_shadowing(self):
        '''
        Initial(at time 0)
        S_{eNB2UE, i}(0) = R * Ni(0)  设备i到所有基站的阴影衰落 大小为(n_BS, 1)
        '''
        # R矩阵
        R = np.sqrt(0.5 * np.ones([self.n_BS, self.n_BS]) + 0.5 * np.identity(self.n_BS))
        # 遍历车辆
        for i in range(0, self.n_veh):
            # 遍历DC-UE设备
            for j in range(0, self.n_DC_UE):
                Ni0 = np.random.normal(0, 8, self.n_BS)         # Ni0随机向量, 均值0，方差8，长度为n_BS
                shadowing = np.dot(R, Ni0)                      # shadowing大小为(n_BS, 1)
                # 同一个BS的所有RB共享阴影衰落值，进行矩阵扩展
                shadowing = np.repeat(shadowing, self.n_DCUE2BS_RB, axis=0).reshape((self.n_BS, self.n_DCUE2BS_RB))
                self.DCUE2BS_shadowings[i][j] = shadowing
            # 对于某辆车的VAP
            Ni0_VAP = np.random.normal(0, 8, self.n_BS)
            shadowing_VAP = np.dot(R, Ni0_VAP)
            self.VAP2BS_shadowings[i] = np.repeat(shadowing_VAP, self.n_VAP2BS_RB, axis=0).reshape(
                (self.n_BS, self.n_VAP2BS_RB))

    # 重新初始化环境
    def new_random_game(self):
        # 清空车辆list
        self.vehicles = []
        # 添加车辆
        self.add_new_vehicles_by_number(self.n_veh)
        # 初始化 所有DC-UE2BS 和 所有VAP2BS 的阴影衰落
        self.Initial_shadowing()


# 可视化所有车辆的移动
def plot_vehs_movement(env1, n_step):
    plt.ion()  # 打开交互模式
    plt.figure(dpi=150)  # 分辨率
    pathloss = []
    shadowing = []
    slowfading = []
    channel_gain = []

    for i in range(n_step):
        env1.renew_vehs_position_and_delta_distance()           # update vehicles position
        env1.renew_channels_slow_fading()                       # 更新信道慢衰落信息
        env1.renew_channels_fast_slow_fading()

        # 车辆1VAP 到基站1 的RB1的路径损耗
        pathloss.append(env1.VAP2BS_pathlosses[0][0][0])
        shadowing.append(env1.VAP2BS_shadowings[0][0][0])
        slowfading.append(env1.VAP2BS_slowfadings[0][0][0])
        channel_gain.append(env1.VAP2BS_Channels[0][0][0])

        # # 产生对数正态分布用于对比阴影衰落
        # lognormal.append(np.random.lognormal(mean=0.0, sigma=1, size=1)[0])

        # 间隔一段时间作图
        if i % 5 == 0:
            # 子图 1
            plt.subplot(1, len(env1.vehicles) + 2, 1)
            # 标题
            plt.title('time: ' + str(i * 0.1) + ' s')
            # 画出背景（车道）
            # 上车道 红
            for u_x in env1.up_lanes:
                plt.vlines(u_x, 0, env1.height, colors="r")
            # 下车道 蓝
            for d_x in env1.down_lanes:
                plt.vlines(d_x, 0, env1.height, colors="b")
            # 左车道 绿
            for l_y in env1.left_lanes:
                plt.hlines(l_y, 0, env1.width, colors='g')
            # 右车道 橘
            for r_y in env1.right_lanes:
                plt.hlines(r_y, 0, env1.width, colors='orange')
            # 保持图像横纵比
            plt.axis('equal')
            # 画出基站位置
            for BS_position in env1.BS_positions:
                plt.plot(BS_position[0], BS_position[1], '.b')
            # 遍历车辆
            for v in env1.vehicles:
                plt.plot(v.position[0], v.position[1], '*r')    # 车辆 红*
                # 车辆到基站连线
                for BS_position in env1.BS_positions:
                    plt.plot([v.position[0], BS_position[0]], [v.position[1], BS_position[1]], '-.g')   # 两点连线

            # 遍历车辆画局部图
            for m in range(2, len(env1.vehicles) + 2):
                # 车辆周边子图
                plt.subplot(1, len(env1.vehicles) + 2, m)
                plt.title('veh_' + str(m-1) + ' direction: ' + env1.vehicles[m-2].direction)
                # 画出背景（车道）
                # 上车道 红
                for u_x in env1.up_lanes:
                    plt.vlines(u_x, 0, env1.height, colors="r")
                # 下车道 蓝
                for d_x in env1.down_lanes:
                    plt.vlines(d_x, 0, env1.height, colors="b")
                # 左车道 绿
                for l_y in env1.left_lanes:
                    plt.hlines(l_y, 0, env1.width, colors='g')
                # 右车道 橘
                for r_y in env1.right_lanes:
                    plt.hlines(r_y, 0, env1.width, colors='orange')
                # 保持图像横纵比自动
                plt.axis('equal')
                x = env.vehicles[m-2].position[0]
                y = env.vehicles[m-2].position[1]
                # 车辆 红*
                plt.plot(x, y, '*r')
                # 车辆到基站连线
                for BS_position in env1.BS_positions:
                    # 两点连线
                    plt.plot([x, BS_position[0]], [y, BS_position[1]], '-.g')
                # 限制作图范围在车辆周围
                plt.xlim(x - 50, x + 50)
                plt.ylim(y - 50, y + 50)

            x_index = [k for k in range(0, i+1)]
            plt.subplot(1, len(env1.vehicles) + 2, len(env1.vehicles) + 2)

            plt.xlim(i - 100, i + 50)
            plt.plot(x_index, pathloss, 'r')
            # plt.plot(x_index, shadowing, 'g')
            plt.plot(x_index, slowfading, 'b')
            plt.plot(x_index, channel_gain, 'orange')


            plt.show()
            plt.pause(0.01)
            plt.clf()


env = Environ(n_veh=1, n_DC_UE=8, n_WU=4)
env.new_random_game()

plot_vehs_movement(env, n_step=10000)