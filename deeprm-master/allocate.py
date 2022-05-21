# -*- coding: utf-8 -*-
"""
@Copyright (C) 2022 mewhaku . All Rights Reserved 
@Time ： 2022/3/6 14:51
@Author ： mewhaku
@File ：allocate.py
@IDE ：PyCharm

目的描述 ：
  通过外部分配实现资源调度

方法描述：
    使用深度图像表示集群资源状况和任务的资源需求情况，将集群资源描述矩
    阵T*N*M和任务的资源需求描述矩阵1*T*M连接在一起，形成一个(N+1)*T*M维的矩阵，将之
    作为深度强化学习的状态s，输入所述Q神经网络对将任务放置在每个机器的Q值进行评估，
    进而通过Q值比较选出任务要放置的机器，产生调度动作；其中N表示计算机集群中机器的
    个数，T表示未来T个时间粒度，M表示资源种类数。

parameters：
    N = num_machine
    T = time_horizon
    M = res_slot


"""
import parameters
import numpy as np

class Allocate:
    def __init__(self, pa):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot

        self.num_machine = pa.num_machine     #添加的机器数参数

        self.avbl_res = np.ones((self.res_slot + 1), self.time_horizon, self.num_machine)    #集群资源描述矩阵T*N*M和任务的资源需求描述矩阵1*T*M连接在一起，形成一个(N+1)*T*M维的矩阵 深度强化学习的状态s


        #单个机器的资源数组
        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot
        #扩展到多机器
        self.avbl_slot_multi = np.ones((self.num_machine,self.time_horizon, self.num_res)) * self.res_slot   #生成time_horizon行 num_res列 num_machine个 的每个数为res_slot 的数组   集群机器的总可用槽数


        self.running_job = []
        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)
        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))


        #图像描述
        self.canvas = np.zeros((pa.num_machine, pa.time_horizon, pa.num_res))


    #分配机器
    def allocate_machine(self, machine, job, curr_time):

        allocated = False

        for t in range(0, self.num_machine):


            self.allocate_job(self, job, curr_time)




    #分配作业
    def allocate_job(self, job, curr_time):

        allocated = False

        for t in range(0, self.time_horizon - job.len):

            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec  #不懂

            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0]
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color

                break

        return allocated

    def time_proceed(self, curr_time):

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot

        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0


"""
定义Machine类
machine_len  机器数
...
"""
class Machine:
    def __init__(self, res_vec, machine_id, enter_time, pa):
        self.id = machine_id
        self.res_vec = res_vec
        self.len = pa.num_machine
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1

"""
定义Job类
...
"""
class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1
