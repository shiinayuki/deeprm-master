# DeepRM
HotNets'16 http://people.csail.mit.edu/hongzi/content/publications/DeepRM-HotNets16.pdf

Install prerequisites

```
sudo apt-get update
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
pip install --user Theano
pip install --user Lasagne==0.1
sudo apt-get install python-matplotlib
```

In folder RL, create a data/ folder. 
  在RL文件夹下创建数据或者文件夹

Use `launcher.py` to launch experiments. 
#使用launcher.py 来启动实验


```
--exp_type <type of experiment>
  实验类型
--num_res <number of resources> 
  资源数量
--num_nw <number of visible new work> 
  可视的新工作数量
--simu_len <simulation length> 
  模拟长度
--num_ex <number of examples>
  案例数量 
--num_seq_per_batch <rough number of samples in one batch update>
  样本在一批更新中的原始数量
--eps_max_len <episode maximum length (terminated at the end)>
  章节最大长度（在最终停止）
--num_epochs <number of epoch to do the training>
  训练的代数
--time_horizon <time step into future, screen height>
  未来的时间步长，屏幕高度
--res_slot <total number of resource slots, screen width> 
  资源槽的总数量，屏幕宽度
--max_job_len <maximum new job length> 
  最大新任务的长度
--max_job_size <maximum new job resource request> 
  最大新任务资源请求
--new_job_rate <new job arrival rate> 
  新任务的到达率
--dist <discount factor>
  折扣因子（马尔科夫决策） 
--lr_rate <learning rate> 
  学习率
--ba_size <batch size>
  批大小 
--pg_re <parameter file for pg network>
  Pixel to Global Matching Network像素全局查找网络 像素全局查找网络参数文件
--v_re <parameter file for v network>
  v型网络的参数文件
--q_re <parameter file for q network> 
  q网络的参数文件
--out_freq <network output frequency> 
  网络输出频率
--ofile <output file name> 
  输出文档名字
--log <log file name> 
  记录文件名字
--render <plot dynamics>
  plot dynamics绘制时间关系插件
--unseen <generate unseen example> 
  生成不可见案例
```


The default variables are defined in `parameters.py`.


Example: 
  - launch supervised learning for policy estimation 
  - 监督学习
  
  ```
  python launcher.py 
  --exp_type=pg_su 策略梯度算法 
  --simu_len=50 模拟次数 50 
  --num_ex=1000 样本（作业个数） 1000 
  --ofile=data/pg_su 保存文件/pg_su
  --out_freq=10 保存频率
  ```
  - launch policy gradient using network parameter just obtained
  
  ```
  python launcher.py --exp_type=pg_re --pg_re=data/pg_su_net_file_20.pkl --simu_len=50 --num_ex=10 --ofile=data/pg_re
  ```
  - launch testing and comparing experiemnt on unseen examples with pg agent just trained
  
  ```
  python launcher.py --exp_type=test --simu_len=50 --num_ex=10 --pg_re=data/pg_re_1600.pkl --unseen=True
  ```
