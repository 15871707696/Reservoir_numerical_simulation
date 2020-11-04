import pandas as pd
import numpy as np
from math import pi, log
import matplotlib.pyplot as plt
import seaborn as sns

# 单相渗流 油藏数值模拟习题
# 采用SI单位制
# 参数初始化
p_init = 20 * 10 ** 6  # 初始压力 注意这里压力单位为：MPa
deta_x = 200  # 采用均匀网格
deta_y = 200
rw = 0.1
re = 0.208 * deta_x
miu = 5 * 10 ** (-3)  # 粘度
C = 2 * 10 ** (-10)  # 压缩系数
wells = {"coord": [(3, 7), (7, 4)], "well": ["w1", "w2"], "Q": [30 / 86400, 0],
         "pwf": [0, 15 * 10 ** 6]}  # 井的信息  定产生产井pwf为0，定压生产井q为0
# 全局变量
grid_index = np.zeros((11, 11))  # 给定网格的编号 以区分边界与内部

# 作业中油藏静态参数依据
res_attr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 一行表示 i ,列着表示j，0表示虚拟网格
                     [0, 0, 0, 0, 355, 340, 350, 330, 310, 259, 0],
                     [0, 0, 0, 0, 335, 320, 300, 290, 240, 222, 0],
                     [0, 0, 0, 0, 315, 290, 280, 270, 235, 200, 0],
                     [0, 355, 340, 325, 310, 310, 259, 250, 228, 190, 0],
                     [0, 335, 320, 300, 290, 240, 222, 230, 210, 180, 0],
                     [0, 315, 290, 280, 270, 235, 200, 205, 195, 185, 0],
                     [0, 295, 260, 240, 250, 228, 190, 0, 0, 0, 0],
                     [0, 275, 235, 210, 230, 210, 180, 0, 0, 0, 0],
                     [0, 255, 225, 215, 205, 195, 185, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                     ])

h = res_attr.copy() / 70  # 储层厚度场
k = (res_attr.copy() + 80) * 10 ** (-15)  # 储层渗透率场
poro = (res_attr.copy() * 0.05 + 10) / 100  # 储层孔隙度场

# 初始条件
P_I = np.full((11, 11), p_init, dtype=np.float64)  # 初始压力场

# 无效网格取为0，有效网格取为正数
# 确定定压边界网格  编号为1
for i in range(1, 4):
    grid_index[i, 4] = 1
for i in range(1, 4):
    grid_index[4, i] = 1
for i in range(5, 10):
    grid_index[i, 1] = 1

# 确定封闭边界网格  编号为2
for j in range(5, 9):
    grid_index[1, j] = 2
grid_index[1, 9] = 2
for i in range(2, 6):
    grid_index[i, 9] = 2
grid_index[6, 9] = 2
for j in range(7, 9):
    grid_index[6, j] = 2
for i in range(7, 9):
    grid_index[i, 6] = 2
grid_index[9, 6] = 2
for j in range(2, 6):
    grid_index[9, j] = 2

# 确定内部网格  根据边界网格自行确定内部网格 编号为3
(rows, cols) = grid_index.shape
for i in range(rows):
    s = 0
    all = sum(grid_index[i])
    for j in range(cols):
        if grid_index[i, j] != 0:
            s += grid_index[i, j]
            continue
        if s > 0 and s < all:
            grid_index[i, j] = 3
        if s == all:
            break


# 系数计算 输入为（i，j)
def hk(i, j):
    return h[i, j] * k[i, j]


def c(i, j, deta_T):
    if grid_index[i, j - 1] == 0:
        return deta_T * hk(i, j) / (miu * C * h[i, j] * poro[i, j] * deta_y ** 2)
    else:
        return deta_T * 2 * hk(i, j - 1) * hk(i, j) / (hk(i, j - 1) + hk(i, j)) / (
                miu * C * h[i, j] * poro[i, j] * deta_y ** 2)


def a(i, j, deta_T):
    if grid_index[i - 1, j] == 0:
        return deta_T * hk(i, j) / (miu * C * h[i, j] * poro[i, j] * deta_y ** 2)
    else:
        return deta_T * 2 * hk(i - 1, j) * hk(i, j) / (hk(i - 1, j) + hk(i, j)) / (
                miu * C * h[i, j] * poro[i, j] * deta_x ** 2)


def b(i, j, deta_T):
    if grid_index[i + 1, j] == 0:
        return deta_T * hk(i, j) / (miu * C * h[i, j] * poro[i, j] * deta_y ** 2)
    else:
        return deta_T * 2 * hk(i + 1, j) * hk(i, j) / (hk(i + 1, j) + hk(i, j)) / (
                miu * C * h[i, j] * poro[i, j] * deta_x ** 2)


def d(i, j, deta_T):
    if grid_index[i, j + 1] == 0:
        return deta_T * hk(i, j) / (miu * C * h[i, j] * poro[i, j] * deta_y ** 2)
    else:
        return deta_T * 2 * hk(i, j + 1) * hk(i, j) / (hk(i, j + 1) + hk(i, j)) / (
                miu * C * h[i, j] * poro[i, j] * deta_y ** 2)


def e(i, j, deta_T):
    return 1 - c(i, j, deta_T) - a(i, j, deta_T) - b(i, j, deta_T) - d(i, j, deta_T)


# 输出生产信息
def q_pwf(p1):  # 输出某个压力场所对应的所有井的生产信息  传入参数为油藏区域压力场（这里传入的为转换为题目坐标形式的DateFram）
    for i in range(len(wells["well"])):
        coord_ = wells["coord"][i]
        Q_ = wells["Q"][i]
        pwf_ = wells["pwf"][i]
        if Q_ != 0:
            print("定产%s(m^3/d)的生产井%s的井底压力为%s(MPa)" % (Q_ * 86400, coord_, p1[coord_[0]][coord_[1]]))
        else:
            print("定压%s(MPa)的生产井%s的产量为%s(m^3/d)" % (pwf_ * 10 ** (-6), coord_, (2 * pi * hk(coord_[0], coord_[1]) * (
                    p1[coord_[0]][coord_[1]] * 10 ** 6 - pwf_) / (miu * log(re / rw)) * 86400)))


# 绘图
def pic(target):  # 传入参数为油藏区域压力场（这里传入的为转换为题目坐标形式的DateFram）
    # 设置中文和负号正常显示
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    ax = sns.heatmap(target,  # 指定绘图数据
                     cmap='rainbow',  # 指定填充色
                     linewidths=.1,  # 设置每个单元方块的间隔
                     annot=True,  # 显示数值
                     fmt='.4f',  # 保留小数点后四位
                     annot_kws={'size': 5, 'color': 'black'}  # 矩阵上数字的大小颜色
                     )
    # 添加x轴刻度标签(加0.5是为了让刻度标签居中显示)
    plt.xticks(np.arange(11) + 0.5, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # 刻度标签置于顶部显示
    # ax.xaxis.tick_top()
    # 添加y轴刻度标签
    plt.yticks(np.arange(11) + 0.5, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    # 旋转y刻度0度，即水平显示
    plt.yticks(rotation=0)
    # 设置标题和坐标轴标签
    ax.set_title('油藏区域压力分布图')
    ax.set_xlabel('')
    ax.set_ylabel('')
    # 显示图形
    plt.show()


# 显式方法进行压力计算
def ex_pre(deta_T=6120):  # 传入参数为时间步长（s)默认为1.7h，输出为所有稳定前时间步的压力场（题目坐标形式的DateFram)的列表
    steps = 0
    p_list = []  # 存放每个时间步的压力场的列表
    p1 = P_I.copy()  # 下一个时间步的压力场
    p_list.append(p1.copy())
    while True:
        p = p_list[steps]
        for i in range(rows):
            for j in range(cols):
                if grid_index[i, j]:
                    if grid_index[i, j] == 1:  # 定压边界
                        p[i, j] = p_init
                        continue
                    if grid_index[i, j - 1] == 0:  # 注意这里可能会使虚拟网格处的压力变化，不过没有影响
                        p[i, j - 1] = p[i, j]
                    if grid_index[i, j + 1] == 0:
                        p[i, j + 1] = p[i, j]
                    if grid_index[i - 1, j] == 0:
                        p[i - 1, j] = p[i, j]
                    if grid_index[i + 1, j] == 0:
                        p[i + 1, j] = p[i, j]
                    if not (i, j) in wells["coord"]:  # 网格中没有井
                        q = 0
                    else:
                        index = wells["coord"].index((i, j))
                        if wells["Q"][index] != 0:  # 定产生产井
                            q = -wells["Q"][index] / (h[i, j] * deta_x * deta_y)
                        else:  # 定压生产井
                            q = (-2 * pi * hk(i, j) * (p[i, j] - wells["pwf"][index]) / (miu * log(re / rw))) / (
                                    h[i, j] * deta_x * deta_y)
                    p1[i, j] = c(i, j, deta_T) * p[i, j - 1] + a(i, j, deta_T) * p[i - 1, j] + e(i, j, deta_T) * p[
                        i, j] + b(i, j, deta_T) * p[i + 1, j] + d(i, j, deta_T) * p[i, j + 1] + deta_T * q / (
                                       C * poro[i, j])
        p_list.append(p1.copy())
        steps += 1
        dp = p_list[steps] - p_list[steps - 1]
        if np.abs(np.where(grid_index > 0, dp, 0)).max() * 10 ** (-6) <= 0.000001:  # 判断是否稳定
            break
    print("显式方法稳定时间为:第%s天" % (steps * deta_T / 86400))
    p_list = [np.where(grid_index != 0, i, np.nan) for i in p_list]  # 去除无效网格
    p_list = [pd.DataFrame((i * 10 ** (-6)).T).reindex([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) for i in p_list]
    # 将压力分布变换成题目中的形式（即进行坐标轴的变换）同时把单位变成MPa用于输出，此时输出为含DateFram形式的压力场的列表
    q_pwf(p_list[-1])  # 输出为稳定时间步的生产信息
    return p_list


# 隐式方法进行压力计算
def in_pre(deta_T=86400):  # 传入参数为时间步长（s)默认为1天，输出为稳定前的所有时间步的压力场（题目坐标形式的DateFram)的列表
    steps = 0
    p_list = []  # 存放每个时间步的压力场的列表
    p1 = P_I.copy()  # 下一个时间步的压力场
    p1 = np.where(grid_index != 0, p1, np.nan)  # 去除无效网格
    p_list.append(p1.copy())
    A = np.zeros((52, 52))  # 存放方程组系数的系数矩阵
    G = np.zeros((52, 1))  # 存放方程组右端项
    bian_hao = np.full((11, 11), -1, dtype=np.int8)
    n = 0
    for i in range(rows):  # 进行网格编号（这里是按列进行编号)
        for j in range(cols):
            if grid_index[i, j] in (2, 3):
                bian_hao[i, j] = n
                n += 1
    while True:
        p = p_list[steps]
        for i in range(rows):
            for j in range(cols):
                if bian_hao[i, j] != -1:
                    G[bian_hao[i, j], 0] = -p[i, j]
                    A[bian_hao[i, j], bian_hao[i, j]] = e(i, j, deta_T) - 2  # 注意隐式中e(i,j)表达式变化了

                    if grid_index[i, j + 1] == 0:  # 考虑一个网格周围存在封闭边界网格的情况
                        A[bian_hao[i, j], bian_hao[i, j]] += d(i, j, deta_T)
                    elif bian_hao[i, j + 1] != -1:
                        A[bian_hao[i, j], bian_hao[i, j + 1]] = d(i, j, deta_T)

                    if grid_index[i + 1, j] == 0:
                        A[bian_hao[i, j], bian_hao[i, j]] += b(i, j, deta_T)
                    elif bian_hao[i + 1, j] != -1:
                        A[bian_hao[i, j], bian_hao[i + 1, j]] = b(i, j, deta_T)

                    if grid_index[i, j - 1] == 0:
                        A[bian_hao[i, j], bian_hao[i, j]] += c(i, j, deta_T)
                    elif bian_hao[i, j - 1] != -1:
                        A[bian_hao[i, j], bian_hao[i, j - 1]] = c(i, j, deta_T)

                    if grid_index[i - 1, j] == 0:
                        A[bian_hao[i, j], bian_hao[i, j]] += a(i, j, deta_T)
                    elif bian_hao[i - 1, j] != -1:
                        A[bian_hao[i, j], bian_hao[i - 1, j]] = a(i, j, deta_T)

                    if grid_index[i, j + 1] == 1:  # 考虑一个网格周围存在定压边界网格的情况
                        G[bian_hao[i, j], 0] -= d(i, j, deta_T) * p_init

                    if grid_index[i + 1, j] == 1:
                        G[bian_hao[i, j], 0] -= b(i, j, deta_T) * p_init

                    if grid_index[i, j - 1] == 1:
                        G[bian_hao[i, j], 0] -= c(i, j, deta_T) * p_init

                    if grid_index[i - 1, j] == 1:
                        G[bian_hao[i, j], 0] -= a(i, j, deta_T) * p_init

                    if (i, j) in wells["coord"]:  # 考虑网格中存在井井的情况
                        index = wells["coord"].index((i, j))
                        if wells["Q"][index] != 0:  # 定产生产井
                            q = wells["Q"][index] / (h[i, j] * deta_x * deta_y)
                            G[bian_hao[i, j], 0] += deta_T * q / (C * poro[i, j])
                        else:  # 定压生产井
                            J = -2 * pi * k[i, j] * deta_T / (miu * C * poro[i, j] * deta_x * deta_y * log(re / rw))
                            A[bian_hao[i, j], bian_hao[i, j]] += J
                            G[bian_hao[i, j], 0] += J * wells["pwf"][index]
        result = np.linalg.solve(A, G)
        for i in range(rows):
            for j in range(cols):
                if bian_hao[i, j] != -1:
                    p1[i, j] = result[bian_hao[i, j], 0]
        p_list.append(p1.copy())
        steps += 1
        dp = p_list[steps] - p_list[steps - 1]
        if np.abs(np.where(grid_index > 0, dp, 0)).max() * 10 ** (-6) <= 0.000001:  # 判断是否稳定
            break
    print("隐式方法稳定时间为:第%s天" % (steps * deta_T / 86400))
    p_list = [pd.DataFrame((i * 10 ** (-6)).T).reindex([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) for i in p_list]
    pd.DataFrame(p_list).to_excel("D:/homework.xlsx")
    # 将压力分布变换成题目中的形式（即进行坐标轴的变换）同时把单位变成MPa用于输出，此时输出为含DateFram形式的压力场的列表
    q_pwf(p_list[-1])  # 输出为稳定时间步的生产信息
    return p_list


List1 = ex_pre()

List2 = in_pre()

# 初始时刻
pic(List1[0])
# 10 天
pic(List1[140])
# 一个月
pic(List1[422])
# 稳定时刻
pic(List1[-1])
