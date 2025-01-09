import numpy as np
import matplotlib.pyplot as plt

# 博弈模型参数
n = 25  # 网络节点数
C0_values = [200, 400, 600, 800, 1000, 1200]  # 防御者的总防御资源
V_A = 5  # 攻击者每次可以攻击的目标数
theta = np.random.rand(n) * 100  # 节点的重要性（放大到 [0, 100] 范围）
p = lambda c: 1 - np.exp(-c)  # 防御成功的概率函数

# 改进的WCA算法参数
N = 50  # 规模大小
alpha = 4  # 规模因子
F = 0.6  # 差分进化的缩放因子
crossrate = 0.8  # 交叉率
max_iter = 500  # 最大迭代次数


# 改进的WCA算法
def improved_wca(C0, theta, V_A, N, alpha, F, crossrate, max_iter):
    # 初始化
    wolves = np.random.rand(N, n)  # 防御策略
    wolves = wolves / wolves.sum(axis=1, keepdims=True) * C0  # 确保总防御资源为C0

    # 迭代优化
    for iteration in range(max_iter):
        # 变异
        for i in range(N):
            a, b = np.random.choice(N, 2, replace=False)  # 随机选择两个点
            mutant = wolves[a] + F * (wolves[b] - wolves[a])  # 变异操作
            mutant = np.clip(mutant, 0, C0)  # 限制防御资源在[0, C0]范围内

            # 交叉
            cross_mask = np.random.rand(n) < crossrate
            wolves[i] = np.where(cross_mask, mutant, wolves[i])

        # 计算适应度
        fitness = np.array([-np.sum(theta * (1 - p(wolf))) - np.sum(wolf) for wolf in wolves])

        # 选择
        best_idx = np.argmax(fitness)
        wolves = np.array([wolves[best_idx]] * N)  # 保留最优个体

    return wolves[0]  # 返回最优防御策略


# 实验：不同防御资源C0对防御策略的影响
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
for i, C0 in enumerate(C0_values):
    # 求解最优防御策略
    defense_strategy = improved_wca(C0, theta, V_A, N, alpha, F, crossrate, max_iter)

    # 攻击策略：选择重要性最高的V_A个节点进行攻击
    attack_strategy = np.zeros(n)
    top_nodes = np.argsort(theta)[-V_A:]  # 选择重要性最高的V_A个节点
    attack_strategy[top_nodes] = 1  # 标记攻击的节点

    # 绘制子图
    row = i // 3
    col = i % 3
    ax = axs[row, col]

    # 创建双纵轴
    ax2 = ax.twinx()

    # 绘制重要程度柱状图
    ax.bar(range(n), theta, color='#1f77b4', alpha=0.8, label='Important Degree', width=0.8)

    # 绘制攻击策略柱状图
    ax.bar(range(n), attack_strategy * theta.max(), color='#ff7f0e', alpha=0.6, label='Attack Strategy', width=0.6)

    # 绘制防御策略折线图
    ax2.plot(range(n), defense_strategy, color='red', marker='o', markersize=5, label='Defend Strategy',
             linewidth=2)

    # 设置子图标题和标签
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Important Degree')
    ax2.set_ylabel('Defense Resources')
    ax.set_title(f'Defense Strategy with C0={C0}')

    # 合并图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')

    ax.grid(True)

plt.tight_layout(pad=3.0)
plt.show()