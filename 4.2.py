import matplotlib.pyplot as plt
import numpy as np

# 初始条件
x0_values = [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]  # 制造商A的初始选择“合作开发”策略的概率
y0_values = [0.3, 0.6]  # 制造商B的初始选择“合作开发”策略的概率
KA = 100  # 制造商A的补丁开发速度
KB = 100  # 制造商B的补丁开发速度
mu_A = 0.4  # 制造商A对共享信息的吸收率
tau = 0.4  # 补丁开发信息共享带来的溢出效应
xi = 0.3  # 漏洞被完全披露前黑客攻击的概率
rho_values = [0.01, 0.05]  # 漏洞被完全披露前黑客攻击成功的概率
psi = 10  # 漏洞被完全披露后黑客攻击能力的倍数
CA = 50  # 制造商A的补丁开发成本
CB = 50  # 制造商B的补丁开发成本
C0 = 30  # 共享成本
D = 100  # 黑客成功攻击漏洞时，制造商的最大损失
P = 10  # 潜在风险

# 动态方程
def dynamic_equation(x, y, KA, KB, CA, CB, C0, rho):
    U_Ayes = (1 - y) * (KA - CA - C0 - xi * D * rho - (1 - xi) * D * rho * psi) + \
             y * ((mu_A * KB + KA) * (1 + tau) - CA - D * rho)

    U_Ano = (1 - y) * (KA - CA - xi * D * rho - (1 - xi) * D * rho * psi) + \
            y * ((mu_A * KB + KA) - CA - xi * D * rho - (1 - xi) * D * rho * psi - P)

    U_Aavg = x * U_Ayes + (1 - x) * U_Ano

    return x * (U_Ayes - U_Aavg)

# 模拟演化过程
def simulate_evolution(x0, y0, rho, num_steps=100, initial_step_size=0.04, decay_rate=0.99):
    x = x0
    y = y0
    x_history = [x]
    step_size = initial_step_size  # 初始步长

    for epoch in range(num_steps):
        # 计算梯度（动态方程的返回值即为梯度）
        gradient = dynamic_equation(x, y, KA, KB, CA, CB, C0, rho)

        # 更新步长
        step_size = initial_step_size * (decay_rate ** epoch)

        # 更新x
        x += step_size * gradient
        x_history.append(x)

    return x_history

# 标记样式
markers = ['o', '^', 's', 'p', 'x', '*', '>', '.']

# 实验1：不同初始概率x0和y0对演化结果的影响
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
for i, y0 in enumerate(y0_values):
    for j, x0 in enumerate(x0_values):
        x_history = simulate_evolution(x0, y0, rho_values[0])
        axs[i].plot(range(len(x_history)), x_history, label=f'x0={x0}', marker=markers[j], markevery=5)

    axs[i].set_xlabel('Time Steps')
    axs[i].set_ylabel('Probability of Choosing Cooperative Development (X)')
    axs[i].set_title(f'Evolution of Manufacturer A\'s Strategy with y0={y0}')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

# 实验2：不同黑客攻击概率rho对演化结果的影响
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
for i, rho in enumerate(rho_values):
    for j, x0 in enumerate(x0_values):
        x_history = simulate_evolution(x0, y0_values[0], rho)
        axs[i].plot(range(len(x_history)), x_history, label=f'x0={x0}', marker=markers[j], markevery=5)

    axs[i].set_xlabel('Time Steps')
    axs[i].set_ylabel('Probability of Choosing Cooperative Development (X)')
    axs[i].set_title(f'Evolution of Manufacturer A\'s Strategy with y0={y0_values[0]} and rho={rho}')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()