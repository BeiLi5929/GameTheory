import numpy as np
import matplotlib.pyplot as plt


# 动态方程
def dynamic_equation(x, y, omega1, omega2, gamma, lambda_beta, C1, K1, K2, alpha1, alpha2, e):
    # 白帽黑客1选择知识共享策略的期望收益
    U_yes = (1 - y) * (K1 * alpha1 * e + ((K1 * omega1 + K2) * (1 + gamma) * alpha2 + K1 * alpha1) * lambda_beta - C1) + \
            y * ((K2 * omega2 + K1) * (1 + gamma) * alpha1 * e + (
                (K2 * omega2 + K1) * (1 + gamma) * alpha1 + (K1 * omega1 + K2) * (
                    1 + gamma) * alpha2) * lambda_beta - C1)

    # 白帽黑客1选择不共享策略的期望收益
    U_no = y * (K2 * omega2 + K1) * (1 + gamma) * alpha1 * e + (1 - y) * K1 * alpha1 * e

    # 白帽黑客1的平均收益
    U_avg = x * U_yes + (1 - x) * U_no

    # 白帽黑客1选择知识共享策略的动态方程
    return x * (U_yes - U_avg)


# 实验1
def experiment_1():
    # 参数设置
    omega1 = 0.5  # 白帽黑客1的信任度
    omega2 = 0.5  # 白帽黑客2的信任度
    gamma = 0.3   # 增加知识增值率
    lambda_beta = 0.45  # 增加漏洞奖励率
    C1 = 0.9        # 知识共享成本
    K1 = 1        # 白帽黑客1的固有知识量
    K2 = 1       # 白帽黑客2的固有知识量
    alpha1 = 0.6  # 白帽黑客1的漏洞转换率
    alpha2 = 0.6  # 白帽黑客2的漏洞转换率
    e = 1         # 漏洞的平均收益
    x0_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]  # 白帽黑客1的初始概率
    y0_values = [0.1, 0.9]  # 白帽黑客2的初始概率
    t = np.arange(0, 100, 0.05)  # 时间步长为 0.05
    markers = ['o', '^', 's', 'D', 'P', '*']  # 不同的标记样式

    # 绘制图像
    plt.figure(figsize=(14, 6))

    for i, y0 in enumerate(y0_values):
        plt.subplot(1, 2, i + 1)
        for j, x0 in enumerate(x0_values):
            x = [x0]
            for _ in t[1:]:
                dx = dynamic_equation(x[-1], y0, omega1, omega2, gamma, lambda_beta, C1, K1, K2, alpha1, alpha2, e)
                x.append(x[-1] + dx * 0.05)  # 时间调整为 0.05

            # 使用相同的颜色，不同的标记样式
            plt.plot(t, x, label=f'x0={x0}', marker=markers[j], markevery=50)

        plt.xlabel('Time')
        plt.ylabel('Probability of choosing KSS (X)')
        plt.title(f'Influence of initial probabilities (y0={y0})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# 实验2：不同x0和ω1的影响
def experiment_2():
    # 参数设置
    omega2 = 0.5  # 白帽黑客2的信任度
    gamma = 0.5   # 知识增值率
    lambda_beta = 0.45  # 漏洞奖励率
    C1 = 0.93       # 知识共享成本
    K1 = 1        # 白帽黑客1的固有知识量
    K2 = 1        # 白帽黑客2的固有知识量
    alpha1 = 0.6  # 白帽黑客1的漏洞转换率
    alpha2 = 0.6  # 白帽黑客2的漏洞转换率
    e = 1         # 漏洞的平均收益
    x0_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]  # 白帽黑客1的初始概率
    y0 = 0.3      # 白帽黑客2的初始概率
    omega1_values = [0.3, 0.5]  # 白帽黑客1的信任度
    t = np.arange(0, 100, 0.1)  # 时间步长
    markers = ['o', '^', 's', 'D', 'P', '*']  # 不同的标记样式

    # 绘制图像
    plt.figure(figsize=(14, 6))

    for i, omega1 in enumerate(omega1_values):
        plt.subplot(1, 2, i + 1)
        for j, x0 in enumerate(x0_values):
            x = [x0]
            for _ in t[1:]:
                dx = dynamic_equation(x[-1], y0, omega1, omega2, gamma, lambda_beta, C1, K1, K2, alpha1, alpha2, e)
                x.append(x[-1] + dx * 0.1)  # 时间步长调整为 0.1

            # 使用相同的颜色，不同的标记样式
            plt.plot(t, x, label=f'x0={x0}', marker=markers[j], markevery=50)

        plt.xlabel('Time')
        plt.ylabel('Probability of choosing KSS (X)')
        plt.title(f'Influence of trust degree (y0={y0}, omega1={omega1})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# 实验3：不同x0和γ的影响
def experiment_3():
    # 参数设置
    omega1 = 0.5  # 白帽黑客1的信任度
    omega2 = 0.5  # 白帽黑客2的信任度
    lambda_beta = 0.45  # 漏洞奖励率
    C1 = 0.9        # 知识共享成本
    K1 = 1        # 白帽黑客1的固有知识量
    K2 = 1        # 白帽黑客2的固有知识量
    alpha1 = 0.6  # 白帽黑客1的漏洞转换率
    alpha2 = 0.6  # 白帽黑客2的漏洞转换率
    e = 1         # 漏洞的平均收益
    x0_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]  # 白帽黑客1的初始概率
    y0 = 0.3      # 白帽黑客2的初始概率
    gamma_values = [0.3, 0.6]  # 知识增值率
    t = np.arange(0, 100, 0.1)  # 时间步长
    markers = ['o', '^', 's', 'D', 'P', '*']  # 不同的标记样式

    # 绘制图像
    plt.figure(figsize=(14, 6))

    for i, gamma in enumerate(gamma_values):
        plt.subplot(1, 2, i + 1)
        for j, x0 in enumerate(x0_values):
            x = [x0]
            for _ in t[1:]:
                dx = dynamic_equation(x[-1], y0, omega1, omega2, gamma, lambda_beta, C1, K1, K2, alpha1, alpha2, e)
                x.append(x[-1] + dx * 0.1)  # 时间步长调整为 0.1

            # 使用相同的颜色，不同的标记样式
            plt.plot(t, x, label=f'x0={x0}', marker=markers[j], markevery=50)

        plt.xlabel('Time')
        plt.ylabel('Probability of choosing KSS (X)')
        plt.title(f'Influence of knowledge value-added rate (y0={y0}, gamma={gamma})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    experiment_1()
    experiment_2()
    experiment_3()