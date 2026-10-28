import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D # 导入 Line2D 用于折线图图例
import matplotlib.ticker as mticker # 导入 mticker 用于刻度格式化

# ====== 数据定义 ======
recall = [0.84, 0.86, 0.88, 0.90, 0.92, 0.94]

# Sift100M 数据
sift_dpu_rate = {
    "Prism":    [0.6283, 0.6742, 0.7993, 0.8182, 0.8228, 0.8261],
    "PIMANN":   [0.3395, 0.3483, 0.3493, 0.3561, 0.3593, 0.3749], 
    "DRIM-ANN": [0.0978, 0.1138, 0.1202, 0.1306, 0.1466, 0.1671]
}

# SpaceV100M 数据
space_dpu_rate = {
    "Prism":    [0.8731, 0.8792, 0.8811, 0.886, 0.8876, 0.8969],
    "PIMANN":   [0.2294, 0.2309, 0.2398, 0.2442, 0.2444, 0.2463], 
    "DRIM-ANN": [0.0974, 0.123, 0.1394, 0.1711, 0.1999, 0.2412]
}

# ====== 颜色定义 ======
colors = {
    "Prism": "#b7e3a8",       # 绿色
    "PIMANN": "#a9c1e6",     # 蓝色 
    "DRIM-ANN": "#c6b3e0",   # 紫色 
}

# ====== 标记定义 (新增) ======
markers = {
    "Prism": 'o',       # 圆圈
    "PIMANN": 's',     # 方块
    "DRIM-ANN": 'D',   # 菱形
}

# 根据新的数据集调整keys_order
keys_order = ["Prism", "PIMANN", "DRIM-ANN"]

# ====== Y 轴限制 (对数刻度下的独立限制) ======
# Y 轴对数刻度范围设置为 50ms 到 20000 / 50000 ms
Y_MIN = 0
Y_MAX = 1

# 定义通用的对数刻度点（10的幂）
TICKS_COMMON = [0.2, 0.4, 0.6, 0.8]


# ====== 全局字体与样式（Arial） ======
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"]
})

# 将 recall 转换为浮点数列表作为 x 轴坐标
recall_float = [float(r) for r in recall] 

# ===================================================
# 1. 生成独立的图例 (dpu_rate_legend.pdf)
# ===================================================
def generate_legend():
    # 创建一个极小的图和轴，用于承载图例
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis('off') # 隐藏坐标轴

    # 构造图例句柄 (使用 Line2D 来显示标记和线条)
    legend_handles = [
        Line2D([0], [0], 
               marker=markers[k], 
               color='black',              # 线条颜色
               markerfacecolor=colors[k],  # 标记填充色
               markeredgecolor='black',    # 标记边框色
               markersize=8, 
               linewidth=2,
               label=k,
               linestyle='-',
               alpha=0.75) # 应用 80% 透明度到图例
        for k in keys_order
    ]

    # 将图例添加到图上，并定位到中央
    legend = fig.legend(
        handles=legend_handles, 
        ncol=4, 
        loc="center", 
        frameon=False # 移除图例外框
    )

    # 确保在保存时只包含图例
    fig.savefig("dpu_rate_legend.pdf", 
                bbox_inches=legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted()), 
                pad_inches=0.1)

    print("✅ Saved legend to dpu_rate_legend.pdf")
    plt.close(fig)


def generate_space_plot():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.5)) 

    # 绘图 (SPACE-100M)
    for key in keys_order:  # 注意这里更新为实际数据存在的三条线
        y_percent = [v * 100 for v in space_dpu_rate[key]]  # 转换为百分比
        ax.plot(recall_float, y_percent, 
                marker=markers.get(key, 'o'),
                linestyle='-',
                color=colors.get(key, 'gray'),
                linewidth=2,
                markeredgecolor='black',
                markersize=8,
                label=key,
                alpha=0.85)

    # === 坐标轴设置 ===
    ax.set_xticks(recall_float)
    ax.set_xticklabels(recall)
    ax.set_xlabel("Recall@100")

    ax.set_ylim(0, 100)  # 百分比范围
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())  # 显示为 %

    ax.set_ylabel("DPU Utilization (%)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    fig.savefig("dpu_rate_spacev100m.pdf", bbox_inches="tight")
    print("✅ Saved figure to dpu_rate_spacev100m.pdf")
    plt.close(fig)


def generate_sift_plot():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.5)) 

    # 绘图 (SIFT-100M)
    for key in keys_order:
        y_percent = [v * 100 for v in sift_dpu_rate[key]]
        ax.plot(recall_float, y_percent, 
                marker=markers.get(key, 'o'),
                linestyle='-',
                color=colors.get(key, 'gray'),
                linewidth=2,
                markeredgecolor='black',
                markersize=8,
                label=key,
                alpha=0.85)

    # === 坐标轴设置 ===
    ax.set_xticks(recall_float)
    ax.set_xticklabels(recall)
    ax.set_xlabel("Recall@100")

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    ax.set_ylabel("DPU Utilization (%)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    fig.savefig("dpu_rate_sift100m.pdf", bbox_inches="tight")
    print("✅ Saved figure to dpu_rate_sift100m.pdf")
    plt.close(fig)


def generate_plot_with_legend(dataset):
    # 创建图和轴
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))

    # 选择数据集进行绘制 (SIFT-100M / SPACE-100M)
    dpu_rate = sift_dpu_rate
    if (dataset == "spacev100m"):
        dpu_rate = space_dpu_rate
    
    for key in keys_order:  # 这里根据实际需要的key选择
        y_percent = [v * 100 for v in dpu_rate[key]]  # 将值转换为百分比
        ax.plot(recall_float, y_percent,
                marker=markers.get(key, 'o'),
                linestyle='-',
                color=colors.get(key, 'gray'),
                linewidth=2,
                markeredgecolor='black',
                markersize=8,
                label=key,
                alpha=0.85)

    # === 坐标轴设置 ===
    ax.set_xticks(recall_float)
    ax.set_xticklabels(recall)
    ax.set_xlabel("Recall@100")

    # Y轴设置为百分比
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    ax.set_ylabel("DPU Utilization (%)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # === 添加图例，放置在顶部 ===
    ax.legend(
        frameon=False,              # 移除图例的外框
        fontsize=12,                # 设置图例字体大小
        loc='upper center',        # 图例位置
        bbox_to_anchor=(0.5, 1.25),  # 在图的上方添加图例
        ncol=3,                     # 图例三列排列
        handlelength=2.5            # 调整图例符号的长度
    )

    # 调整布局，确保不会覆盖图例
    fig.tight_layout(rect=[0, 0, 1, 0.9])  # 保证图例不被截断

    # 保存图像
    file_name = "dpu_rate_sift100m_with_legend.pdf"
    if dataset == "spacev100m":
        file_name = "dpu_rate_spacev100m_with_legend.pdf"

    fig.savefig(file_name, bbox_inches="tight")
    print("✅ Saved figure to " + file_name)


# ===================================================
# 运行所有生成函数
# ===================================================
if __name__ == '__main__':
    # generate_legend()
    # generate_space_plot()
    # generate_sift_plot()
    generate_plot_with_legend("sift100m")
    generate_plot_with_legend("spacev100m")
