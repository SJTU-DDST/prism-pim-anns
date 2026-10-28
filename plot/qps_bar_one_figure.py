import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ====== 数据定义 ======
recall = [0.84, 0.86, 0.88, 0.90, 0.92, 0.94]

# Sift100M 数据
sift = {
    "Prism":    [1834, 1823, 1839, 1718, 1508, 1250],
    "PIMANN":   [674,  568,  511,  445,  362,  290],
    "DRIM-ANN": [206,  194,  188,  174,  157,  141],
    "Faiss-CPU":[141,  122,  107,  89,   80,   63]
}

# SpaceV100M 数据
space = {
    "Prism":    [1812, 1507, 1196, 939, 694, 472],
    "PIMANN":   [650,  474,  377,  283, 190, 117],
    "DRIM-ANN": [291,  264,  233,  206, 163, 120],
    "Faiss-CPU":[100,   79,   63,   48,  36,  27]
}

# ====== 颜色定义 ======
colors = {
    "Prism": "#b7e3a8",       # 绿色
    "PIMANN": "#a9c1e6",      # 蓝色
    "DRIM-ANN": "#c6b3e0",  # 紫色
    "Faiss-CPU": "#f4a7a1"        # 粉红
}
keys_order = ["Prism", "PIMANN", "DRIM-ANN", "Faiss-CPU"]

# 统一的 Y 轴上限
Y_LIM_MAX = 2000 

# ====== 全局字体与样式（Arial） ======
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"]
})

x = np.arange(len(recall))
width = 0.19

# ===================================================
# 1. 创建统一的 Figure 和 1x2 子图 (水平并排)
# ===================================================
# 创建一个 Figure，包含 1 行 2 列的子图 (Axes)
# figsize=(宽度, 高度), sharey=True 确保 Y 轴统一
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 2.75), sharey=True) 

# ===================================================
# 2. 绘制 SIFT-100M (左侧, ax1)
# ===================================================
for i, key in enumerate(keys_order):
    ax1.bar(x + (i - 1.5) * width, sift[key], width,
            color=colors[key], edgecolor="black", zorder=2)

# 标注 (a) Sift100M
ax1.text(0.98, 0.95, '(a) Sift100M', transform=ax1.transAxes, 
         ha='right', va='top', fontsize=13, fontweight='bold')

# 设置坐标轴 (左侧图)
ax1.set_xticks(x)
ax1.set_xticklabels(recall)
ax1.set_xlabel("Recall@100")
ax1.set_yticks(np.arange(0, Y_LIM_MAX + 1, 400))
ax1.set_ylabel("Throughput (QPS)") # 左侧图保留 Y 轴标签
ax1.set_ylim(0, Y_LIM_MAX)
ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# ===================================================
# 3. 绘制 SPACE-100M (右侧, ax2)
# ===================================================
for i, key in enumerate(keys_order):
    ax2.bar(x + (i - 1.5) * width, space[key], width,
            color=colors[key], edgecolor="black", zorder=2)

# 标注 (b) SpaceV100M
ax2.text(0.98, 0.95, '(b) SpaceV100M', transform=ax2.transAxes, 
         ha='right', va='top', fontsize=13, fontweight='bold')

# 设置坐标轴 (右侧图)
ax2.set_xticks(x)
ax2.set_xticklabels(recall)
ax2.set_xlabel("Recall@100")
ax2.set_ylabel("Throughput (QPS)") # 可选添加 Y 轴标签 (因为 sharey=True)
# Y 轴刻度/范围已通过 sharey=True 自动同步
ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
ax2.tick_params(axis='y', labelleft=True)  # 右侧子图也显示坐标轴

# ===================================================
# 4. 生成并放置共享图例 (Figure 顶部)
# ===================================================
# 构造图例句柄
legend_handles = [Patch(facecolor=colors[k], edgecolor="black", label=k) for k in keys_order]

# 将图例添加到 Figure 级别, 放置在顶部中央
fig.legend(
    handles=legend_handles, 
    ncol=4, 
    loc="upper center", 
    frameon=False,
    bbox_to_anchor=(0.48, 0.97),  # 调整锚点
    fontsize=13  # 增大图例字体
)

# ===================================================
# 5. 调整布局并保存
# ===================================================
# 自动调整布局，并为顶部的图例留出 9% 的空间
fig.tight_layout(rect=[0, 0, 1, 0.92]) 

# 保存为 PDF (设置 pad_inches 裁剪四周留白，越小则留白越小，默认 0.10)
fig.savefig("qps_combined.pdf", 
            bbox_inches="tight",
            pad_inches=0.07)

print("✅ Saved combined figure to qps_combined.pdf")
plt.close(fig)