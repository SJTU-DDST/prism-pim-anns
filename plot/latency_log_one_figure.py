import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D # 导入 Line2D 用于折线图图例
import matplotlib.ticker as mticker # 导入 mticker 用于刻度格式化

# ====== 数据定义 ======
recall = [0.84, 0.86, 0.88, 0.90, 0.92, 0.94]

# Sift100M 延迟数据 (Latency)
sift_latency = {
    "Prism":    [329,  330,  326,  274,  314,  372],
    "PIMANN":   [499,  558,  580,  640,  763,  858], 
    "DRIM-ANN": [1132, 1209, 1245, 1349, 1485, 1655],
    "Faiss-CPU":[944,  1085, 1246, 1496, 1654, 2112]
}

# SpaceV100M 延迟数据 (Latency)
space_latency = {
    "Prism":    [262,  303,  369,  457,  597,  852],
    "PIMANN":   [496,  611,  683,  860,  1215, 1735], 
    "DRIM-ANN": [803,  885,  999,  1127, 1420, 1928],
    "Faiss-CPU":[1327, 1678, 2097, 2782, 3688, 4906]
}

# ====== 颜色定义 ======
colors = {
    "Prism": "#b7e3a8",       # 绿色
    "PIMANN": "#a9c1e6",     # 蓝色 
    "DRIM-ANN": "#c6b3e0",   # 紫色 
    "Faiss-CPU": "#f4a7a1"   # 粉红
}

# ====== 标记定义 (新增) ======
markers = {
    "Prism": 'o',      # 圆圈
    "PIMANN": 's',     # 方块
    "DRIM-ANN": 'D',   # 菱形
    "Faiss-CPU": '^'   # 三角形
}

# 根据新的数据集调整keys_order
keys_order = ["Prism", "PIMANN", "DRIM-ANN", "Faiss-CPU"]

# ====== Y 轴限制 (对数刻度下的独立限制) ======
Y_MIN = 200
Y_MAX_SIFT = 3000
Y_MAX_SPACE = 7000

LOG_TICKS_COMMON_SIFT = [500, 1000, 2000]
LOG_TICKS_COMMON_SPACE = [500, 1000, 2000, 5000]

# ====== 全局字体与样式（Arial） ======
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12, # 全局图例字体 (我们将在下面覆盖)
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"]
})

# 将 recall 转换为浮点数列表作为 x 轴坐标
recall_float = [float(r) for r in recall] 


# 新增：Alpha 混合辅助函数
def hex_to_rgb(h):
    """将 6 位 Hex 颜色转换为 (R, G, B) 元组"""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """将 (R, G, B) 元组转换为 Hex 颜色"""
    # 确保 RGB 值在 0-255 范围内并取整
    rgb_int = (int(max(0, min(c, 255))) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*rgb_int)

def blend_color(hex_color, alpha, background_hex='#FFFFFF'):
    """计算颜色与白色背景混合 0.75 Alpha 后的扁平化颜色"""
    # 转换为 (R, G, B)
    rgb_color = hex_to_rgb(hex_color)
    rgb_background = hex_to_rgb(background_hex)
    
    # 应用 Alpha 混合公式: New = (Color * Alpha) + (BG * (1 - Alpha))
    blended_rgb = []
    for i in range(3):
        blended_channel = (rgb_color[i] * alpha) + (rgb_background[i] * (1 - alpha))
        blended_rgb.append(blended_channel)
        
    # 转换回 Hex
    return rgb_to_hex(blended_rgb)

# 计算 0.75 Alpha "扁平化" 后的颜色字典
alpha_value = 0.75
flattened_colors = {
    key: blend_color(value, alpha_value) for key, value in colors.items()
}
# (例如: '#b7e3a8' -> '#cae9bf', '#a9c1e6' -> '#bfcde9')


# ===================================================
# 1. 创建统一的 Figure 和 1x2 子图 (水平并排)
# ===================================================
#
# 使用您为 QPS 图选择的 figsize (11, 2.75)
# *** 注意：此处不使用 sharey=True，因为 Y 轴刻度和范围不同 ***
#
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 2.75)) 


# ===================================================
# 2. 绘制 SIFT-100M (左侧, ax1)
# ===================================================
for key in keys_order:
    ax1.plot(recall_float, sift_latency[key], 
             marker=markers[key], 
             linestyle='-', 
             color=colors[key], 
             linewidth=2,
             markeredgecolor='black', # 标记边框
             markersize=8,
             label=key,
             alpha=0.75)
    
# 标注 (a) Sift100M
# (注意：原代码中 Sift 图用的是左上角标注)
ax1.text(0.02, 0.95, '(a) Sift100M', transform=ax1.transAxes, 
         ha='left', va='top', fontsize=13, fontweight='bold')

# 设置 Y 轴为对数刻度
ax1.set_yscale('log')

# 设置坐标轴 (左侧图)
ax1.set_xticks(recall_float)
ax1.set_xticklabels(recall)
ax1.set_xlabel("Recall@100")
ax1.set_ylim(Y_MIN, Y_MAX_SIFT)
sift_ticks = [Y_MIN] + LOG_TICKS_COMMON_SIFT + [Y_MAX_SIFT]
ax1.set_yticks(sift_ticks) 
ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
ax1.set_ylabel("Latency (ms)") # 保留 Y 轴标签
ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)


# ===================================================
# 3. 绘制 SPACE-100M (右侧, ax2)
# ===================================================
for key in keys_order:
    ax2.plot(recall_float, space_latency[key], 
             marker=markers[key], 
             linestyle='-', 
             color=colors[key], 
             linewidth=2,
             markeredgecolor='black', # 标记边框
             markersize=8,
             label=key,
             alpha=0.75)
    
# 标注 (b) SpaceV100M
# (注意：原代码中 Space 图用的是左上角标注)
ax2.text(0.02, 0.95, '(b) SpaceV100M', transform=ax2.transAxes, 
         ha='left', va='top', fontsize=13, fontweight='bold')
         
# 设置 Y 轴为对数刻度
ax2.set_yscale('log')

# 设置坐标轴 (右侧图)
ax2.set_xticks(recall_float)
ax2.set_xticklabels(recall)
ax2.set_xlabel("Recall@100")
ax2.set_ylim(Y_MIN, Y_MAX_SPACE)
space_ticks = [Y_MIN] + LOG_TICKS_COMMON_SPACE + [Y_MAX_SPACE]
ax2.set_yticks(space_ticks) 
ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
ax2.set_ylabel("Latency (ms)") # 保留 Y 轴标签
ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)


# ===================================================
# 4. 生成并放置共享图例 (Figure 顶部)
# ===================================================
# ▼▼▼ 修改图例句柄 (按照您的 Alpha 预混合方案) ▼▼▼
legend_handles = [
    Line2D([0], [0], 
           marker=markers[k], 
           color='black',                     # 保持黑线
           markerfacecolor=flattened_colors[k], # <-- 使用新计算的【扁平化】颜色
           markeredgecolor='black',           # 标记边框色
           markersize=8, 
           linewidth=2,
           label=k,
           linestyle='-',                     # 保持线条样式
           alpha=1.0)                         # <-- 设为 1.0 (完全不透明)
    for k in keys_order
]

# 使用 QPS 图指定的图例设置
fig.legend(
    handles=legend_handles, 
    ncol=4, 
    loc="upper center", 
    frameon=False,
    bbox_to_anchor=(0.48, 0.94),  # 保持您设定的锚点
    fontsize=13                   # 保持您设定的字体大小
)

# ===================================================
# 5. 调整布局并保存
# ===================================================
# 使用 QPS 图指定的布局调整 (为图例留出空间) 
fig.tight_layout(rect=[0, 0, 1, 0.88]) 

# 保存为 PDF
fig.savefig("latency_combined.pdf",             
            bbox_inches="tight",
            pad_inches=0.07)

print("✅ Saved combined figure to latency_combined.pdf")
plt.close(fig)