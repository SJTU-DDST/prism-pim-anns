import matplotlib.pyplot as plt


dataset = "spacev100m"  # 可选 spacev100m, sift100m

# === 全局字体和样式设置 ===
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def read_dpu_counts(file_path, num_lines):
    """读取文件前 num_lines 行整数数据"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                try:
                    # 1. 读取原始整数
                    val = int(line.strip())
                    # 2. 应用变换: 乘40除39，并四舍五入保留整数，扩展到40个rank
                    transformed_val = round(val * 40 / 39)
                    data.append(transformed_val)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Error: file not found -> {file_path}")
    return data

def plot_multi_dpu_counts(file_paths, labels, colors, linestyles, interval_ms, num_lines, out_file):
    """绘制多条 DPU 活跃数曲线"""
    plt.figure(figsize=(11, 3.2))  # 对于 IEEE 模板，可采用(7, 3)；对于 ACM 模板，可采用 (11, 2.75)

    # === X 轴时间序列（自动根据参数计算） ===
    time_points = [i * interval_ms / 1000.0 for i in range(num_lines)]  # 秒
    total_time = num_lines * interval_ms / 1000.0

    # === 绘制每条曲线 ===
    for path, label, style in zip(file_paths, labels, linestyles):
        y = read_dpu_counts(path, num_lines)
        plt.plot(
            time_points, y,
            label=label,
            color=colors[label],
            linestyle=style,
            linewidth=2
        )

    # === 坐标轴与网格 ===
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("# of working DPUs", fontsize=14)
    plt.xlim(0, total_time)
    # plt.ylim(bottom=0)
    # plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

    max_dpu = 2560  # 虚线位置
    
    # 1. 绘制辅助线 (zorder=0 使其位于网格之上和数据之上)
    plt.axhline(y=max_dpu, color='black', linestyle='--', linewidth=1.8, zorder=0, alpha=0.6)
    
    ticks = [0, 500, 1000, 1500, 2000, 2500]  # 设置 y 轴刻度值
    plt.yticks(ticks, fontsize=13)
    
    # 3. 将 Y 轴上限设置为 MAX 上方一点点，以确保辅助线可见
    plt.ylim(bottom=0, top=max_dpu * 1.05) # 在 MAX 上方留 8% 的 padding
    #
    # ▲▲▲ 修改结束 ▲▲▲
    #

    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

    # === 图例（放在图外上方） ===
    plt.legend(
        frameon=False,
        # frameon=True,            # <-- 1. 启用图例框
        # edgecolor='#CCCCCC',     # <-- 2. 设置边框颜色为浅灰色
        fontsize=14,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,                 # 三列水平排
        handlelength=2.4
    )

    # === 布局与保存 ===
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # 给顶部留出图例空间
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0.07)
    print(f"✅ Figure saved to: {out_file}")
    plt.show()

# ====== 文件路径与参数，默认取 Recall@100 = 0.90 的情况 ======
file_paths = [
    "sift_QM3_C18_BS128_DTYPE1.txt",  # Prism
    "sift-dpu-active-num-nprobe13.txt",      # PIMANN
    "sift-batch-dpu-active-num-nprobe13.txt" # DRIM-ANN
]

if dataset == "spacev100m":
    file_paths = [
        "spacev_QM3_C36_BS128_DTYPE1.txt",  # Prism
        "spacev-dpu-active-num-nprobe25.txt",      # PIMANN
        "spacev-batch-dpu-active-num-nprobe25.txt" # DRIM-ANN
    ]

labels = ["Prism", "PIMANN", "DRIM-ANN"]

# ====== 改进后的颜色定义 ======
colors = {
    "Prism": "#6BAF92",       # 深绿色（稳重、可读）
    "PIMANN": "#5C8AC6",     # 深蓝色（科技感强）
    "DRIM-ANN": "#8F79C7",   # 紫灰色（柔和对比）
}


# ====== 线条样式 ======
linestyles = ["-", "-.", "--"]

# ====== 参数 ======
interval_ms = 2
num_lines = 1212   # 自动调整横坐标为 interval_ms * num_lines
out_file = "nb_running_dpu_" + dataset + ".pdf"

# ====== 执行绘图 ======
if __name__ == "__main__":
    plot_multi_dpu_counts(file_paths, labels, colors, linestyles, interval_ms, num_lines, out_file)
