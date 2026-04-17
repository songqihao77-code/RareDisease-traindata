import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

# =========================
# 1. 中文字体与显示设置
# =========================
# 按顺序尝试常见中文字体，谁可用就用谁
rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


def add_box(ax, x, y, w, h, text, fontsize=11,
            fc="#eef3f8", ec="#4f79a7", lw=1.5, round_size=0.08):
    """添加圆角文本框"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={round_size}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True
    )


def add_arrow(ax, x1, y1, x2, y2, lw=1.6, color="#4f79a7"):
    """添加箭头"""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0
        )
    )


# =========================
# 图1：多传感器融合层级与信息流关系示意图
# =========================
def draw_figure1(save_path="figure1_sensor_fusion_layers.png"):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.95,
        "多传感器融合层级与信息流关系示意图",
        ha="center", va="center",
        fontsize=18, fontweight="bold", color="#355c85"
    )

    # 顶部传感器输入
    top_y = 0.82
    box_w = 0.16
    box_h = 0.06
    xs = [0.03, 0.26, 0.49, 0.72]
    labels = ["相机图像", "LiDAR点云", "毫米波雷达点迹", "IMU / GNSS"]

    for x, label in zip(xs, labels):
        add_box(ax, x, top_y, box_w, box_h, label, fontsize=11)

    # 数据层融合
    add_box(
        ax, 0.23, 0.62, 0.54, 0.08,
        "数据层融合\n原始观测联合处理\n信息保留充分，但时间与标定敏感",
        fontsize=11, fc="#f3f5f7"
    )

    # 特征层融合
    add_box(
        ax, 0.23, 0.40, 0.54, 0.08,
        "特征层融合\n中间表示交互\n兼顾信息利用率与建模灵活性",
        fontsize=11, fc="#f3f5f7"
    )

    # 决策层融合
    add_box(
        ax, 0.23, 0.18, 0.54, 0.08,
        "决策层融合\n结果级整合\n结构清晰，但深层交互不足",
        fontsize=11, fc="#f3f5f7"
    )

    # 底部输出
    ax.text(
        0.5, 0.08,
        "目标检测 / 语义分割 / 多目标跟踪 / 定位导航",
        ha="center", va="center",
        fontsize=13
    )

    # 传感器到数据层的连线
    for x in xs:
        center_x = x + box_w / 2
        add_arrow(ax, center_x, top_y, center_x, 0.70)
    # 汇入中心
    add_arrow(ax, 0.50, 0.62, 0.50, 0.48)
    add_arrow(ax, 0.50, 0.40, 0.50, 0.26)
    add_arrow(ax, 0.50, 0.18, 0.50, 0.10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 图2：基于深度学习的多传感器融合一般框架
# =========================
def draw_figure2(save_path="figure2_deep_fusion_framework.png"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.96,
        "基于深度学习的多传感器融合一般框架",
        ha="center", va="center",
        fontsize=18, fontweight="bold", color="#355c85"
    )

    ax.text(
        0.5, 0.90,
        "前提条件：时间同步  ·  外参标定  ·  坐标统一  ·  数据质量控制",
        ha="center", va="center",
        fontsize=12, color="#7a6a2f"
    )

    # 输入模态
    input_y = 0.78
    add_box(ax, 0.07, input_y, 0.16, 0.065, "多视角相机图像", fontsize=11)
    add_box(ax, 0.32, input_y, 0.14, 0.065, "LiDAR点云", fontsize=11)
    add_box(ax, 0.57, input_y, 0.16, 0.065, "毫米波雷达点迹", fontsize=11)

    # 编码器
    enc_y = 0.63
    add_box(ax, 0.05, enc_y, 0.20, 0.075, "视觉主干网络\n(CNN / ViT)", fontsize=11, fc="#f7f7f7")
    add_box(ax, 0.30, enc_y, 0.18, 0.075, "点云编码器\n(Point / Voxel / SparseConv)", fontsize=10.5, fc="#f7f7f7")
    add_box(ax, 0.55, enc_y, 0.20, 0.075, "雷达编码器\n(Point / BEV)", fontsize=11, fc="#f7f7f7")

    # 跨模态模块
    add_box(
        ax, 0.25, 0.40, 0.50, 0.12,
        "跨模态对齐与交互模块\n"
        "投影映射  ·  共享BEV表示  ·  跨模态注意力  ·  Transformer查询机制\n"
        "目标：在统一空间中建立几何、语义与运动信息的有效融合",
        fontsize=11,
        fc="#edf5ea",
        ec="#5d8a5a"
    )

    # 共享表示
    ax.text(
        0.5, 0.31,
        "共享表示：BEV特征 / Query Tokens / 多模态特征图",
        ha="center", va="center",
        fontsize=12
    )

    # 输出任务
    out_y = 0.12
    add_box(ax, 0.18, out_y, 0.15, 0.065, "3D目标检测", fontsize=11)
    add_box(ax, 0.42, out_y, 0.16, 0.065, "语义 / 全景分割", fontsize=11)
    add_box(ax, 0.67, out_y, 0.15, 0.065, "多目标跟踪", fontsize=11)

    # 连接线
    add_arrow(ax, 0.15, input_y, 0.15, enc_y + 0.075)
    add_arrow(ax, 0.39, input_y, 0.39, enc_y + 0.075)
    add_arrow(ax, 0.65, input_y, 0.65, enc_y + 0.075)

    add_arrow(ax, 0.15, enc_y, 0.35, 0.52)
    add_arrow(ax, 0.39, enc_y, 0.50, 0.52)
    add_arrow(ax, 0.65, enc_y, 0.65, 0.52)

    add_arrow(ax, 0.50, 0.40, 0.50, 0.33)

    add_arrow(ax, 0.50, 0.28, 0.255, out_y + 0.065)
    add_arrow(ax, 0.50, 0.28, 0.50,  out_y + 0.065)
    add_arrow(ax, 0.50, 0.28, 0.745, out_y + 0.065)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_figure1()
    draw_figure2()
    print("已生成：")
    print(" - figure1_sensor_fusion_layers.png")
    print(" - figure2_deep_fusion_framework.png")