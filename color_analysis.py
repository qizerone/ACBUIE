'''

作者：杨超群
日期：2025年6月25日
工作流程：
1.遍历文件夹下的所有图片
2.分析文件夹中每个图片的颜色分布直方图
3.将颜色分布直方图存储到目标文件夹中

'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 解决中文显示问题：指定支持中文的字体（以 SimSun 为例，可根据系统调整）
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定字体，SimSun 是 Windows 系统的宋体，Linux/Mac 可尝试 'PingFang SC'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# 遍历文件夹下所有图片并处理的函数
def process_images(source_folder, target_folder, hist_range=(0, 256), y_scale=1.1, hist_bins=256):
    """
    处理图片文件夹并生成颜色直方图

    参数:
        source_folder: 源图片文件夹路径
        target_folder: 直方图保存文件夹路径
        hist_range: 直方图横轴范围，元组 (min, max)
        y_scale: 纵轴缩放因子，控制顶部留白比例
        hist_bins: 直方图的分箱数量
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 6))

            # 定义颜色列表（修复未定义的 colors 变量）
            colors = ('r', 'g', 'b')  # 对应 R、G、B 通道的颜色标识

            # 主图：绘制整体直方图曲线
            ax_main = plt.subplot(111)
            max_val = 0
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [hist_bins], hist_range)
                ax_main.plot(hist, color=color, label=color.upper())
                max_val = max(max_val, np.max(hist))

            # 设置纵轴范围和标签
            ax_main.set_ylim(0, max_val * y_scale)  # 纵轴缩放
            # ax_main.set_ylabel('像素数量', fontsize=10)
            # ax_main.set_xlabel('亮度值', fontsize=10)
            # ax_main.set_title('颜色分布直方图', fontsize=12)
            ax_main.legend(loc='upper left')  # 图例放在左上角，给右上角留出空间
            ax_main.grid(True, linestyle='--', alpha=0.7)

            # 在主图右上角内侧创建嵌入轴，放置通道像素数量柱状图
            ax_inset = plt.axes([0.65, 0.65, 0.3, 0.3], facecolor='white')  # 位置和大小可调整

            # 计算各通道像素总量
            hist_b = cv2.calcHist([img], [0], None, [hist_bins], hist_range)
            hist_g = cv2.calcHist([img], [1], None, [hist_bins], hist_range)
            hist_r = cv2.calcHist([img], [2], None, [hist_bins], hist_range)
            sum_b = np.sum(hist_b)
            sum_g = np.sum(hist_g)
            sum_r = np.sum(hist_r)

            # 绘制通道像素数量柱状图
            ax_inset.bar(['B', 'G', 'R'], [sum_b, sum_g, sum_r], color=['b', 'g', 'r'], alpha=0.8)
            # ax_inset.set_title('各通道像素数量', fontsize=9)
            ax_inset.set_xticks([0, 1, 2])  # 设置 x 轴刻度位置
            ax_inset.set_xticklabels(['B', 'G', 'R'], fontsize=8)  # 设置 x 轴刻度标签
            ax_inset.tick_params(axis='y', labelsize=8)  # 调整 y 轴刻度字体大小

            # **关键修改：移除 tight_layout()，改用 subplots_adjust 手动调整边距**
            plt.subplots_adjust(top=0.95, right=0.95, left=0.1, bottom=0.1)  # 调整边距避免裁剪

            # 保存直方图
            save_path = os.path.join(target_folder, f"{os.path.splitext(filename)[0]}_hist.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存时强制紧凑布局
            plt.close()


# 替换为实际路径
source_folder = 'example-FIGS/FIG-VIEW/zoomed_roi'
target_folder = 'example-FIGS/FIG-COLOR'

# 示例：调整横轴范围为 50-200，纵轴留出 20% 的空白，使用 128 个分箱
process_images(
    source_folder,
    target_folder,
    hist_range=(50, 200),  # 调整横轴范围，聚焦于感兴趣的亮度区域
    y_scale=2,  # 纵轴缩放因子，值越大顶部留白越多
    hist_bins=50  # 减少分箱数量使曲线更平滑
)