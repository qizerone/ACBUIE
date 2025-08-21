'''

auther: Chaoqun Yang
date: 2025.6.25
Work programs:
1.Read all picture in  folder
2.Mark the local area of the image for enlargement processing
3.Generate an image and output it to a folder

'''
import cv2
import os
import numpy as np
from typing import Tuple


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
    """绘制虚线"""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = np.sqrt(dx * dx + dy * dy)

    if length > 0:
        unit_dx, unit_dy = dx / length, dy / length
    else:
        return img  # 两点重合，无需绘制

    current_length, is_dash = 0, True
    while current_length < length:
        start_x = int(pt1[0] + current_length * unit_dx)
        start_y = int(pt1[1] + current_length * unit_dy)

        segment_length = dash_length if is_dash else gap_length
        end_x = int(start_x + segment_length * unit_dx)
        end_y = int(start_y + segment_length * unit_dy)

        if current_length + segment_length > length:
            end_x, end_y = pt2[0], pt2[1]

        if is_dash:
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

        current_length += segment_length
        is_dash = not is_dash

    return img


def process_images(folder_path, output_folder, roi_percent: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.2),
                   zoom_factor: float = 2.0, zoom_position: str = 'bottom_right'):
    """
    处理图像并分别保存原图标记结果和放大区域图片

    参数:
    folder_path: 输入图像文件夹路径
    output_folder: 输出文件夹路径（会自动创建子文件夹）
    roi_percent: 放大区域相对位置 (x%, y%, width%, height%)
    zoom_factor: 放大倍数
    zoom_position: 放大区域在原图中的显示位置
    """
    # 创建输出目录结构
    main_output = os.path.join(output_folder, "marked_images")  # 原图标记结果
    zoom_output = os.path.join(output_folder, "zoomed_roi")  # 放大区域单独保存
    for dir_path in [main_output, zoom_output]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 获取图像文件列表
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"[警告] {folder_path} 中未找到图像文件")
        return

    # 处理每张图像
    for filename in image_files:
        file_path = os.path.join(folder_path, filename)
        try:
            image = cv2.imread(file_path)
            if image is None:
                print(f"[错误] 无法读取图像: {filename}")
                continue

            height, width = image.shape[:2]
            # 计算ROI绝对坐标
            x = int(width * roi_percent[0])
            y = int(height * roi_percent[1])
            w = int(width * roi_percent[2])
            h = int(height * roi_percent[3])

            # 边界校验
            x, y = max(0, x), max(0, y)
            w, h = min(width - x, w), min(height - y, h)
            if w <= 0 or h <= 0:
                print(f"[跳过] ROI无效: {filename}")
                continue

            # 提取并放大ROI
            roi = image[y:y + h, x:x + w]
            zoomed_roi = cv2.resize(roi, None, fx=zoom_factor, fy=zoom_factor,
                                    interpolation=cv2.INTER_LINEAR)
            zoom_h, zoom_w = zoomed_roi.shape[:2]

            # 计算放大区域在原图中的位置
            if zoom_position == 'bottom_right':
                right_x = max(width - zoom_w, 0)
                right_y = max(height - zoom_h, 0)
            elif zoom_position == 'top_right':
                right_x = max(width - zoom_w, 0)
                right_y = 0
            elif zoom_position == 'bottom_left':
                right_x = 0
                right_y = max(height - zoom_h, 0)
            else:  # top_left
                right_x, right_y = 0, 0

            # 自适应调整放大区域尺寸（避免超出边界）
            max_fit_w = width - right_x
            max_fit_h = height - right_y
            if zoom_w > max_fit_w or zoom_h > max_fit_h:
                scale = min(max_fit_w / zoom_w, max_fit_h / zoom_h)
                if scale < 1.0:
                    new_size = (int(zoom_w * scale), int(zoom_h * scale))
                    zoomed_roi = cv2.resize(zoomed_roi, new_size)
                    zoom_h, zoom_w = zoomed_roi.shape[:2]
                    if zoom_position == 'bottom_right' or zoom_position == 'top_right':
                        right_x = width - zoom_w

            # 1. 保存原图标记结果
            result_img = image.copy()
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 标记原始ROI

            # 绘制虚线连接
            draw_dashed_line(result_img, (x, y), (right_x, right_y), (0, 0, 255), 2)
            draw_dashed_line(result_img, (x + w, y), (right_x + zoom_w, right_y), (0, 0, 255), 2)
            draw_dashed_line(result_img, (x, y + h), (right_x, right_y + zoom_h), (0, 0, 255), 2)
            draw_dashed_line(result_img, (x + w, y + h), (right_x + zoom_w, right_y + zoom_h), (0, 0, 255), 2)

            # 粘贴放大区域并添加边框
            result_img[right_y:right_y + zoom_h, right_x:right_x + zoom_w] = zoomed_roi
            cv2.rectangle(result_img, (right_x, right_y), (right_x + zoom_w, right_y + zoom_h), (0, 0, 255), 2)

            # 保存标记结果
            marked_path = os.path.join(main_output, f"marked_{filename}")
            cv2.imwrite(marked_path, result_img)

            # 2. 单独保存放大区域
            zoomed_path = os.path.join(zoom_output, f"zoomed_{filename}")
            cv2.imwrite(zoomed_path, zoomed_roi)

            print(f"[完成] {filename} → 标记图: {marked_path} | 放大图: {zoomed_path}")

        except Exception as e:
            print(f"[错误] 处理 {filename} 时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    INPUT_FOLDER = "example-FIGS/FIG-INPUT"  # 输入图像文件夹
    OUTPUT_FOLDER = "example-FIGS/FIG-VIEW"  # 总输出文件夹

    # 配置参数
    ROI_PARAMS = (0.20, 0.43, 0.2, 0.2)  # ROI起始位置(30%,30%)，大小占30%
    ZOOM_FACTOR = 2.2  # 放大倍数
    ZOOM_POSITION = 'bottom_right'  # 放大区域位置

    process_images(INPUT_FOLDER, OUTPUT_FOLDER, ROI_PARAMS, ZOOM_FACTOR, ZOOM_POSITION)