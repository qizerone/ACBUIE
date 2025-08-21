'''

作者：杨超群
日期：2025年6月25日
工作流程
1.读取“images_orinal”文件夹下的所有原图片。
2.读取“images_enhance”文件夹下对应“images_orinal”对应名称的强化图片。
3.计算每张图片的PSNR、SSIM。
4.将每张图片对应的PSNR、SSIM输出到指定的“PSresults.xlsx”文件当中。
存储格式为：    图片名称       PSNR       SSIM

5.计算“images_enhance”文件夹下每张图片的UIQE、UCIQE。
6.将每张图片对应的UIQE、UCIQE输出到指定的“UUresults.xlsx”文件当中
存储格式为：    图片名称       UIQE       UCIQE

7.将“images_enhance”文件夹下的图片对应的PSNR、SSIM进行t检验
8.将t检验得出的置信区间结果输出到指定的“TPSresults.xlsx”文件当中
存储格式为：    文件夹名称       T-data

9.将“images_enhance”文件夹下的图片对应的UIQE、UCIQE进行t检验
10.将t检验得出的置信区间结果输出到指定的“Tuuresults.xlsx”文件当中
存储格式为：    文件夹名称       T-data
'''
import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy import stats
import xlsxwriter
import re
from pathlib import Path


def find_matching_image(original_name, enhanced_images):
    """
    根据原始图像名称在增强图像列表中查找最佳匹配的图像
    严格匹配规则：
    1. 提取文件名中的数字部分必须完全一致
    2. 文件名结构相似（如1.jpg匹配1_enhanced.png）
    """
    # 提取原始文件名中的数字
    original_numbers = re.findall(r'\d+', original_name)
    if not original_numbers:
        return None  # 没有数字无法匹配

    original_num = original_numbers[0]  # 只取第一个连续数字

    # 收集所有可能的匹配项
    matches = []

    for enhanced_name in enhanced_images:
        # 提取增强文件名中的数字
        enhanced_numbers = re.findall(r'\d+', enhanced_name)
        if not enhanced_numbers:
            continue

        enhanced_num = enhanced_numbers[0]

        # 数字必须完全一致
        if original_num != enhanced_num:
            continue

        # 计算文件名相似度（惩罚带额外数字的情况，如100匹配10）
        score = 100 - 10 * abs(len(original_numbers) - len(enhanced_numbers))

        # 优先匹配命名模式相似的（如1.jpg和1_enhanced.png）
        original_base = os.path.splitext(original_name)[0]
        enhanced_base = os.path.splitext(enhanced_name)[0]

        if original_base in enhanced_base or enhanced_base in original_base:
            score += 50
        elif "_" in enhanced_name and "_" in original_name:
            if enhanced_name.split("_")[0] == original_name.split("_")[0]:
                score += 30

        matches.append((enhanced_name, score))

    if not matches:
        return None

    # 选择最高分的匹配项
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0]

def calculate_psnr_ssim(original_folder, enhanced_folder, psnr_ssim_output_file):
    """计算原始图像和增强图像之间的PSNR和SSIM指标"""
    # 创建结果存储列表
    results = []

    # 获取原始图像文件夹中的所有图像文件
    original_images = [f for f in os.listdir(original_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    # 获取增强图像文件夹中的所有图像文件
    enhanced_images = [f for f in os.listdir(enhanced_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    # 处理每张图像
    for img_name in original_images:
        try:
            # 查找匹配的增强图像
            matched_enhanced = find_matching_image(img_name, enhanced_images)

            if not matched_enhanced:
                print(f"警告: 找不到对应的增强图像 {img_name}")
                continue

            # 读取原始图像和增强图像
            original_img_path = os.path.join(original_folder, img_name)
            enhanced_img_path = os.path.join(enhanced_folder, matched_enhanced)

            # 读取图像
            original_img = cv2.imread(original_img_path)
            enhanced_img = cv2.imread(enhanced_img_path)

            if original_img is None:
                print(f"警告: 无法读取原始图像 {img_name}")
                continue

            if enhanced_img is None:
                print(f"警告: 无法读取增强图像 {matched_enhanced}")
                continue

            # 确保图像尺寸一致
            if original_img.shape != enhanced_img.shape:
                print(
                    f"注意: 图像尺寸不一致 {img_name} (原始: {original_img.shape}) vs {matched_enhanced} (增强: {enhanced_img.shape}) - 正在调整...")
                enhanced_img = cv2.resize(enhanced_img, (original_img.shape[1], original_img.shape[0]))

            # 计算PSNR和SSIM
            psnr_val = psnr(original_img, enhanced_img, data_range=255)

            # 对于SSIM，我们需要分别计算每个通道，然后取平均值
            ssim_vals = []
            for channel in range(original_img.shape[2]):
                ssim_val = ssim(original_img[:, :, channel],
                                enhanced_img[:, :, channel],
                                data_range=255,
                                win_size=3)  # 使用较小的窗口大小以提高性能
                ssim_vals.append(ssim_val)
            ssim_val = np.mean(ssim_vals)

            # 存储结果
            results.append([img_name, matched_enhanced, psnr_val, ssim_val])

        except Exception as e:
            print(f"处理图像 {img_name} 时出错: {str(e)}")

    # 将结果写入Excel文件
    if results:
        df = pd.DataFrame(results, columns=['原始图片名称', '增强图片名称', 'PSNR', 'SSIM'])
        df.to_excel(psnr_ssim_output_file, index=False, engine='openpyxl')
        print(f"PSNR和SSIM结果已保存到 {psnr_ssim_output_file}")
    else:
        print("没有PSNR和SSIM结果可保存")

    return results


def calculate_uciqe(image):
    """计算UCIQE指标"""
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 计算色度的标准差
    sigma_h = np.std(h)

    # 计算饱和度的均值和标准差
    mu_s = np.mean(s)
    sigma_s = np.std(s)

    # 计算亮度的对比度
    v = v.astype(np.float32)
    top_10_percent = np.percentile(v, 90)
    bot_10_percent = np.percentile(v, 10)
    con_l = (top_10_percent - bot_10_percent) / (top_10_percent + bot_10_percent + 1e-6)  # 添加小常数避免除以零

    # 计算UCIQE
    uciqe = 0.4680 * sigma_h + 0.2745 * mu_s + 0.2576 * con_l
    return uciqe


def calculate_uiqe(image):
    """计算UIQE指标"""
    # 转换到YIQ色彩空间
    yiq = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = yiq[:, :, 0].astype(np.float32) / 255.0

    # 计算对比度
    top_10_percent = np.percentile(y, 90)
    bot_10_percent = np.percentile(y, 10)
    con_l = (top_10_percent - bot_10_percent) / (top_10_percent + bot_10_percent + 1e-6)

    # 计算清晰度
    dy, dx = np.gradient(y)
    grad_mag = np.sqrt(dx ** 2 + dy ** 2)
    avg_grad = np.mean(grad_mag)

    # 计算色度
    rg = image[:, :, 2] - image[:, :, 1]  # R - G
    yb = 0.5 * (image[:, :, 2] + image[:, :, 1]) - image[:, :, 0]  # (R + G)/2 - B

    sigma_rg = np.std(rg)
    sigma_yb = np.std(yb)
    std_chr = np.sqrt(sigma_rg ** 2 + sigma_yb ** 2)

    # 计算UIQE
    uiqe = 0.0282 * con_l + 0.2953 * avg_grad + 0.6765 * std_chr
    return uiqe


def calculate_uiqe_uciqe(enhanced_folder, uiqe_uciqe_output_file):
    """计算增强图像的UIQE和UCIQE指标"""
    # 创建结果存储列表
    results = []

    # 获取增强图像文件夹中的所有图像文件
    enhanced_images = [f for f in os.listdir(enhanced_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    # 处理每张图像
    for img_name in enhanced_images:
        try:
            # 读取增强图像
            enhanced_img_path = os.path.join(enhanced_folder, img_name)
            enhanced_img = cv2.imread(enhanced_img_path)

            if enhanced_img is None:
                print(f"警告: 无法读取增强图像 {img_name}")
                continue

            # 计算UIQE和UCIQE
            uiqe_val = calculate_uiqe(enhanced_img)
            uciqe_val = calculate_uciqe(enhanced_img)

            # 存储结果
            results.append([img_name, uiqe_val, uciqe_val])

        except Exception as e:
            print(f"处理图像 {img_name} 时出错: {str(e)}")

    # 将结果写入Excel文件
    if results:
        df = pd.DataFrame(results, columns=['图片名称', 'UIQE', 'UCIQE'])
        df.to_excel(uiqe_uciqe_output_file, index=False, engine='openpyxl')
        print(f"UIQE和UCIQE结果已保存到 {uiqe_uciqe_output_file}")
    else:
        print("没有UIQE和UCIQE结果可保存")

    return results


def perform_t_test(psnr_ssim_results, t_test_output_file):
    """对PSNR和SSIM结果进行t检验"""
    if not psnr_ssim_results or len(psnr_ssim_results) < 2:
        print("没有足够的PSNR和SSIM结果可用于t检验")
        return

    # 提取PSNR和SSIM值
    psnr_values = [result[2] for result in psnr_ssim_results]
    ssim_values = [result[3] for result in psnr_ssim_results]

    # 执行t检验
    psnr_t_stat, psnr_p_value = stats.ttest_1samp(psnr_values, popmean=0)
    ssim_t_stat, ssim_p_value = stats.ttest_1samp(ssim_values, popmean=0)

    # 计算置信区间
    psnr_ci = stats.t.interval(0.95, len(psnr_values) - 1,
                               loc=np.mean(psnr_values),
                               scale=stats.sem(psnr_values))

    ssim_ci = stats.t.interval(0.95, len(ssim_values) - 1,
                               loc=np.mean(ssim_values),
                               scale=stats.sem(ssim_values))

    # 准备结果
    results = [
        ["PSNR", psnr_t_stat, psnr_p_value, np.mean(psnr_values),
         np.mean(psnr_values) - psnr_ci[0], psnr_ci[1] - np.mean(psnr_values)],
        ["SSIM", ssim_t_stat, ssim_p_value, np.mean(ssim_values),
         np.mean(ssim_values) - ssim_ci[0], ssim_ci[1] - np.mean(ssim_values)]
    ]

    # 将结果写入Excel文件
    with xlsxwriter.Workbook(t_test_output_file) as workbook:
        worksheet = workbook.add_worksheet()

        # 写入表头
        headers = ["指标", "T值", "P值", "均值", "-置信区间下限", "+置信区间上限"]
        for col_num, header in enumerate(headers):
            worksheet.write(0, col_num, header)

        # 写入数据
        for row_num, row_data in enumerate(results, 1):
            for col_num, cell_data in enumerate(row_data):
                worksheet.write(row_num, col_num, cell_data)

    print(f"t检验结果已保存到 {t_test_output_file}")
    return results


def perform_uiqe_uciqe_t_test(uiqe_uciqe_results, t_test_output_file):
    """对UIQE和UCIQE结果进行t检验"""
    if not uiqe_uciqe_results or len(uiqe_uciqe_results) < 2:
        print("没有足够的UIQE和UCIQE结果可用于t检验")
        return

    # 提取UIQE和UCIQE值
    uiqe_values = [result[1] for result in uiqe_uciqe_results]
    uciqe_values = [result[2] for result in uiqe_uciqe_results]

    # 执行t检验
    uiqe_t_stat, uiqe_p_value = stats.ttest_1samp(uiqe_values, popmean=0)
    uciqe_t_stat, uciqe_p_value = stats.ttest_1samp(uciqe_values, popmean=0)

    # 计算置信区间
    uiqe_ci = stats.t.interval(0.95, len(uiqe_values) - 1,
                               loc=np.mean(uiqe_values),
                               scale=stats.sem(uiqe_values))

    uciqe_ci = stats.t.interval(0.95, len(uciqe_values) - 1,
                                loc=np.mean(uciqe_values),
                                scale=stats.sem(uciqe_values))

    # 准备结果
    results = [
        ["UIQE", uiqe_t_stat, uiqe_p_value, np.mean(uiqe_values),
         np.mean(uiqe_values) - uiqe_ci[0], uiqe_ci[1] - np.mean(uiqe_values)],
        ["UCIQE", uciqe_t_stat, uciqe_p_value, np.mean(uciqe_values),
         np.mean(uciqe_values) - uciqe_ci[0], uciqe_ci[1] - np.mean(uciqe_values)]
    ]

    # 将结果写入Excel文件
    with xlsxwriter.Workbook(t_test_output_file) as workbook:
        worksheet = workbook.add_worksheet()

        # 写入表头
        headers = ["指标", "T值", "P值", "均值", "-置信区间下限", "+置信区间上限"]
        for col_num, header in enumerate(headers):
            worksheet.write(0, col_num, header)

        # 写入数据
        for row_num, row_data in enumerate(results, 1):
            for col_num, cell_data in enumerate(row_data):
                worksheet.write(row_num, col_num, cell_data)

    print(f"UIQE和UCIQE的t检验结果已保存到 {t_test_output_file}")
    return results


def main():
    # 设置文件夹路径和输出文件路径

    # 数据集图像
    original_folder_1 = "images_original/EUVP_500/Inp"
    original_folder_2 = "images_original/LSUI_500/input"
    original_folder_3 = "images_original/UWIN_500"

    # DW
    input_1_1 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\DW\EUVP\\"
    input_1_2 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\DW\LSUI\\"
    input_1_3 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\DW\UWIN\\"

    # PU
    input_2_1 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\PU\EUVP\\"
    input_2_2 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\PU\LSUI\\"
    input_2_3 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\PU\UWIN\\"

    # UW
    input_3_1 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\UW\EUVP\\"
    input_3_2 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\UW\LSUI\\"
    input_3_3 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\UW\UWIN\\"

    # PU-DW
    output_1_1 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\PU-DW\EUVP\\"
    output_1_2 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\PU-DW\LSUI\\"
    output_1_3 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\PU-DW\UWIN\\"

    # DW-PU
    output_2_1 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\DW-PU\EUVP\\"
    output_2_2 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\DW-PU\LSUI\\"
    output_2_3 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\DW-PU\UWIN\\"

    # UW-DW
    output_3_1 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\UW-DW\EUVP\\"
    output_3_2 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\UW-DW\LSUI\\"
    output_3_3 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\UW-DW\UWIN\\"

    # DW-UW
    output_4_1 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\DW-UW\EUVP\\"
    output_4_2 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\DW-UW\LSUI\\"
    output_4_3 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\DW-UW\UWIN\\"

    # PU-UW
    output_5_1 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\PU-UW\EUVP\\"
    output_5_2 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\PU-UW\LSUI\\"
    output_5_3 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\PU-UW\UWIN\\"

    # UW-PU
    output_6_1 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\UW-PU\EUVP\\"
    output_6_2 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\UW-PU\LSUI\\"
    output_6_3 = r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance\UW-PU\UWIN\\"

    # 数据文件保存路径
    # DW
    data_1_1 = "results/DW/EUVP"
    data_1_2 = "results/DW/LSUI"
    data_1_3 = "results/DW/UWIN"

    # PU
    data_2_1 = "results/PU/EUVP/"
    data_2_2 = "results/PU/LSUI/"
    data_2_3 = "results/PU/UWIN/"

    # PW
    data_3_1 = "results/UW/EUVP/"
    data_3_2 = "results/UW/LSUI/"
    data_3_3 = "results/UW/UWIN/"

    # DW-PU
    data_4_1 = "results/DW-PU/EUVP/"
    data_4_2 = "results/DW-PU/LSUI/"
    data_4_3 = "results/DW-PU/UWIN/"

    # DW-UW
    data_5_1 = "results/DW-UW/EUVP/"
    data_5_2 = "results/DW-UW/LSUI/"
    data_5_3 = "results/DW-UW/UWIN/"

    # PU-DW
    data_6_1 = "results/PU-DW/EUVP/"
    data_6_2 = "results/PU-DW/LSUI/"
    data_6_3 = "results/PU-DW/UWIN/"

    # PU-UW
    data_7_1 = "results/PU-UW/EUVP/"
    data_7_2 = "results/PU-UW/LSUI/"
    data_7_3 = "results/PU-UW/UWIN/"

    # UW-DW
    data_8_1 = "results/UW-DW/EUVP/"
    data_8_2 = "results/UW-DW/LSUI/"
    data_8_3 = "results/UW-DW/UWIN/"

    # UW-PU
    data_9_1 = "results/UW-PU/EUVP/"
    data_9_2 = "results/UW-PU/LSUI/"
    data_9_3 = "results/UW-PU/UWIN/"

    # 文件名称
    file_name_PS = "PSresults.xlsx"
    file_name_UU = "UUresults.xlsx"
    file_name_TPS = "TPSresults.xlsx"
    file_name_TUU = "TUUresults.xlsx"

    # 原始图像选择
    original_folder = original_folder_3

    # 增强图像选择
    enhanced_folder = output_6_3

    # 选择保存路径
    psnr_ssim_output_file = data_9_3 + file_name_PS
    uiqe_uciqe_output_file = data_9_3 + file_name_UU
    psnr_ssim_t_test_output_file = data_9_3 + file_name_TPS
    uiqe_uciqe_t_test_output_file = data_9_3 + file_name_TUU

    # 确保文件夹存在
    for folder in [original_folder, enhanced_folder]:
        if not os.path.exists(folder):
            print(f"错误: 文件夹 {folder} 不存在")
            return

    # 创建输出目录（如果不存在）
    for output_file in [psnr_ssim_output_file, uiqe_uciqe_output_file,
                        psnr_ssim_t_test_output_file, uiqe_uciqe_t_test_output_file]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # 计算PSNR和SSIM
    psnr_ssim_results = calculate_psnr_ssim(original_folder, enhanced_folder, psnr_ssim_output_file)

    # 计算UIQE和UCIQE
    uiqe_uciqe_results = calculate_uiqe_uciqe(enhanced_folder, uiqe_uciqe_output_file)

    # 执行PSNR和SSIM的t检验
    perform_t_test(psnr_ssim_results, psnr_ssim_t_test_output_file)

    # 执行UIQE和UCIQE的t检验
    perform_uiqe_uciqe_t_test(uiqe_uciqe_results, uiqe_uciqe_t_test_output_file)

    print("所有分析已完成!")


if __name__ == "__main__":
    main()