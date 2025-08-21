import math
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


from scipy import ndimage
from PIL import Image
import numpy as np
import math



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

            # 构建完整路径
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

            # 存储结果（包含完整路径）
            results.append([original_img_path, enhanced_img_path, img_name, matched_enhanced, psnr_val, ssim_val])

        except Exception as e:
            print(f"处理图像 {img_name} 时出错: {str(e)}")

    # 将结果写入Excel文件
    if results:
        df = pd.DataFrame(results, columns=['原始图片完整路径', '增强图片完整路径', '原始图片名称', '增强图片名称', 'PSNR', 'SSIM'])
        df.to_excel(psnr_ssim_output_file, index=False, engine='openpyxl')
        print(f"PSNR和SSIM结果已保存到 {psnr_ssim_output_file}")
    else:
        print("没有PSNR和SSIM结果可保存")

    return results


def calculate_uciqe(img):
    """
    计算水下图像质量评价指标 UCIQE（修正版）
    :param img: 输入图像（BGR格式，uint8类型）
    :return: UCIQE 值
    """
    # 检查输入是否为有效的图像数组
    if not isinstance(img, np.ndarray) or img.ndim != 3:
        raise ValueError("输入必须是一个三维的numpy数组")

    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_LAB = np.array(img_LAB, dtype=np.float64)
    # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
    coe_Metric = [0.4680, 0.2745, 0.2576]

    img_lum = img_LAB[:, :, 0] / 255.0
    img_a = img_LAB[:, :, 1] / 255.0
    img_b = img_LAB[:, :, 2] / 255.0

    # item-1
    chroma = np.sqrt(np.square(img_a) + np.square(img_b))
    sigma_c = np.std(chroma)

    # item-2
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum) * 0.99)]
    bottom_index = sorted_index[int(len(img_lum) * 0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]

    # item-3
    chroma = chroma.flatten()
    sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum != 0)
    avg_sat = np.mean(sat)

    uciqe = sigma_c * coe_Metric[0] + con_lum * coe_Metric[1] + avg_sat * coe_Metric[2]
    return uciqe


def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    k1 = int(k1)
    k2 = int(k2)
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)


def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)


def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)


def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
    val = 0
    k1 = int(k1)
    k2 = int(k2)
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val



def calculate_uiqm(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### UIQM https://ieeexplore.ieee.org/abstract/document/7305804
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm


# def calculate_uiqm(img, window_size=8):
#     """
#     计算水下图像质量指标UIQM
#     :param img: 输入图像(BGR格式)
#     :param window_size: 分块大小(默认8x8)
#     :return: UIQM分数
#     """
#     # 验证输入
#     if img is None:
#         raise ValueError("输入图像为空")
#     if len(img.shape) != 3 or img.shape[2] != 3:
#         raise ValueError("输入图像必须是3通道彩色图像")
#
#     # 转换为RGB并归一化
#     if img.dtype == np.uint8:
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     else:
#         img_rgb = img
#
#     img_norm = normalize_image(img_rgb)
#
#     # 验证图像尺寸
#     h, w = img_norm.shape[:2]
#     if h < window_size or w < window_size:
#         # 对于小图像的特殊处理
#         window_size = min(h, w)
#         if window_size < 2:
#             return 0.0  # 返回默认值
#
#     try:
#         # 计算三个子指标
#         uicm = _uicm(img_norm)
#         uism = _uism(img_norm, window_size)
#         uiconm = _uiconm(img_norm, window_size)
#
#         # 加权组合
#         return 0.282 * uicm + 0.2953 * uism + 0.4677 * uiconm
#     except Exception as e:
#         print(f"计算UIQM时出错: {str(e)}")
#         return 0.0  # 返回默认值


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
            # 构建完整路径
            enhanced_img_path = os.path.join(enhanced_folder, img_name)

            # 读取增强图像
            enhanced_img = cv2.imread(enhanced_img_path)

            if enhanced_img is None:
                print(f"警告: 无法读取增强图像 {img_name}")
                continue

            # 计算UIQM和UCIQE
            uiqe_val = calculate_uiqm(enhanced_img)
            uciqe_val = calculate_uciqe(enhanced_img)

            # 存储结果（包含完整路径）
            results.append([enhanced_img_path, img_name, uiqe_val, uciqe_val])

        except Exception as e:
            print(f"处理图像 {img_name} 时出错: {str(e)}")

    # 将结果写入Excel文件
    if results:
        df = pd.DataFrame(results, columns=['图片完整路径', '图片名称', 'UIQM', 'UCIQE'])
        df.to_excel(uiqe_uciqe_output_file, index=False, engine='openpyxl')
        print(f"UIQM和UCIQE结果已保存到 {uiqe_uciqe_output_file}")
    else:
        print("没有UIQM和UCIQE结果可保存")

    return results


def perform_t_test(psnr_ssim_results, t_test_output_file):
    """对PSNR和SSIM结果进行t检验"""
    if not psnr_ssim_results or len(psnr_ssim_results) < 2:
        print("没有足够的PSNR和SSIM结果可用于t检验")
        return

    # 提取PSNR和SSIM值
    psnr_values = [result[4] for result in psnr_ssim_results]
    ssim_values = [result[5] for result in psnr_ssim_results]

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
    uiqe_values = [result[2] for result in uiqe_uciqe_results]
    uciqe_values = [result[3] for result in uiqe_uciqe_results]

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
        ["UIQM", uiqe_t_stat, uiqe_p_value, np.mean(uiqe_values),
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

    print(f"UIQM和UCIQE的t检验结果已保存到 {t_test_output_file}")
    return results


def setup_paths():
    """设置文件夹路径和输出文件路径"""
    base_dir = {
        "original": "images_original",
        "enhanced": r"C:\Users\qizerone\Desktop\paper1\evaluate\images_enhance",
        "results": "results/results-FIX"
    }

    datasets = {
        "EUVP": {"original": "EUVP_500/Inp",
                 "sub_dirs": ["DW", "PU", "UW", "DW-PU", "DW-UW", "PU-DW", "PU-UW", "UW-DW", "UW-PU"]},
        "LSUI": {"original": "LSUI_500/input",
                 "sub_dirs": ["DW", "PU", "UW", "DW-PU", "DW-UW", "PU-DW", "PU-UW", "UW-DW", "UW-PU"]},
        "UWIN": {"original": "UWIN_500",
                 "sub_dirs": ["DW", "PU", "UW", "DW-PU", "DW-UW", "PU-DW", "PU-UW", "UW-DW", "UW-PU"]},
        "U45": {"original": "U45_45",
                 "sub_dirs": ["DW", "PU", "UW", "DW-PU", "DW-UW", "PU-DW", "PU-UW", "UW-DW", "UW-PU"]}
    }

    method_pairs = ["DW", "PU", "UW", "DW-PU", "DW-UW", "PU-DW", "PU-UW", "UW-DW", "UW-PU"]

    paths = {
        "original": {},
        "enhanced": {},
        "results": {}
    }

    for dataset, info in datasets.items():
        paths["original"][dataset] = os.path.join(base_dir["original"], info["original"])

    for dataset, info in datasets.items():
        paths["enhanced"][dataset] = {}
        for method in info["sub_dirs"]:
            paths["enhanced"][dataset][method] = os.path.join(base_dir["enhanced"], method, dataset)

    for dataset in datasets.keys():
        paths["results"][dataset] = {}
        for method_pair in method_pairs:
            if isinstance(method_pair, tuple):
                output_method = f"{method_pair[0]}-{method_pair[1]}"
                paths["results"][dataset][output_method] = os.path.join(
                    base_dir["results"], output_method, dataset)
            else:
                output_method = method_pair
                paths["results"][dataset][output_method] = os.path.join(
                    base_dir["results"], f"{method_pair}-original", dataset)

    file_names = {
        "PS": "PSresults.xlsx",      #将所有输出的TPSresults.xlsx集成到一个AllTPSresults.xlsx文件当中，将所有的文件夹的TUUresults.xlsx集成到ALLTPS.xlsx文件当中
        "UU": "UUresults.xlsx",
        "TPS": "TPSresults.xlsx",
        "TUU": "TUUresults.xlsx"
    }

    return paths, method_pairs, file_names

def main():
    # 获取路径配置
    paths, method_pairs, file_names = setup_paths()

    # 对每个数据集和方法对执行评估
    for dataset in paths["original"].keys():
        for method_pair in method_pairs:
            if isinstance(method_pair, tuple):
                output_method = f"{method_pair[0]}-{method_pair[1]}"
            else:
                output_method = method_pair

            # 获取原始图像文件夹
            original_folder = paths["original"][dataset]

            # 获取增强图像文件夹
            if output_method in paths["enhanced"][dataset]:
                enhanced_folder = paths["enhanced"][dataset][output_method]
            else:
                print(f"错误: 方法 {output_method} 没有对应的增强图像文件夹")
                continue

            # 获取结果文件夹
            results_folder = paths["results"][dataset].get(output_method, None)
            if not results_folder:
                print(f"错误: 没有为 {output_method} 设置结果文件夹")
                continue

            # 构建输出文件路径
            psnr_ssim_output_file = os.path.join(results_folder, file_names["PS"])
            uiqe_uciqe_output_file = os.path.join(results_folder, file_names["UU"])
            psnr_ssim_t_test_output_file = os.path.join(results_folder, file_names["TPS"])
            uiqe_uciqe_t_test_output_file = os.path.join(results_folder, file_names["TUU"])

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

            print(f"数据集 {dataset} 和方法 {output_method} 的所有分析已完成!")

if __name__ == "__main__":
    main()