'''

auther:Chaoqun Yang
work process：
1、读取指定文件夹下的所有图片文件
2、设定指定的像素参数如256x256
3、将修改号的图片保存到指定文件夹中

'''

import os
from PIL import Image
import argparse


def process_images(input_folder, output_folder, size):
    """处理图片：调整尺寸并保存到输出文件夹"""
    # 若输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 定义支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_ext = os.path.splitext(filename)[1].lower()

        # 检查文件是否为支持的图片格式
        if file_ext in supported_formats:
            input_path = os.path.join(input_folder, filename)

            try:
                # 打开图片
                with Image.open(input_path) as img:
                    # 调整图片尺寸
                    resized_img = img.resize(size, Image.LANCZOS)

                    # 构建输出路径
                    output_path = os.path.join(output_folder, filename)

                    # 保存调整后的图片
                    # 对于PNG图片保留透明度
                    if file_ext == '.png':
                        resized_img.save(output_path, 'PNG')
                    else:
                        resized_img.save(output_path)

                    print(f"已处理: {filename} -> {size}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='批量调整图片尺寸')
    parser.add_argument('--input', required=True, help='输入文件夹路径')
    parser.add_argument('--output', required=True, help='输出文件夹路径')
    parser.add_argument('--size', type=int, nargs=2, default=[256, 256],
                        help='目标尺寸，格式为 WIDTH HEIGHT，默认为 256 256')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用处理函数
    process_images(args.input, args.output, tuple(args.size))


if __name__ == "__main__":
    main()