import json
import os
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import argparse

def visualize_sample_matplotlib(image_path: str, label_path: str, class_names: List[str] = None):
    """
    使用Matplotlib可视化图像和YOLO标签
    :param image_path: 图像路径
    :param label_path: 标签路径（YOLO格式）
    :param class_names: 类别名称列表（如["cat", "dog"]），若为None则显示类别ID
    """
    # 读取图像并转换颜色通道（OpenCV默认BGR，Matplotlib需要RGB）
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 读取标签文件
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # 创建画布
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image_rgb)
    
    # 绘制每个标注框
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # 跳过格式错误的行
        
        # 解析标签
        class_id = parts[0]
        x_center = float(parts[1]) * image.shape[1]  # 归一化坐标转实际像素
        y_center = float(parts[2]) * image.shape[0]
        width = float(parts[3]) * image.shape[1]
        height = float(parts[4]) * image.shape[0]
        
        # 计算边界框坐标
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        # print(x_center, y_center, width, height, x_min, x_max, y_min,y_max )
        # 生成矩形框和标签文本
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 显示类别名称或ID
        # if class_names and class_id < len(class_names):
        #     label = class_names[class_id]
        # else:
        #     label = f"Class {class_id}"

        label = class_id
        
        # ax.text(
        #    x_min, y_min - 5, label,
        #    color='white', fontsize=10,
        #    bbox=dict(facecolor='red', alpha=0.5, edgecolor='none')
        # )
    # plt.xlim([400,1000])
    # plt.ylim([400,1000])
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

def clamp_01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
    
def check_yolo_txt_range_and_bbox(txt_path):
    """
    检查 YOLO 格式标签文件：
    1. 原始归一化 (x_center, y_center, w, h) 是否在 [0,1]
    2. 转换后的 (x1, y1, x2, y2) 是否在 [0,1]

    Args:
        txt_path (str): 标签文件路径

    Returns:
        List[Tuple[int, str, str]]: 返回 (行号, 原始行, 错误类型) 的列表
    """
    invalid_lines = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            parts = line.strip().split()

            if len(parts) != 5:
                invalid_lines.append((idx, line.strip(), "格式错误"))
                continue

            try:
                cls_id = parts[0]
                x_center, y_center, w, h = map(float, parts[1:])

                # 1. 检查中心点与宽高
                if not all(0 <= v <= 1 for v in (x_center, y_center, w, h)):
                    invalid_lines.append((idx, line.strip(), "中心点/宽高超出范围"))

                # 2. 转换为 (x1, y1, x2, y2)
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2

                if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                    invalid_lines.append((idx, line.strip(), "边界框超出图像范围"))

            except ValueError:
                invalid_lines.append((idx, line.strip(), "解析错误"))

    return invalid_lines

def check_yolo_range_and_bbox(lines):
    """
    检查 YOLO 格式标签文件：
    1. 原始归一化 (x_center, y_center, w, h) 是否在 [0,1]
    2. 转换后的 (x1, y1, x2, y2) 是否在 [0,1]

    Args:
        txt_path (str): 标签文件路径

    Returns:
        List[Tuple[int, str, str]]: 返回 (行号, 原始行, 错误类型) 的列表
    """
    invalid_lines = []
    for idx, line in enumerate(lines, start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            invalid_lines.append((idx, line.strip(), "格式错误"))
            continue

        try:
            cls_id = parts[0]
            x_center, y_center, w, h = map(float, parts[1:])

            # # 1. 检查中心点与宽高
            # if not all(0 <= v <= 1 for v in (x_center, y_center, w, h)):
            #     invalid_lines.append((idx, line.strip(), "中心点/宽高超出范围"))

            # 2. 转换为 (x1, y1, x2, y2)
            x1 = round(x_center - w / 2, 6)
            y1 = round(y_center - h / 2, 6)
            x2 = round(x_center + w / 2, 6)
            y2 = round(y_center + h / 2, 6)

            if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                invalid_lines.append((idx, line.strip(), "边界框超出图像范围"))

        except ValueError:
            invalid_lines.append((idx, line.strip(), "解析错误"))

    return invalid_lines

label_dict = {
    "welt": 0,
    "welt_mix": 1,
    "welt_rosin": 2,
    "all": 3,

}


def labelme_to_txt(json_path, txt_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    with open(txt_path, 'w') as f_txt:
        for shape in data['shapes']:
            height = data['imageHeight']
            width  = data['imageWidth']
            label  = shape['label']
            points = shape['points']
            # 格式化为：label x1 y1 x2 y2 ...
            x1, y1 = points[0]
            x2, y2 = points[1]
            x_center = (x1 + x2)/2/width 
            y_center = (y1 + y2)/2/height
            obj_width = abs(x2 - x1)/width 
            obj_height = abs(y2 - y1)/height
            
            f_txt.write(f"{label_dict[label]} {x_center:.6f} {y_center:.6f} {obj_width:.6f} {obj_height:.6f}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="label json 2 train txt "
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input json file path"
    )

    args = parser.parse_args()
    root_path = args.input 
    file_list = os.listdir(root_path)
  
    for file_name in file_list:
        if file_name.endswith('.json'):
            # print(root_path + file_name)
            json_file = root_path + file_name
            labelme_to_txt(json_file, json_file.replace('json','txt'))

    
    # visualize CHECK 
    # visualize_sample_matplotlib(
        # image_path=r"D:\Desktop\notebook\augmented_dataset_new/aug_0067_12500_5_0.jpg",
        # label_path=r"D:\Desktop\notebook\augmented_dataset_new/aug_0067_12500_5_0.txt",
    #    image_path=r"/Path/to/CT/1020/choose/41.png",
    #     label_path=r"/Path/to/CT/1020/choose/41.txt",
    #    class_names=["welt", "mix", "weak", "all"]
    # )
