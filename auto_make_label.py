import json
import os
import cv2
import numpy as np
from src.ai_inference  import *
from tool import load_processed_data
 
def bin2png( file_path, save_folder):
    os.makedirs( save_folder, exist_ok=True)
    volume_data = load_processed_data(file_path)
    idx = 0
    for data in volume_data:
        data = np.array(data)
        if np.sum(data) == 0:
            continue
        data = np.array( ( data - data.min() ) / ( data.max() - data.min() ) * 255, dtype = np.uint8)
        cv2.imwrite(os.path.join(save_folder, f"{idx}.png"), data)
        print(os.path.join(save_folder, f"{idx}.png"))
        idx += 1
        
def save_yolo_to_labelme(result, image_path, class_names, save_path):
    """
    将 YOLOv5 格式结果转换为 LabelMe JSON 格式并保存
    
    Args:
        result (np.ndarray): [N, 6] -> (x_center, y_center, w, h, conf, class_id)
        image_path (str): 原始图像路径
        class_names (list): 类别名称列表
        save_path (str): json 保存路径
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    shapes = []
    for det in result:
        # print(det)
        x_c, y_c, bw, bh, conf, cls_id = det
        # print(det)
        cls_id = int(cls_id)
        label = class_names[cls_id]
        # 转换为 x1y1x2y2
        x1 = round(float(x_c), 3) # round(float(max(0, (x_c - bw / 2))), 3)
        y1 = round(float(y_c), 3) # round(float(max(0, (y_c - bh / 2))), 3)
        x2 = round(float(bw), 3) # round(float(min(w - 1, (x_c + bw / 2))), 3)
        y2 = round(float(bh), 3) # round(float(min(h - 1, (y_c + bh / 2))), 3)
         
        shape = {
            "label": label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": cls_id,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
        }
        shapes.append(shape)

    data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved labelme annotation to {save_path}")

if __name__ == "__main__":
    
    DEVICE = "cuda:0"
    WEIGHT_PATH = r"D:/work/Code/auto_make_label/weight/v11/best.pt"  
    model_type = "yolov11"
    folder_path = r'D:/work/Code/auto_make_label/data/CT-2025-10-20-18-35-33/'
    save_folder = r"D:/work/Code/auto_make_label/data/CT-2025-10-20-18-35-33/labelme/"
    os.makedirs(save_folder, exist_ok=True)
   
  
    config = {}
    config['device'] = {}
    config['device']['name'] = DEVICE
    config['paths'] = {}
    config['paths']['model_path'] = WEIGHT_PATH
    config['ai_inference'] = {}
    config['ai_inference']['confidence_threshold'] = 0.35
    config['ai_inference']['iou_threshold'] = 0.2
    inferProcessor = InferenceProcessor(config)
    file_name_list = os.listdir(folder_path)
    class_names = ["welt", "welt_mix", "welt_rosin", "all"]
    
 
    for file_name in file_name_list:
        if not file_name.endswith(".png"):
            continue
        image_path = folder_path + file_name
        save_name = image_path.split("/")[-1][:-3] + "json"
        image = cv2.imread( image_path, -1)
        if len(image.shape) == 3:
            image = image[:,:,0]
        
            
        result = inferProcessor.inference(image, model_type = model_type)
        # 保存为 labelme json
        if len(result) == 0:
            continue
        save_yolo_to_labelme(result[0], image_path, class_names, save_folder + save_name)
