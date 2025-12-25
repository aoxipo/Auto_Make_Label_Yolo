# CT Slice Detection Pipeline (YOLOv5/v11 + LabelMe)

本项目用于标签反打 **CT 切片数据的目标检测与标注生成**，整体流程为：

> **CT 原始 Slice（bin后缀二进制） → PNG → YOLO 推理 → LabelMe 标注（JSON）**

---

## 1. 功能概览

- CT Slice（二进制）转 PNG
- YOLO 目标检测
- 自动生成 LabelMe JSON 标注
- 支持 CUDA 推理

---

## 2. 目录结构示例

/data/lijunlin/data/CT/test/1/
├── Slice/
├── png/1/
│   ├── 0001.png
│   ├── 0002.png
│   └── labelme/
│       ├── 0001.json
│       └── 0002.json

---

## 3. 环境依赖

- Python >= 3.8
- torch / torchvision（CUDA 版本一致）
- opencv-python
- ultralytics

---

## 4. 使用流程

### 4.1 Slice 转 PNG

```python
bin2png(input_data, folder_path)
```

### 4.2 推理与标注生成

```python
inferProcessor = InferenceProcessor(config)
result = inferProcessor.inference(image)
save_yolo_to_labelme(result[0], image_path, class_names, save_path)
```

---

## 5. 注意事项

- torch 与 torchvision CUDA 版本必须一致
- 输入为单通道 CT 图像
- YOLOv11 已内置 NMS

---

