"""
AI推理模块 - 重构版
"""
import cv2
import numpy as np
import torch
import os
import sys
from typing import List, Tuple, Optional, Dict, Any
import sys
from .utils import find_nearest_power_of_two_dimensions
import pathlib
import torch.nn.functional as F
import torchvision

if sys.platform.startswith('win32'):
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
# 动态导入YOLOv5模块
try:
    from yolov5.utils.general import scale_boxes as scale_coords
    from yolov5.utils.general import non_max_suppression
    from yolov5.models.experimental import *
except ImportError:
    print("Warning: YOLOv5 not found. Please install: pip install yolov5")
    scale_coords = None
    non_max_suppression = None

def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode
   
    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
  
    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model




class InferenceProcessor:
    def __init__(self, config):
        self.config = config
        self.device = config['device']['name']
        self.model_path = config['paths']['model_path']
        self.confidence_threshold = config['ai_inference']['confidence_threshold']
        self.iou_threshold = self.config['ai_inference']['iou_threshold']
        self.model =  self.ai_init( self.model_path, self.device)
        
    def ai_init(self, weights="./models/best.pt", device="auto"):
        """初始化AI模型 - 跨平台自动适配"""
        if not os.path.exists(weights):
            print(f"错误: 模型文件未找到: {weights}")
            return None

        # 自动检测最优设备
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # 验证设备可用性
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA不可用，切换到CPU")
            device = "cpu"
        elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("MPS不可用，切换到CPU")
            device = "cpu"

        print(f"使用设备: {device}")

        try:
            # 尝试使用torch.load，设置weights_only=False以兼容YOLOv5模型
            try:
                w = str(weights[0] if isinstance(weights, list) else weights)
                model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, device=device)   #加载模型
                
                return model
            except Exception as e:
                print(f"使用torch.load/attempt_load 时出错: {e}, 尝试其他方案")
                try:
                    checkpoint = torch.load(weights, map_location=device, weights_only=False)
                    model = checkpoint.get('model') or checkpoint.get('ema') or checkpoint
                    if hasattr(model, 'to'):
                        model = model.to(device).float().eval()
                    print(f"使用torch.load加载模型成功: {weights}")
                    return model
                except Exception as e1:
                    print(f"torch.load失败: {e1}")

                    # 尝试使用YOLOv5库
                    try:
                        import yolov5
                        # 临时修改torch.load的默认行为
                        original_load = torch.load
                        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)

                        model = yolov5.load(weights, device=device)

                        # 恢复原始torch.load
                        torch.load = original_load

                        print(f"使用yolov5库加载模型成功: {weights}")
                        return model
                    except Exception as e2:
                        print(f"yolov5库加载失败: {e2}")

                        # 最后尝试torch.hub
                        try:
                            model = torch.hub.load('ultralytics/yolov5', 'custom', weights, device=device, trust_repo=True)
                            print(f"使用torch.hub加载模型成功: {weights}")
                            return model
                        except Exception as e3:
                            print(f"torch.hub加载失败: {e3}")
                            raise e1  # 抛出最初的错误

        except Exception as e:
            print(f"模型加载失败: {e}")
            print("提示: 这可能是PyTorch 2.6的安全特性导致的")
            print("模型文件本身应该没问题，是加载方式的兼容性问题")
            return None

    def remove_multi_label_by_priority_tensor(
        self, 
        detections: torch.Tensor, 
        class_priority=[1, 2, 0, 3], 
        iou_threshold=0.5
    ) -> torch.Tensor:
        """
        按类别优先级去除多标签框（YOLO检测输出版本）
        
        Args:
            detections: Tensor, shape [N, 6], 每行 [x1, y1, x2, y2, conf, class]
            class_priority: list，类别优先级列表（数值小优先级高）
            iou_threshold: IoU 阈值
            
        Returns:
            Tensor, shape [M, 6]，处理后的结果
        """
        if detections.numel() == 0:
            return detections.clone()
        
        # 1. 生成类别优先级映射
        class_rank = {cls: i for i, cls in enumerate(class_priority)}
        priority = torch.tensor(
            [class_rank.get(int(c.item()), len(class_priority)) for c in detections[:, 5]],
            device=detections.device
        )

        # 2. 按优先级升序、置信度降序排序
        sort_keys = priority * 1e6 - detections[:, 4] * 1e3
        detections = detections[torch.argsort(sort_keys)]
        
        # 3. IoU 函数
        def iou_tensor(box1, box2):
            xi1 = torch.max(box1[0], box2[0])
            yi1 = torch.max(box1[1], box2[1])
            xi2 = torch.min(box1[2], box2[2])
            yi2 = torch.min(box1[3], box2[3])
            
            inter_area = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area

        # 4. 遍历去除 IoU 过高的重复框
        keep = []
        used = set()

        N = detections.shape[0]
        for i in range(N):
            if i in used:
                continue

            keep.append(detections[i])
            for j in range(i + 1, N):
                if j in used:
                    continue
                if iou_tensor(detections[i, 0:4], detections[j, 0:4]).item() > iou_threshold:
                    used.add(j)

        if len(keep) == 0:
            return detections.new_empty((0, detections.shape[1]))
        
        return torch.stack(keep, dim=0)

    def warpper_img(self, img):
        if len(img.shape) == 2 or img.shape[-1] == 1:
            img = np.dstack( ( img, img, img)) 
        img_wappered = np.transpose(img, (2, 0, 1))
        img_wappered = ( img_wappered - img_wappered.min())/( img_wappered.max() - img_wappered.min() ) 
        return img_wappered

    def inference(self, img0, model_type = "yolov5"):
        if model_type == "yolov5":
            return self.inference_v5(img0)
        elif model_type == "yolov7":
            raise NotImplementedError   
        elif model_type == "yolov8":
            raise NotImplementedError   
        elif model_type == "yolov11":
            return self.inference_v11(img0)
        else:
            return self.inference_v11(img0)

    def inference_v5(self, img0):
         
        height, width = img0.shape
        best_height, best_width = find_nearest_power_of_two_dimensions(height, width, "单张推理")
        img1 = cv2.resize(img0, (  best_width, best_height))
        img1 = self.warpper_img(img1)

        img = np.expand_dims(img1, axis=0)    #扩展维度至[1,3,1024,1024]
        img = torch.from_numpy(img.copy())   #numpy转tensor
        img = img.to(torch.float32)          #float64转换float32
        img = img.to(self.device)
        pred = self.model(img, augment='store_true', visualize='store_true')[0]
        pred = pred.clone().detach().cpu()
        
        pred = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold, None, False, max_det=1000)  #非极大值抑制
        # pred的长度为1 pred[0].shape 为 (256,6)  ( point number, x0,y0,x1,y1,conf, cls)

        pred = self.remove_multi_label_by_priority_tensor(pred[0])
        pred = [ pred ]
        
        result_l = []

        for i, det in enumerate(pred):
            if len(det):
                # 由于前处理使用的是各向异性resize(分别缩放宽高)，不能直接用yolov5的scale_coords（其假设等比缩放+letterbox）。
                # 这里改为按宽高分别线性缩放把坐标还原到原图尺寸。
                h0, w0 = img0.shape[:2]
                h1, w1 = img.shape[2], img.shape[3]
                if w1 == 0 or h1 == 0:
                    continue
                rw = w0 / float(w1)
                rh = h0 / float(h1)
                det[:, [0, 2]] *= rw
                det[:, [1, 3]] *= rh
                # 裁剪到图像边界，并取整
                det[:, 0].clamp_(0, w0 - 1)
                det[:, 2].clamp_(0, w0 - 1)
                det[:, 1].clamp_(0, h0 - 1)
                det[:, 3].clamp_(0, h0 - 1)
                det[:, :4] = det[:, :4].round()
                result_l.append(det.cpu().numpy())

        return result_l

    def inference_v11(self, img0):
        """
        YOLOv11 inference
        return: List[np.ndarray], 每个元素 shape = [N, 6] (x1,y1,x2,y2,conf,cls)
        """

        h0, w0 = img0.shape[:2]

        best_h, best_w = find_nearest_power_of_two_dimensions(h0, w0, "infer")
        img1 = cv2.resize(img0, (best_w, best_h))
        img1 = self.warpper_img(img1)

        # BCHW
        img = torch.from_numpy(img1).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img)[0]   # [1, 8, 11225]

        # -------- decode --------
        pred = pred.squeeze(0).permute(1, 0).cpu()  # [11225, 8]

        # obj conf
        conf = pred[:, 4]
        mask = conf > self.confidence_threshold
        pred = pred[mask]

        if pred.shape[0] == 0:
            return []

        # class
        cls_scores = pred[:, 5:]
        cls_conf, cls_id = cls_scores.max(dim=1)

        scores = conf[mask] * cls_conf

        # xywh → xyxy
        xywh = pred[:, :4]
        xyxy = torch.zeros_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

        # NMS
        keep = torchvision.ops.nms(xyxy, scores, self.iou_threshold)
        # pred = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold, None, False, max_det=1000)  #非极大值抑制


        det = torch.cat([
            xyxy[keep],
            scores[keep, None],
            cls_id[keep, None].float()
        ], dim=1)

        # scale back
        det[:, [0, 2]] *= w0 / best_w
        det[:, [1, 3]] *= h0 / best_h

        return [det.cpu().numpy()]


    def inference_batch(self, batch_image):

         
        batch, height, width = batch_image.shape
        best_height, best_width = find_nearest_power_of_two_dimensions(height, width, "批量推理")

        batch_tensor = torch.from_numpy(batch_image).float().to(self.device)  # (B,H,W) 或 (B,C,H,W)
        if batch_tensor.ndim == 3:  # 单通道
            batch_tensor = batch_tensor.unsqueeze(1)  # (B,1,H,W)
            batch_tensor = batch_tensor.repeat(1,3,1,1)

        batch_image_resize = F.interpolate(batch_tensor, size=(best_height, best_width), mode='bilinear', align_corners=False)
        batch_image_resize = batch_image_resize/batch_image_resize.max()
    
        pred = self.model(batch_image_resize, augment='store_true', visualize='store_true')[0]
        pred = pred.clone().detach().cpu()
         
        pred_batch = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold, None, False, max_det=1000)  #非极大值抑制
       
        result_l = []
        for i, det in enumerate(pred_batch):
            if len(det):
                # 应用多标签去重处理
                det = self.remove_multi_label_by_priority_tensor(det)

                h0, w0 = height, width
                h1, w1 = best_height, best_width
                if w1 == 0 or h1 == 0:
                    continue
                rw = w0 / float(w1)
                rh = h0 / float(h1)
                det[:, [0, 2]] *= rw
                det[:, [1, 3]] *= rh

                det[:, 0].clamp_(0, w0 - 1)
                det[:, 2].clamp_(0, w0 - 1)
                det[:, 1].clamp_(0, h0 - 1)
                det[:, 3].clamp_(0, h0 - 1)
                det[:, :4] = det[:, :4].round()
                result_l.append(det.cpu().numpy())
            else:
                result_l.append(None)
        
        return result_l

    # def get_mask(self, pred, origin_shape):
    #     mask = np.zeros(origin_shape)
    #     for i, det in enumerate(pred):
    #         if len(det):
    #             for *xyxy, conf, cls in reversed(det):
    #                 mask[ int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = 1
    #     return mask
    
    def show_result(self, pred, img0):
        for i, det in enumerate(pred):
            if det is not None:
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        img0 = cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        return img0

# 开发环境下的简单自测入口（按需启用）
# if __name__ == "__main__":
#     DEVICE = "mps"
#     WEIGHT_PATH = "./models/best.pt"
#     test_image_path = "./data/processed/test_image.png"
#     if os.path.exists(test_image_path):
#         img0 = cv2.imread(test_image_path, 0)
#         img = warpper_img(img0)
#         model = ai_init(WEIGHT_PATH, DEVICE)
#         if model is not None:
#             result = inference(model, img, DEVICE)
#             mask = get_mask(result, img0.shape)
#             rect_show = show_result(result, img, img0.copy())
#             print("AI推理测试完成")
#         else:
#             print("模型加载失败")
#     else:
#         print(f"测试图像不存在: {test_image_path}")
#         print("请先运行数据预处理脚本生成测试数据")
