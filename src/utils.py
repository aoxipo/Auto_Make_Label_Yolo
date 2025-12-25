"""
工具函数模块
"""
import os
import yaml
import math
import bisect
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 处理自动设备检测
        if config.get('device', {}).get('name') == 'auto':
            config['device']['name'] = get_optimal_device()

        return config
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        print("使用默认配置")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'paths': {
            'data_root': './data',
            'raw_data': './data/raw',
            'processed_data': './data/processed',
            'output_data': './data/output',
            'model_path': './models/best.pt'
        },
        'device': {'name': get_optimal_device()},  # 自动检测最优设备
        'ai_inference': {
            'confidence_threshold': 0.35,
            'iou_threshold': 0.45,
            'max_detections': 1000,
            'normalize': True
        },
        'auto_focus': {
            'focus_threshold': 100,
            'grid_size': 4,
            'peak_range_ratio': 0.1,
            'use_parallel': True,
            'max_workers': 8
        },
        'data_processing': {
            'slice_height': 536,
            'output_format': 'png',
            'default_dtype': 'uint16'
        }
    }


def ensure_directories(config: Dict[str, Any]) -> None:
    """确保必要的目录存在"""
    paths_to_create = [
        config['paths']['processed_data'],
        config['paths']['output_data'],
        os.path.dirname(config['paths']['model_path'])
    ]
    
    for path in paths_to_create:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_data_files(data_path: str) -> Dict[str, list]:
    """
    获取数据文件列表
    
    Args:
        data_path: 数据目录路径
        
    Returns:
        包含不同类型文件列表的字典
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        return {'rec': [], 'rec2': [], 'vgi': [], 'png_dirs': []}
    
    # 查找不同类型的文件
    rec_files = list(data_path.glob("**/*.rec"))
    rec2_files = list(data_path.glob("**/*.rec2"))
    vgi_files = list(data_path.glob("**/*.vgi"))
    
    # 查找包含PNG文件的目录
    png_dirs = []
    for root, dirs, files in os.walk(data_path):
        png_files = [f for f in files if f.lower().endswith('.png')]
        if png_files:
            png_dirs.append(Path(root))
    
    return {
        'rec': rec_files,
        'rec2': rec2_files,
        'vgi': vgi_files,
        'png_dirs': png_dirs
    }


def print_data_summary(data_files: Dict[str, list]) -> None:
    """打印数据文件摘要"""
    print("数据文件摘要:")
    print(f"  REC 文件: {len(data_files['rec'])} 个")
    print(f"  REC2 文件: {len(data_files['rec2'])} 个")
    print(f"  VGI 文件: {len(data_files['vgi'])} 个")
    print(f"  PNG 目录: {len(data_files['png_dirs'])} 个")


def validate_model_file(model_path: str) -> bool:
    """验证模型文件是否存在且有效"""
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    if not model_path.endswith('.pt'):
        print(f"⚠️  模型文件格式可能不正确: {model_path}")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(model_path)
    if file_size < 1024:  # 小于1KB可能是无效文件
        print(f"⚠️  模型文件大小异常: {file_size} bytes")
        return False
    
    print(f"✅ 模型文件验证通过: {model_path} ({file_size / 1024 / 1024:.1f} MB)")
    return True

def get_color_for_class(cls_id):
    # 这里给每个类别分配不同颜色 (BGR格式)
    colors = [
        (0, 0, 255),    # 类别0 红色
        (0, 255, 0),    # 类别1 绿色
        (255, 0, 0),    # 类别2 蓝色
        (0, 255, 255),  # 类别3 黄色
        (255, 0, 255),  # 类别4 紫色
        (255, 255, 0),  # 类别5 青色
    ]
    return colors[cls_id % len(colors)]  # 避免越界


def crop_traget_area( preds, img_w, img_h):

    # 原始全局 min/max
    x_min = np.min(preds[:, 0])
    x_max = np.max(preds[:, 2])
    y_min = np.min(preds[:, 1])
    y_max = np.max(preds[:, 3])

    # 宽高
    width = x_max - x_min
    height = y_max - y_min

    # 向外扩 20%
    x_min_expanded = x_min - 0.2 * width
    x_max_expanded = x_max + 0.2 * width
    y_min_expanded = y_min - 0.2 * height
    y_max_expanded = y_max + 0.2 * height

    # 限制在图像范围
    x_min_expanded = int(max(0, x_min_expanded))
    y_min_expanded = int(max(0, y_min_expanded))
    x_max_expanded = int(min(img_w, x_max_expanded))
    y_max_expanded = int(min(img_h, y_max_expanded))
    return x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded


def normalize_image(image: np.ndarray, target_dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    归一化图像到指定数据类型
    
    Args:
        image: 输入图像
        target_dtype: 目标数据类型
        
    Returns:
        归一化后的图像
    """
    if image.dtype == target_dtype:
        return image
    
    # 归一化到0-1
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # 转换到目标类型
    if target_dtype == np.uint8:
        return (image_norm * 255).astype(np.uint8)
    elif target_dtype == np.uint16:
        return (image_norm * 65535).astype(np.uint16)
    else:
        return image_norm.astype(target_dtype)

def nearest_power_of_two(n):
    """返回最接近n的2的幂次方"""
    if n <= 0:
        return 1
    # 计算log2(n)
    log2 = math.log2(n)
    # 找到相邻的两个幂次方
    lower_pow = int(log2)
    higher_pow = lower_pow + 1
    # 计算对应的数值
    lower_val = 2 ** lower_pow
    higher_val = 2 ** higher_pow
    # 判断哪个更接近
    # return (higher_val + lower_val)//2
    # if abs(n - lower_val) < abs(n - higher_val):
    #     return lower_val
    # else:
    #     return higher_val
    if n == lower_val:
        return lower_val
    if n == higher_val:
        return higher_val
    
    return (higher_val + lower_val)//2

def nearest_multiple_of_two(n):
    """返回最接近 n 的 2 的倍数"""
    return round(n / 2) * 2
 
def find_best_scale(srcW: int) -> int:
    """返回 ≥srcW 的最小 table_idx 值（类似 ceil 操作）"""
    table_idx = [ (i+1) * 32 for i in range(40)]
    
    if srcW < table_idx[0]:
        return  128 # 原代码逻辑，但可能是 bug（见下方分析）
    elif srcW > table_idx[-1]:
        return  1024  # 原代码逻辑，但可能是 bug（见下方分析）
    else:
        # 找到第一个 ≥srcW 的索引
        idx = bisect.bisect_left(table_idx, srcW)
        return table_idx[idx]

def find_nearest_power_of_two_dimensions(width, height, purpose="推理"):
    """返回宽度和高度的最接近的2的幂次方"""
    nearest_width = find_best_scale(width)
    nearest_height = find_best_scale(height)
    print(f"find best infer shape: {purpose} - 原始尺寸({width}x{height}) -> 最佳推理尺寸({nearest_width}x{nearest_height})")
    return nearest_width, nearest_height


def create_output_filename(input_path: str, suffix: str = "", extension: str = ".png") -> str:
    """
    创建输出文件名
    
    Args:
        input_path: 输入文件路径
        suffix: 文件名后缀
        extension: 文件扩展名
        
    Returns:
        输出文件名
    """
    input_path = Path(input_path)
    base_name = input_path.stem
    
    if suffix:
        output_name = f"{base_name}_{suffix}{extension}"
    else:
        output_name = f"{base_name}{extension}"
    
    return output_name


def safe_create_directory(directory: str) -> bool:
    """
    安全创建目录
    
    Args:
        directory: 目录路径
        
    Returns:
        是否创建成功
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ 创建目录失败 {directory}: {e}")
        return False


def get_optimal_device() -> str:
    """自动检测最优计算设备"""
    try:
        import torch

        # 优先级: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def get_system_info() -> Dict[str, str]:
    """获取系统信息"""
    import platform

    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'optimal_device': get_optimal_device()
    }

    try:
        import torch
        info['torch_version'] = torch.__version__

        # 检查GPU支持
        if torch.cuda.is_available():
            info['cuda_available'] = 'Yes'
            info['cuda_version'] = torch.version.cuda
        else:
            info['cuda_available'] = 'No'

        # 检查MPS支持 (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps_available'] = 'Yes'
        else:
            info['mps_available'] = 'No'
    except ImportError:
        info['torch_version'] = 'Not installed'
        info['cuda_available'] = 'No'
        info['mps_available'] = 'No'

    return info


def print_system_info() -> None:
    """打印系统信息"""
    info = get_system_info()
    print("系统信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # 测试工具函数
    print("🔧 WeltDetect 工具函数测试")
    print("=" * 40)
    
    # 打印系统信息
    print_system_info()
    
    # 测试配置加载
    print("\n📝 测试配置加载:")
    config = load_config()
    print(f"设备: {config['device']['name']}")
    print(f"模型路径: {config['paths']['model_path']}")
    
    # 测试数据文件查找
    print("\n📁 测试数据文件查找:")
    data_files = get_data_files("./data/raw")
    print_data_summary(data_files)
