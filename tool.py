"""
数据加载模块 - 处理rec和rec2文件的读取和转换
"""
import os
import numpy as np
import configparser
from PIL import Image, ImageFile
import cv2
from typing import Tuple, List, Optional, Dict, Any
import yaml
# 处理超大文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_root = config['paths']['raw_data']
        self.processed_root = config['paths']['processed_data']
        self.default_dtype = getattr(np, config['data_processing']['default_dtype'])
        
    def load_volume_from_ini(self, folder_path: str, ini_name: str) -> np.ndarray:
        """从ini配置文件加载volume数据"""
        # 读取ini文件
        ini_path = os.path.join(folder_path, ini_name)
        if not os.path.exists(ini_path):
            raise FileNotFoundError(f"ImageParam.ini not found in {folder_path}")
            
        config = configparser.ConfigParser(interpolation=None)
        config.read(ini_path, encoding='utf-8')
        
        # 获取图像参数
        width = int(config['RawImageInfo']['Width']) if ini_name == "ImageParam.ini" else int(config['Raw2VTKMhdParam']['Width']) 
        height = int(config['RawImageInfo']['Height']) if ini_name == "ImageParam.ini" else int(config['Raw2VTKMhdParam']['Height']) 
        begin_index = int(config['FileModule']['BeginIndex']) if ini_name == "ImageParam.ini" else int(config['Raw2VTKMhdParam']['BeginIndex']) 
        end_index = int(config['FileModule']['EndIndex']) if ini_name == "ImageParam.ini" else int(config['Raw2VTKMhdParam']['EndIndex']) 
        template = config['FileModule']['NameTemplate'] if ini_name == "ImageParam.ini" else "slice%d.bin"
        
        # 像素数据类型
        bits_allocated = int(config['RawImageInfo']['BitsAllocated']) if ini_name == "ImageParam.ini" else  16
        pixel_repr = int(config['RawImageInfo']['PixelRepresentation']) if ini_name == "ImageParam.ini" else  1
        
        # 推断numpy dtype
        if bits_allocated == 16:
            dtype = np.int16 if pixel_repr == 1 else np.uint16
        elif bits_allocated == 8:
            dtype = np.uint8
        else:
            dtype = self.default_dtype
            
        # 初始化体数据数组
        depth = end_index - begin_index + 1
        volume = np.zeros((depth, height, width), dtype=dtype)
        
        # 读取每个切片
        for i in range(begin_index, end_index + 1):
            # print(template)
            filename = template % i
            file_path = os.path.join(folder_path, filename)
            
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping...")
                continue
                
            with open(file_path, 'rb') as f:
                slice_data = np.frombuffer(f.read(), dtype=dtype)
                if slice_data.size != width * height:
                    print(f"Warning: Slice {filename} size mismatch")
                    continue
                volume[i - begin_index] = slice_data.reshape((height, width))
        
        return volume
    
    def load_volume_from_vgi_rec(self, vgi_path: str) -> np.ndarray:
        """从vgi文件加载对应的rec文件"""
        vgi_params = self._parse_vgi_file(vgi_path)
        rec_path = vgi_path[:-4] + '.rec' if vgi_path.lower().endswith('.vgi') else vgi_path + '.rec'
        if not os.path.exists(rec_path):
            alt = vgi_path[:-4] + '.REC'
            if os.path.exists(alt):
                rec_path = alt
            else:
                raise FileNotFoundError(f"REC file not found: {rec_path}")
        return self._load_rec_file(rec_path, vgi_params)

    def load_volume_from_rec(self, rec_path: str) -> np.ndarray:
        """从rec文件加载volume，自动查找同名vgi并解析尺寸"""
        base, ext = os.path.splitext(rec_path)
        vgi_path = base + '.vgi'
        if not os.path.exists(vgi_path):
            alt = base + '.VGI'
            if os.path.exists(alt):
                vgi_path = alt
            else:
                raise FileNotFoundError(f"VGI file not found for: {rec_path}")
        vgi_params = self._parse_vgi_file(vgi_path)
        return self._load_rec_file(rec_path, vgi_params)
    
    def _parse_vgi_file(self, vgi_path: str) -> dict:
        """
        解析vgi文件，兼容常见写法：
        - Size X/Size Y/Size Z + Bits per voxel
        - size = W H Z + bitsperelement
        - datatype 可选（unsigned integer/short等），位深优先以bits字段为准
        """
        kv = {}
        with open(vgi_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    kv[k.strip().lower()] = v.strip()

        # 优先解析三元 size
        width = height = depth = 0
        bits = None
        if 'size' in kv:
            parts = [p for p in kv['size'].replace(',', ' ').split() if p.isdigit()]
            if len(parts) >= 3:
                width, height, depth = map(int, parts[:3])
        # 兼容 Size X/Y/Z
        if width == 0:
            for key, alias in [('size x', 'x'), ('size y', 'y'), ('size z', 'z')]:
                if key in kv and kv[key].isdigit():
                    val = int(kv[key])
                    if alias == 'x':
                        width = val
                    elif alias == 'y':
                        height = val
                    else:
                        depth = val

        # 位深
        if 'bits per voxel' in kv and kv['bits per voxel'].isdigit():
            bits = int(kv['bits per voxel'])
        if bits is None and 'bitsperelement' in kv:
            num = ''.join(ch for ch in kv['bitsperelement'] if ch.isdigit())
            if num:
                bits = int(num)
        if bits is None:
            bits = 16

        data_type = kv.get('datatype', kv.get('data type', 'unsigned integer')).lower()
        return {
            'width': width,
            'height': height,
            'depth': depth,
            'data_type': data_type,
            'bits': bits
        }
    
    def _load_rec_file(self, rec_path: str, params: dict) -> np.ndarray:
        """加载rec文件"""
        width, height, depth = params['width'], params['height'], params['depth']
        bits = params['bits']
        
        # 确定数据类型
        if bits == 8:
            dtype = np.uint8
        elif bits == 16:
            dtype = np.uint16
        else:
            dtype = self.default_dtype
        
        # 读取数据
        with open(rec_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=dtype)
            
        # 重塑为volume
        expected_size = width * height * depth
        if data.size != expected_size:
            print(f"Warning: Data size mismatch. Expected {expected_size}, got {data.size}")
            # 截断或填充数据
            if data.size > expected_size:
                data = data[:expected_size]
            else:
                padded_data = np.zeros(expected_size, dtype=dtype)
                padded_data[:data.size] = data
                data = padded_data
        
        volume = data.reshape((depth, height, width))
        return volume


class Rec2Processor:
    """rec2文件处理器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.slice_height = config['data_processing']['slice_height']
        self.output_format = config['data_processing']['output_format']
        
    def extract_pngs_from_rec2(self, rec2_path: str, output_dir: str) -> List[str]:
        """从rec2文件提取PNG图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(rec2_path, 'rb') as f:
            data = f.read()
        
        ihdr_pattern = b'IHDR'
        iend_pattern = b'IEND'
        png_signature = b'\x89PNG\r\n\x1a\n'
        
        offset = 0
        count = 0
        extracted_files = []
        
        while True:
            # 找到IHDR块
            ihdr_pos = data.find(ihdr_pattern, offset)
            if ihdr_pos == -1:
                break
            
            # PNG块格式：长度(4字节) + 类型(4字节) + 数据 + CRC(4字节)
            chunk_start = ihdr_pos - 4
            if chunk_start < 0:
                break
            
            # 找到IEND结束块
            iend_pos = data.find(iend_pattern, ihdr_pos)
            if iend_pos == -1:
                break
            
            chunk_end = iend_pos + 8  # 4字节'IEND' + 4字节CRC
            
            # 构造完整PNG数据
            png_data = png_signature + data[chunk_start:chunk_end]
            
            # 保存PNG文件
            output_file = os.path.join(output_dir, f"slice_long_{count:03}.png")
            with open(output_file, 'wb') as out:
                out.write(png_data)
            
            extracted_files.append(output_file)
            print(f"Extracted PNG slice {count}")
            
            offset = chunk_end
            count += 1
        
        print(f"✅ 提取完毕，共提取 {count} 张图像")
        return extracted_files
    
    def split_long_png_to_slices(self, png_path: str, output_dir: str) -> List[str]:
        """将长PNG图像分割为切片"""
        os.makedirs(output_dir, exist_ok=True)
        
        img = Image.open(png_path)
        width, height = img.size
        
        num_slices = height // self.slice_height
        print(f"图像尺寸：{width}x{height}，将分割为 {num_slices} 张切片")
        
        slice_files = []
        for i in range(num_slices):
            box = (0, i * self.slice_height, width, (i + 1) * self.slice_height)
            slice_img = img.crop(box)
            
            slice_file = os.path.join(output_dir, f"slice_{i:03}.png")
            slice_img.save(slice_file)
            slice_files.append(slice_file)
        
        print("✅ 分割完成")
        return slice_files
    
    def process_rec2_to_volume(self, rec2_path: str, output_dir: str) -> np.ndarray:
        """处理rec2文件并转换为volume数据"""
        # 提取PNG文件
        temp_dir = os.path.join(output_dir, "temp")
        long_pngs = self.extract_pngs_from_rec2(rec2_path, temp_dir)
        
        # 分割PNG并加载为volume
        all_slices = []
        for long_png in long_pngs:
            slice_files = self.split_long_png_to_slices(long_png, output_dir)
            
            # 加载切片数据
            for slice_file in slice_files:
                slice_img = cv2.imread(slice_file, cv2.IMREAD_UNCHANGED)
                if slice_img is not None and (slice_img > 0).sum() > slice_img.size * 0.5:
                    all_slices.append(slice_img)
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if not all_slices:
            raise ValueError("No valid slices found in rec2 file")
        
        volume = np.array(all_slices)
        print(f"✅ 处理完成，volume shape: {volume.shape}")
        
        return volume


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

def load_processed_data(data_path):
    """加载处理后的数据"""
    if not os.path.exists(data_path):
        print(f"错误: 数据路径不存在: {data_path}")
        return None

    ini_name = "ImageParam.ini"
    ini_file = os.path.join(data_path, ini_name)
    if os.path.exists(ini_file):
        print(f"加载bin格式数据: {data_path}")
        try:
            config = load_config()
            data_loader = DataLoader(config)
            volume_data = data_loader.load_volume_from_ini(data_path, ini_name)
            print(f"数据加载成功，形状: {volume_data.shape}")
            return volume_data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    ini_name = "CreateVolumeParam.ini"
    ini_file = os.path.join(data_path, ini_name)
    if os.path.exists(ini_file):
        print(f"加载bin格式数据: {data_path}")
        try:
            config = load_config()
            data_loader = DataLoader(config)
            volume_data = data_loader.load_volume_from_ini(data_path, ini_name)
            print(f"数据加载成功，形状: {volume_data.shape}")
            return volume_data
        except Exception as e:
         
            print(f"数据加载失败: {e}")
            return None

    # vgi+rec: 传入的是vgi文件路径
    if os.path.isfile(data_path) and data_path.lower().endswith('.vgi'):
        try:
            print(f"加载REC格式数据: {data_path}")
            config = load_config()
            data_loader = DataLoader(config)
            volume_data = data_loader.load_volume_from_vgi_rec(data_path)
            print(f"数据加载成功，形状: {volume_data.shape}")
            return volume_data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    # 直接传入rec
    if os.path.isfile(data_path) and data_path.lower().endswith('.rec'):
        try:
            print(f"加载REC格式数据: {data_path}")
            config = load_config()
            data_loader = DataLoader(config)
            volume_data = data_loader.load_volume_from_rec(data_path)
            print(f"数据加载成功，形状: {volume_data.shape}")
            return volume_data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    # 目录中存在PNG
    png_files = [f for f in os.listdir(data_path)] if os.path.isdir(data_path) else []
    png_files = [f for f in png_files if f.lower().endswith('.png')]
    png_files.sort( key = lambda x: float(x.split('_')[-1][:-4]))

    if png_files:
        print(f"加载PNG格式数据: {data_path}")
        try:
            from src.auto_focus import load_volume_from_png_folder
            volume_data = load_volume_from_png_folder(data_path)
            print(f"数据加载成功，形状: {volume_data.shape}")
            return volume_data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    print(f"错误: 未找到支持的数据格式")
    return None
