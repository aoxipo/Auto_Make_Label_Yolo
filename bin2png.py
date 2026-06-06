import json
import os
import cv2
import numpy as np
from src.ai_inference  import *
from tool import load_processed_data
import argparse

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
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert bin volume data to png images"
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input bin file path"
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output folder path"
    )
    
    args = parser.parse_args()

    bin2png(args.input, args.output)
