import json
import os
import cv2
import numpy as np
import argparse
from glob import glob
import shutil

def get_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    ]


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area

    if union == 0:
        return 0

    return inter_area / union


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_labelme(folder, save_folder, iou_thresh=0.5):
    os.makedirs( save_folder, exist_ok=True)
    json_files = glob(os.path.join(folder, "*.json"))

    if len(json_files) == 0:
        print("No json files found")
        return

    # 找到 shapes 最多的 json
    max_shapes = -1
    base_json = None
    base_data = None

    for jf in json_files:
        data = load_json(jf)

        num_shapes = len(data.get("shapes", []))

        print(f"{os.path.basename(jf)} -> {num_shapes}")

        if num_shapes > max_shapes:
            max_shapes = num_shapes
            base_json = jf
            base_data = data

    print(f"\nBase json: {base_json}")
    shutil.copy( base_json[:-4] + "png", save_folder + f"/{os.basename(base_json).replace('json','png')}" )
    merged_shapes = base_data["shapes"]

    existing_boxes = [
        get_bbox(s["points"])
        for s in merged_shapes
    ]

    # 合并其它 json
    for jf in json_files:

        if jf == base_json:
            continue

        data = load_json(jf)

        for shape in data.get("shapes", []):

            new_box = get_bbox(shape["points"])

            duplicated = False

            for old_box in existing_boxes:

                iou = bbox_iou(new_box, old_box)

                if iou > iou_thresh:
                    duplicated = True
                    break

            if not duplicated:
                merged_shapes.append(shape)
                existing_boxes.append(new_box)

    base_data["shapes"] = merged_shapes

    save_path = os.path.join( save_folder, f"{os.basename(base_json)}")
       
    save_json(base_data, save_path)

    print(f"\nMerged json saved to:\n{save_path}")
    print(f"Total shapes: {len(merged_shapes)}")


   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="merge json data to one"
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

    folder = args.input
    save_folder = args.output

    merge_labelme(folder, save_folder)
 
