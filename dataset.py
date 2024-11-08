from dataclasses import dataclass
from typing import Optional
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
import os
import numpy as np

def convert_yolo_to_xy(yolo_line, img_width, img_height):
    """
    Converts a YOLO format bounding box to xmin, ymin, xmax, ymax.
    Args:
        yolo_line: A string representing a YOLO format bounding box.
        img_width: The width of the image.
        img_height: The height of the image.
    Returns:
        A tuple containing (xmin, ymin, xmax, ymax).
    """
    class_id, x_center, y_center, width, height = map(float, yolo_line.split())
    xmin = int((x_center - width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    xmax = int((x_center + width / 2) * img_width)
    ymax = int((y_center + height / 2) * img_height)
    return xmin, ymin, xmax, ymax, class_id

@dataclass
class Dataset:
    folder_path: str
    prefix: str
    dataframe: Optional[pd.DataFrame] = None

    @classmethod
    def from_folder(cls, folder_path: str, prefix: str) -> 'Dataset':
        df = cls._load_image_info(folder_path, prefix)
        return cls(folder_path=folder_path, prefix=prefix, dataframe=df)

    @classmethod
    def _load_image_info(cls, folder_path: str, prefix: str) -> pd.DataFrame:
        """Load image annotations and record object info."""
        image_files = []
        image_extension = ('.png', '.jpg', '.jpeg')
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(image_extension) and file.startswith(prefix):
                    image_files.append((file, root))
        print(f"Found {len(image_files)} images to process...")
        records = []
        for file, root in tqdm(image_files, desc='Processing images...', unit='file'):
            image_path = Path(root) / file
            image = cv2.imread(str(image_path))
            
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(image_gray)
            if image is None:
                print(f"Failed to load the image: {file}")
                continue

            h, w = image.shape[:2]  
            image_id = Path(file).stem  
            annot_path = image_path.with_suffix('.txt')
            if not annot_path.exists():
                print(f"Annotations file not found for {file}")
                continue

            try:
                with open(annot_path, 'r') as f:
                    for line in f:
                        xmin, ymin, xmax, ymax, class_id = convert_yolo_to_xy(
                            line, w, h
                        )
                        record = {
                            'filename': file,
                            'image_id': image_id,
                            'full_path': image_path,
                            'xmax': xmax,
                            'ymax': ymax,
                            'xmin': xmin,
                            'ymin': ymin,
                            'center_x': (xmax - xmin) // 2 + xmin,
                            'center_y': (ymax - ymin) // 2+ ymin,
                            'class': class_id,
                            'image_height': int(h),
                            'image_width': int(w),
                            'brightness': brightness
                        }
                        records.append(record)
            except Exception as e:
                print(f"Error processing annotation for {file}: {e}")

        df = pd.DataFrame(records)
        return df