from typing import List, Dict, Optional
from PIL import Image
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from utils import calculate_iou
from config import CutPasteConfig
from sklearn.neighbors import NearestNeighbors
from enhancement import enhance_image, detect_light_source
import os



def process_single_image(image_name: str, 
                      df: pd.DataFrame, 
                      folder_path: str, 
                      config: CutPasteConfig,
                      existing_records: List[Dict] = None,
                      max_attempts: int = 100)-> List[Dict]:
    """
    Process cut-paste operations for a single image with enhanced debugging
    """

    
    records = []
    
    df_image = df[df['filename'] == image_name]
    df_available = df[df['filename'] != image_name]
   
    
    target_brightness = df_image['brightness'].iloc[0]
    df_available = df_available.assign(brightness_diff=abs(df_available['brightness'] - target_brightness))
    df_available = df_available.sort_values('brightness_diff')
    
    list_of_center_x_y_image = df_image[['center_x', 'center_y']].apply(tuple, axis=1).tolist()
    
    
    if existing_records:
        for record in existing_records:
            center_x = (record['xmax'] + record['xmin']) / 2
            center_y = (record['ymax'] + record['ymin']) / 2
            list_of_center_x_y_image.append((center_x, center_y))
    
    for class_id in config.class_priority:
        if class_id not in config.objects_per_class:
            continue
            
        min_objects, max_objects = config.objects_per_class[class_id]
        num_objects_added = 0
        attempts = 0
        available_objects = df_available[df_available['class'] == class_id]
        
        while num_objects_added < max_objects and attempts < max_attempts:
            attempts += 1
            
            if available_objects.empty:
                break
            
            if class_id == 0.0:  
                image_points = np.array(list_of_center_x_y_image)
                available_centers = available_objects[['center_x', 'center_y']].values
                k = min(config.k_nearest, len(image_points))
                
                knn = NearestNeighbors(n_neighbors=k)
                knn.fit(image_points)
                _, indices = knn.kneighbors(available_centers)
                
                for idx in range(len(available_centers)):
                    neighbor_indices = indices[idx]
                    valid_placement = True
                    
                    current_obj = available_objects.iloc[idx]
                    box1 = (
                        current_obj['xmin'],
                        current_obj['ymin'],
                        current_obj['xmax'] - current_obj['xmin'],
                        current_obj['ymax'] - current_obj['ymin']
                    )
                    
                    for neighbor_idx in neighbor_indices:
                        neighbor_point = tuple(image_points[neighbor_idx])
                        
                        if neighbor_point in list_of_center_x_y_image:
                            neighbor_index = list_of_center_x_y_image.index(neighbor_point)
                            neighbor_obj = df_image.iloc[neighbor_index]
                            box2 = (
                                neighbor_obj['xmin'],
                                neighbor_obj['ymin'],
                                neighbor_obj['xmax'] - neighbor_obj['xmin'],
                                neighbor_obj['ymax'] - neighbor_obj['ymin']
                            )
                            iou = calculate_iou(box1, box2)
                            
                            if not (config.iou_range[0] <= iou <= config.iou_range[1]):
                                valid_placement = False
                                break
                    
                    if valid_placement:
                        placement_found = True
                        break
                        
            else:  
                for idx in range(len(available_objects)):
                    valid_placement = True
                    current_obj = available_objects.iloc[idx]
                    box1 = (
                        current_obj['xmin'],
                        current_obj['ymin'],
                        current_obj['xmax'] - current_obj['xmin'],
                        current_obj['ymax'] - current_obj['ymin']
                    )
                    
                    for _, existing_obj in df_image.iterrows():
                        box2 = (
                            existing_obj['xmin'],
                            existing_obj['ymin'],
                            existing_obj['xmax'] - existing_obj['xmin'],
                            existing_obj['ymax'] - existing_obj['ymin']
                        )
                        iou = calculate_iou(box1, box2)
                        
                        if not (config.iou_range[0] <= iou <= config.iou_range[1]):
                            valid_placement = False
                            break
                    
                    if valid_placement:
                        placement_found = True
                        break
            
            
            full_path = df_image['full_path'].iloc[0]
            middle_path = Path(full_path).parent
            if valid_placement:
                record = {
                    'image_origin_path': str(Path(middle_path) / image_name),
                    'image_name_from_paste_source': str(Path(middle_path) / current_obj['filename']),
                    'xmin': current_obj['xmin'],
                    'ymin': current_obj['ymin'],
                    'xmax': current_obj['xmax'],
                    'ymax': current_obj['ymax'],
                    'class_paste': class_id
                }
                records.append(record)
                new_center_x = int((current_obj['xmax'] + current_obj['xmin']) / 2)
                new_center_y = int((current_obj['ymax'] + current_obj['ymin']) / 2)
            
                list_of_center_x_y_image.append((new_center_x, new_center_y))
                new_object_df = pd.DataFrame([{
                        'filename': image_name,
                        'image_id': Path(image_name).stem,
                        'xmax': current_obj['xmax'],
                        'ymax': current_obj['ymax'],
                        'xmin': current_obj['xmin'],
                        'ymin': current_obj['ymin'],
                        'center_x': new_center_x,
                        'center_y': new_center_y,
                        'class': class_id,
                        'image_height': df_image['image_height'].iloc[0], 
                        'image_width': df_image['image_width'].iloc[0]     
                }])

                df_image = pd.concat([df_image, new_object_df], ignore_index=True)
                num_objects_added += 1
                available_objects = available_objects.drop(available_objects.index[idx])
            else:
                ...
                    
            if num_objects_added >= max_objects:
                break
    return records

def perform_cut_paste(records, is_night: bool):
    """
    Perform cut-paste operations based on the provided records.
    
    Args:
        records: List of dictionaries containing cut-paste information
    
    Returns:
        Augmented image with pasted objects
    """
    target_img = Image.open(records[0]['image_origin_path'])
    target_img = target_img.convert('RGB')  
    for i, record in enumerate(records):
        source_img = Image.open(record['image_name_from_paste_source'])
        source_img = source_img.convert('RGB')
        
        x1, y1, x2, y2 = (
            int(record['xmin']), 
            int(record['ymin']), 
            int(record['xmax']), 
            int(record['ymax'])
        )
        cropped_obj = source_img.crop((x1, y1, x2, y2))        
        class_id = record['class_paste']
        if is_night:
            sorted_centers, _ = detect_light_source(np.array(cropped_obj), 1 if class_id == 0.0 else 2)
            image_en = enhance_image(
                np.array(cropped_obj),
                sorted_centers,
                5,
                1.2
            )
            cropped_obj = Image.fromarray(np.uint8(image_en))
        
        mask = Image.new('L', cropped_obj.size, 255)  
        target_img.paste(cropped_obj, (x1, y1), mask)
    return target_img








