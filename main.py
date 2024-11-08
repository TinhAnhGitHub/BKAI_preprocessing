import argparse
import json
from dataset import Dataset
from augmentation import process_single_image, perform_cut_paste
from visualization import visualize_results
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from config import CutPasteConfig, get_random_configs
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from PIL import Image
import albumentations as A
import logging

logging.basicConfig(
    filename='image_augmentation.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        json_dict = json.load(f)
    return json_dict


def create_augmentation_pipeline(aug_config: Dict[str, Any]) -> A.Compose:

    transforms = []
    # if aug_config.get('basic_transform', []):
    #     basic = aug_config['basic_transform']
    #     if basic.get('horizontal_flip', 0):
    #         transforms.append(A.HorizontalFlip(p=basic['horizontal_flip']))
    #     if basic.get('vertical_flip', 0):
    #       transforms.append(A.VerticalFlip(p=basic['vertical_flip']))
    #     if basic.get('rotate', 0):
    #         transforms.append(A.Rotate(limit=basic.get('rotate_limit', 45), 
    #                               p=basic['rotate']))
    if aug_config.get('weather_transforms', {}):
        weather = aug_config['weather_transforms']
        if weather.get('random_rain', 0):
            transforms.append(A.RandomRain(p=weather['random_rain']))
        if weather.get('random_snow', 0):
            transforms.append(A.RandomSnow(p=weather['random_snow']))
        if weather.get('random_fog', 0):
            transforms.append(A.RandomFog(p=weather['random_fog']))
        if weather.get('random_shadow', 0):
            transforms.append(A.RandomShadow(p=weather['random_shadow']))

    # if aug_config.get('noise_transforms', {}):
    #     noise = aug_config['noise_transforms']
    #     if noise.get('gaussian_noise', 0):
    #         transforms.append(A.GaussNoise(p=noise['gaussian_noise']))
    #     if noise.get('motion_blur', 0):
    #         transforms.append(A.MotionBlur(p=noise['motion_blur']))

    
    return A.Compose(transforms)

def apply_augmentations(image: np.ndarray, transform: A.Compose) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """Apply Albumentations transforms to image and bounding boxes."""
    transformed = transform(
        image=image
       
    )
    return transformed['image'] 

def process_image(
    image_name: str,
    df: pd.DataFrame,
    folder_path: str,
    random_configs: List[CutPasteConfig],
    is_night_time: bool,
    output_folder: str,
    aug_pipeline: A.Compose = None
)->Optional[pd.DataFrame]:
    config = random.choice(random_configs)
    logging.info("Chose config done!")
 
    records = process_single_image(
            image_name,
            df,
            folder_path,
            config
        )
    logging.info("Processing records succesfully!")
    
    if not records:
        print(f"No records generated for {image_name}")
        return
    augmented_img = perform_cut_paste(
        records,
        is_night_time
    )
    logging.info("Cut paste done, product augmented image~")
    if aug_pipeline:
        img_np = np.array(augmented_img)
        bboxes = []
        class_labels = []
        for record in records:
            bboxes.append([record['xmin'], record['ymin'], record['xmax'], record['ymax']])
            class_labels.append(int(record['class_paste']))

        img_np = apply_augmentations(img_np, aug_pipeline)

        augmented_image = Image.fromarray(img_np)
        brightness = np.mean(augmented_image) 
        df_image = df[df['filename'] == image_name]

        image_id = df_image['image_id'].iloc[0]
        image_height, image_width = df_image['image_height'].iloc[0], df_image['image_width'].iloc[0]
        image_path = df_image['full_path'].iloc[0]
        new_records = []
        old_image_name = image_name
        image_name = "modified_"+ image_name
        image_id = "modified_" + image_id
        for record in records:
            new_record = {
                'filename': image_name,
                'image_id': image_id,
                'full_path': image_path,
                'xmax': record['xmax'],
                'ymax': record['ymax'],
                'xmin': record['xmin'],
                'ymin': record['ymin'],
                'center_x': record['xmax'] - record['xmin'] // 2 + record['xmin'],
                'center_y': record['ymax'] - record['ymin'] // 2 + record['ymin'],
                'class': record['class_paste'],
                'image_height': image_height,
                'image_width': image_width,
                'brightness': brightness
            }
            new_records.append(new_record)
        new_df = pd.DataFrame(new_records)
        df = pd.concat([df, new_df], ignore_index=True)

        annot_aug = visualize_results(
            augmented_img.copy(),
            df[(df['filename'] == old_image_name) | (df['filename'] == image_name)].copy()
        )
        
        output_image_path = Path(output_folder) / 'image' / image_name
        output_annot_path = Path(output_folder) / 'image_annote' / image_name

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        output_annot_path.parent.mkdir(parents=True, exist_ok=True)

        augmented_img.save(output_image_path)
        annot_aug.save(output_annot_path)
        return df


def main():
    parser = argparse.ArgumentParser(description='Image augmentation pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration JSON file')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Input dataset folder path')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Output folder path for augmented images')
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_folder, exist_ok=True)

    aug_pipeline = None
    if 'augmentations' in config:
        aug_pipeline = create_augmentation_pipeline(config['augmentations'])
    
    cam_groups = config.get('cam_groups', [f'cam_{str(i).zfill(2)}' for i in range(1, 11)])
    night_time_cams = config.get('night_time_cams', ['cam_03', 'cam_05', 'cam_08'])

    random_configs = get_random_configs(
        num_configs=config.get('num_random_configs', 100),
    )
    all_updated_dfs = [] 

    for cam_group in cam_groups:
        print(f"\nProcessing {cam_group}...")
        dataset = Dataset.from_folder(args.input_folder, cam_group)
        df = dataset.dataframe
        is_night_time = cam_group in night_time_cams

        image_names = df['filename'].unique()
        

        updated_dfs = Parallel(n_jobs=-1)(
            delayed(process_image)(
                image_name,
                df,
                args.input_folder,
                random_configs,
                is_night_time,
                args.output_folder,
                aug_pipeline
            ) for image_name in tqdm(image_names, desc=f'Processing {cam_group}')
        )
        
    
    if all_updated_dfs:
        final_df = pd.concat(all_updated_dfs, ignore_index=True)
        output_csv_path = Path(args.output_folder) / 'all_annotations.csv'
        final_df.to_csv(output_csv_path, index=False)
        print(f"Saved all annotations to {output_csv_path}")


if __name__ == "__main__":
    main()







    
