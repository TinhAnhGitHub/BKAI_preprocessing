from PIL import ImageDraw, Image
import numpy as np
import pandas as pd



def visualize_results(original_image, df:pd.DataFrame) -> Image:
    """
    Visualize original and augmented images side by side with bounding boxes and top-left corner annotations.

    Args:
        original_image: the image in PIL
        df: contain all bounding box for annotations
       
    Returns:
        - Annotated image
    """

    
    orig_draw = ImageDraw.Draw(original_image)
    
    
    colors = {
        0: (255, 0, 0),      # Red for class 0 motorbike
        1: (0, 255, 0),      # Green for class 1 car
        2: (0, 0, 255),      # Blue for class 2 bus
        3: (255, 255, 0)     # Yellow for class 3 container
    }

    for _, row in df.iterrows():
        coords = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        color = colors.get(row['class'], (255, 255, 255)) 
        orig_draw.rectangle(coords, outline=color, width=2)

    
    return original_image