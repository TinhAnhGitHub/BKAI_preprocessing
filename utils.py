from typing import Tuple
import numpy as np


def is_contained(box1, box2):
    """
    Checks if box2 is fully contained within box1.
    Each box is represented as (xmin, ymin, xmax, ymax).
    """
    box1 = np.array(box1)
    box2 = np.array(box2)

    return np.all(box1[:2] <= box2[:2]) and np.all(box2[2:] <= box1[2:])

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two boxes.
    Each box is represented as (xmin, ymin, width, height).
    """
    box1 = np.array([box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]])
    box2 = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]])

    if is_contained(box1, box2) or is_contained(box2, box1):
        return 1.0

    inter_xmin = np.maximum(box1[0], box2[0])
    inter_ymin = np.maximum(box1[1], box2[1])
    inter_xmax = np.minimum(box1[2], box2[2])
    inter_ymax = np.minimum(box1[3], box2[3])

    inter_width = np.maximum(0, inter_xmax - inter_xmin)
    inter_height = np.maximum(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


