# config.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random

@dataclass
class CutPasteConfig:
  """Configuration for cut-paste operations."""
  class_priority: List[int]
  objects_per_class: Dict[int, Tuple[int, int]]
  k_nearest: int
  iou_range: Tuple[float, float]
  n_jobs: int = -1

def get_random_configs(num_configs: int = 10) -> List[CutPasteConfig]:
  """Generate a list of random cut-paste configurations."""
  configs = []
  for _ in range(num_configs):
      config = CutPasteConfig(
          class_priority=[3, 2, 1, 0],
          objects_per_class={
              0: (0, random.randint(25, 40)),  
              1: (0, random.randint(20, 35)),  
              2: (0, random.randint(0, 5)),  
              3: (0, random.randint(0, 1)),   
          },
          k_nearest=random.randint(10, 50),
          iou_range=(0.0, 0.0),
          n_jobs=-1,
      )
      configs.append(config)
  return configs