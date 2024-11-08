from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np



def detect_light_source(image, n_clusters, brightness_threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = gray > brightness_threshold
    coords = np.column_stack(np.where(masks))
    
    if len(coords)  < n_clusters:
        return [], []
    
    kmeans = KMeans(n_clusters, random_state=42)
    kmeans.fit(coords)
    
    centers = kmeans.cluster_centers_
    
    brightness_values = []
    for center in centers:
        y, x = map(int, center)
        window = gray[max(0, y-2):min(gray.shape[0], y+3), max(0, x-2): min(gray.shape[1], x+3)] # 5x5 window
        avg_brightness = np.mean(window)
        brightness_values.append(avg_brightness)
    sorted_indices = np.argsort(brightness_values)[::-1]
    sorted_centers = [(int(centers[i][1]), int(centers[i][0])) for i in sorted_indices]
    sorted_brightness = [brightness_values[i] for i in sorted_indices]
    
    return sorted_centers, sorted_brightness

def enhance_lab(img, centers, radius=20, intensity=1.2):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mask = np.zeros_like(l)
    for x, y in centers:
        cv2.circle(mask, (x, y), radius, 1, -1)
    l = l.astype(float)
    l[mask == 1] = np.clip(l[mask == 1] * intensity, 0, 255)
    lab = cv2.merge([l.astype(np.uint8), a, b])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def enhance_gaussian(img, centers, radius=20, intensity=1.2):
    result = img.copy()
    for x, y in centers:
        kernel_size = radius * 2 + 1
        gaussian = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = np.sqrt((i-radius)**2 + (j-radius)**2)
                gaussian[i,j] = np.exp(-0.5 * (dist/radius)**2)

        y1, y2 = max(0, y-radius), min(img.shape[0], y+radius+1)
        x1, x2 = max(0, x-radius), min(img.shape[1], x+radius+1)
        gy1, gy2 = radius-(y-y1), radius+(y2-y)
        gx1, gx2 = radius-(x-x1), radius+(x2-x)

        mask = gaussian[gy1:gy2, gx1:gx2]
        for c in range(3):  
            result[y1:y2, x1:x2, c] = np.clip(
                result[y1:y2, x1:x2, c] * (1 + mask * (intensity - 1)),
                0, 255
            ).astype(np.uint8)

    return result
def enhance_clahe(img, centers, radius=20):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    mask = np.zeros_like(l)
    for x, y in centers:
        cv2.circle(mask, (x, y), radius, 1, -1)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = l.copy()
    l_enhanced[mask == 1] = clahe.apply(l)[mask == 1]

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def enhance_image(image: Image.Image, centers: List[Tuple[int,int]], radius:int = 5, intensity: float = 1.2,  method: str = 'enhance_gaussian') -> Image.Image:
    img_np = np.array(image)
    if method == 'enhance_gaussian':
        return enhance_gaussian(img=image, centers=centers, radius=radius, intensity=intensity)
    elif method == 'enhance_lab':
        return enhance_lab(img=image, centers=centers, radius=radius, intensity=intensity)
    elif method == 'enhance_clahe':
        return enhance_clahe(img=image, centers=centers, radius=radius)
    return Image.fromarray(img_np.astype('uint8'))



