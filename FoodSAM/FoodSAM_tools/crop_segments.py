import os
import mmcv
import numpy as np
from skimage import measure
import cv2

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0

def non_max_suppression(boxes, scores, iou_threshold):
    """Perform Non-Maximum Suppression on bounding boxes."""
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        
        iou = np.array([calculate_iou(boxes[i], boxes[j]) for j in indices[1:]])
        indices = indices[1:][iou < iou_threshold]
    
    return keep

def crop_and_save_segments(img_path, mask_path, output_dir):
    """Crop segments from an image based on a segmentation mask and save them."""
    img = mmcv.imread(img_path)
    seg = mmcv.imread(mask_path, flag='grayscale')
    
    unique_labels = np.unique(seg)
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        
        binary_mask = (seg == label).astype(np.uint8)
        contours = measure.find_contours(binary_mask, 0.5)
        
        boxes = []
        crops = []
        areas = []
        
        for contour in contours:
            contour = np.fliplr(contour).astype(np.int32)
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 32 or h < 32:  # Minimum size threshold
                continue
            
            cropped_img = img[y:y+h, x:x+w]
            boxes.append([x, y, w, h])
            crops.append(cropped_img)
            areas.append(w * h)
        
        # Perform NMS
        keep = non_max_suppression(boxes, areas, iou_threshold=0.5)
        
        for i, idx in enumerate(keep[:5]):  # Limit to 5 crops per label
            crop_filename = f"crop_label_{label}_contour_{i}.png"
            mmcv.imwrite(crops[idx], os.path.join(output_dir, 'crops', crop_filename))

def process_semantic_results(base_dir):
    """Process semantic segmentation results and create cropped segments."""
    for root, dirs, files in os.walk(base_dir):
        if 'pred_mask.png' in files and 'input.jpg' in files:
            img_path = os.path.join(root, 'input.jpg')
            mask_path = os.path.join(root, 'pred_mask.png')
            crops_dir = os.path.join(root, 'crops')
            os.makedirs(crops_dir, exist_ok=True)
            crop_and_save_segments(img_path, mask_path, root)