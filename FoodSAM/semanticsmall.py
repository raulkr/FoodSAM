import sys
import argparse
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import numpy as np
from typing import Any, Dict, List
import shutil, logging
import torch

# Argument parser setup
parser = argparse.ArgumentParser(
    description=(
        "Runs SAM automatic mask generation on an input image or directory of images and extracts boundaries for food items."
    )
)

parser.add_argument(
    "--img_path",
    type=str,
    default=None,
    help="Path to a single input image.",
)

parser.add_argument(
    "--output",
    type=str,
    default='Output/SAM_Results',
    help="Path to the directory where results will be output. Output will be a folder."
)

parser.add_argument(
    "--SAM_checkpoint",
    type=str,
    default="ckpts/sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--model-type",
    type=str,
    default='vit_h',
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument("--device", type=str, default="cpu", help="The device to run generation on.")

# Helper function to extract boundaries
def extract_boundaries_from_masks_with_background_removal(masks, original_image, min_contour_area=500, focus_center=True):
    combined_boundary_image = original_image.copy()
    height, width = original_image.shape[:2]
    image_area = height * width
    min_relative_area = 0.005  # Adjust as needed

    if focus_center:
        x_min, x_max = int(0.1 * width), int(0.9 * width)
        y_min, y_max = int(0.1 * height), int(0.9 * height)

    for mask in masks:
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = merge_nearby_contours(contours)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area and (area / image_area) > min_relative_area:
                x, y, w, h = cv2.boundingRect(contour)
                if not focus_center or (x_min < x < x_max and y_min < y < y_max):
                    simplified_contour = simplify_contour(contour)
                    hull = cv2.convexHull(simplified_contour)
                    cv2.drawContours(combined_boundary_image, [hull], -1, (0, 255, 0), 2)
    
    return combined_boundary_image

def simplify_contour(contour, epsilon=0.02):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon * peri, True)

def merge_nearby_contours(contours, distance_threshold=20):
    merged = []
    for contour in contours:
        if not merged:
            merged.append(contour)
        else:
            can_merge = False
            for i, existing in enumerate(merged):
                if cv2.matchShapes(contour, existing, 1, 0.0) < distance_threshold:
                    merged[i] = np.concatenate((existing, contour))
                    can_merge = True
                    break
            if not can_merge:
                merged.append(contour)
    return merged




# Main processing function
def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    # Create logger
    log_file = os.path.join(args.output, 'sam_process.log')
    logging.basicConfig(
        format='[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
    logger = logging.getLogger()
    logger.info("Running SAM mask generation...")

    # Load SAM model
    sam = sam_model_registry[args.model_type](checkpoint=args.SAM_checkpoint)
    sam.to(device=args.device)
    
    # Prepare for mask generation
    output_mode = "binary_mask"
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
    
    if args.img_path:
        targets = [args.img_path]
    else:
        logger.error("Image path not provided.")
        return
    
    for t in targets:
        logger.info(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            logger.error(f"Could not load '{t}' as an image, skipping...")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate SAM masks
        masks = generator.generate(image_rgb)
        
        # Save SAM masks (optional, but you can skip this if not needed)
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        os.makedirs(save_base, exist_ok=True)

        mask_paths = []
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            mask_filename = os.path.join(save_base, f"mask_{i}.png")
            cv2.imwrite(mask_filename, mask * 255)  # Save binary mask
            mask_paths.append(mask_filename)
        
        # Extract boundaries from all masks and save the combined boundary image
        combined_boundary_image = extract_boundaries_from_masks_with_background_removal([mask_data["segmentation"] for mask_data in masks], image, min_contour_area=500, focus_center=True)
        boundary_filename = os.path.join(save_base, f"combined_boundary.png")
        cv2.imwrite(boundary_filename, combined_boundary_image)  # Save combined boundary image

        logger.info(f"Saved combined boundary image in '{boundary_filename}'.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
