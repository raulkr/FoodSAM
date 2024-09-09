import cv2
import numpy as np
import os
import logging

def calculate_single_image_masks_label(mask_file, pred_mask_file, category_list, sam_mask_label_file_name, sam_mask_label_file_dir):
    """
 mask_index, category_id, category_name, category_count, mask_count
    """
    sam_mask_data = np.load(mask_file)
    pred_mask_img = cv2.imread(pred_mask_file)[:,:,-1] # red channel
    shape_size = pred_mask_img.shape[0] * pred_mask_img.shape[1]
    logger = logging.getLogger()
    folder_path = os.path.dirname(pred_mask_file)
    sam_mask_category_folder = os.path.join(folder_path, sam_mask_label_file_dir)
    os.makedirs(sam_mask_category_folder, exist_ok=True)
    mask_category_path = os.path.join(sam_mask_category_folder, sam_mask_label_file_name)
    with open(mask_category_path, 'w') as f:
        f.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        for i in range(sam_mask_data.shape[0]):
            single_mask = sam_mask_data[i]
            single_mask_labels = pred_mask_img[single_mask]
            unique_values, counts = np.unique(single_mask_labels, return_counts=True, axis=0)
            max_idx = np.argmax(counts)
            single_mask_category_label = unique_values[max_idx]
            count_ratio = counts[max_idx]/counts.sum()

            logger.info(f"{folder_path}/sam_mask/{i} assign label: [ {single_mask_category_label}, {category_list[single_mask_category_label]}, {count_ratio:.2f}, {counts.sum()/shape_size:.4f} ]")
            f.write(f"{i},{single_mask_category_label},{category_list[single_mask_category_label]},{count_ratio:.2f},{counts.sum()/shape_size:.4f}\n")

    f.close()


def predict_sam_label(data_folder, category_txt,
                      masks_path_name="sam_mask/masks.npy",
                      sam_mask_label_file_name="sam_mask_label.txt",
                      pred_mask_file_name="pred_mask.png",
                      sam_mask_label_file_dir="sam_mask_label"):

    category_lists = []
    with open(category_txt, 'r') as f:
        category_lines = f.readlines()
        category_list = [' '.join(line_data.split('\t')[1:]).strip() for line_data in category_lines]
        f.close()
        category_lists.append(category_list)
    
    for test_path, category_list in zip(data_folder, category_lists):
        img_ids = os.listdir(test_path)
        for img_id in img_ids:
            mask_file_path = os.path.join(test_path, img_id, masks_path_name)
            pred_mask_file_path = os.path.join(test_path, img_id, pred_mask_file_name)
            if os.path.exists(mask_file_path) and os.path.exists(pred_mask_file_path):
                calculate_single_image_masks_label(mask_file_path, pred_mask_file_path, category_list, sam_mask_label_file_name, sam_mask_label_file_dir)




def visualization_save(mask, save_path, img_path, color_list):
    values = set(mask.flatten().tolist())
    final_masks = []
    label = []
    for v in values:
        final_masks.append((mask[:,:] == v, v))
    np.random.seed(42)
    if len(final_masks) == 0:
        return
    h, w = final_masks[0][0].shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8) 
    for m, label in final_masks:
        result[m, :] = color_list[label] 
    image = cv2.imread(img_path)
    vis = cv2.addWeighted(image, 0.5, result, 0.5, 0) 
    cv2.imwrite(save_path, vis)

def create_boundary_image(enhanced_mask_path, original_image_path):
    # Read the enhanced mask
    enhanced_mask = cv2.imread(enhanced_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Read the original image
    original_image = cv2.imread(original_image_path)
    
    # Apply Canny edge detection
    edges = cv2.Canny(enhanced_mask, 50, 150)
    
    # Dilate the edges to make them more visible
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image
    boundary_image = original_image.copy()
    
    # Draw all contours
    cv2.drawContours(boundary_image, contours, -1, (0, 255, 0), 2)
    
    return boundary_image

def enhance_masks(data_folder, category_txt, color_list_path, num_class=104, area_thr=0, ratio_thr=0.5, top_k=80,
                  masks_path_name="sam_mask/masks.npy",
                  new_mask_label_file_name="semantic_masks_category.txt",
                  pred_mask_file_name="pred_mask.png",
                  enhance_mask_name='enhance_mask.png',
                  enhance_mask_vis_name='enhance_vis.png',
                  boundary_mask_name='boundary_vis.png',
                  sam_mask_label_file_dir='sam_mask_label'):
        
    predict_sam_label([data_folder], category_txt, masks_path_name, new_mask_label_file_name, pred_mask_file_name, sam_mask_label_file_dir)
    color_list = np.load(color_list_path)
    color_list[0] = [238, 239, 20]
    for img_folder in os.listdir(data_folder):
        if img_folder.startswith('.') or not os.path.isdir(os.path.join(data_folder, img_folder)):
            continue  # Skip hidden files and non-directories
        
        category_info_path = os.path.join(data_folder, img_folder, sam_mask_label_file_dir, new_mask_label_file_name)
        sam_mask_folder = os.path.join(data_folder, img_folder)
        pred_mask_path = os.path.join(data_folder, img_folder, pred_mask_file_name)
        img_path = os.path.join(data_folder, img_folder, 'input.jpg')
        
        if not os.path.exists(pred_mask_path) or not os.path.exists(img_path):
            print(f"Skipping {img_folder}: Required files not found.")
            continue
        
        save_dir = os.path.join(data_folder, img_folder)
        # Check if save_dir is a file, and if so, create a unique directory name
        suffix = 1
        while os.path.isfile(save_dir):
            save_dir = os.path.join(data_folder, f"{img_folder}_enhanced_{suffix}")
            suffix += 1
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, enhance_mask_name)
        vis_save_path = os.path.join(save_dir, enhance_mask_vis_name)
        boundary_save_path = os.path.join(save_dir, boundary_mask_name)

        pred_mask = cv2.imread(pred_mask_path)
        if pred_mask is None:
            print(f"Failed to read pred_mask for {img_folder}. Skipping.")
            continue
        pred_mask = pred_mask[:,:,2]

        if not os.path.exists(category_info_path):
            print(f"Category info file not found for {img_folder}. Skipping.")
            continue
        
        with open(category_info_path, 'r') as f:
            category_info = f.readlines()[1:]
        
        category_area = np.zeros((num_class,))
        for info in category_info:
            parts = info.split(',')
            if len(parts) < 5:
                continue  # Skip malformed lines
            label, area = int(parts[1]), float(parts[4])
            category_area[label] += area

        category_info = sorted(category_info, key=lambda x: float(x.split(',')[4]) if len(x.split(',')) > 4 else 0, reverse=True)
        category_info = category_info[:top_k]
        
        enhanced_mask = pred_mask
        
        sam_masks_path = os.path.join(sam_mask_folder, masks_path_name)
        if not os.path.exists(sam_masks_path):
            print(f"SAM masks not found for {img_folder}. Skipping.")
            continue
        
        sam_masks = np.load(sam_masks_path)
        for info in category_info:
            parts = info.split(',')
            if len(parts) < 5:
                continue  # Skip malformed lines
            idx, label, count_ratio, area = int(parts[0]), int(parts[1]), float(parts[3]), float(parts[4])
            if area < area_thr or count_ratio < ratio_thr:
                continue
            sam_mask = sam_masks[idx].astype(bool)
            assert (sam_mask.sum() / (sam_mask.shape[0] * sam_mask.shape[1]) - area) < 1e-4
            enhanced_mask[sam_mask] = label
        
        cv2.imwrite(save_path, enhanced_mask)
        visualization_save(enhanced_mask, vis_save_path, img_path, color_list)
        
        # Create and save the boundary image using the enhanced mask
        boundary_image = create_boundary_image(save_path, img_path)
        cv2.imwrite(boundary_save_path, boundary_image)

    print("Enhance masks process completed.")