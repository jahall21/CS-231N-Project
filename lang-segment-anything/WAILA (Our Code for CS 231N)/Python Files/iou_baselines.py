import numpy as np
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted # type: ignore

def load_masks(folder_path):
    """
    Load all masks from a specified folder.
    """
    mask_files = natsorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    masks = [(np.array(Image.open(os.path.join(folder_path, f)).convert("L")), f) for f in mask_files]
    
    return masks

def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0  # Avoid division by zero
    return intersection / union

def evaluate_iou(masks_folder, ground_truth_mask_folder):
    """
    Evaluate IoU for all masks in the specified folder against the ground truth masks.
    """
    masks = load_masks(masks_folder)
    ground_truth_masks = load_masks(ground_truth_mask_folder)
    
    if len(masks) != len(ground_truth_masks):
        raise ValueError("Number of masks in the folder does not match the number of ground truth masks.")
    
    iou_results = []
    
    for i, mask_tuple in enumerate(masks):
        mask = mask_tuple[0]  # Get the mask array
        gt_mask = ground_truth_masks[i][0]  # Get the ground truth mask array

        iou_score = calculate_iou(mask > 0, gt_mask > 0)  # Convert to binary
        iou_results.append((os.path.basename(os.listdir(masks_folder)[i]), iou_score))
    
    return iou_results

def run_iou(masks_folder, ground_truth_mask_folder, category):
    
    iou_results = evaluate_iou(masks_folder, ground_truth_mask_folder)

    _, iou_scores = zip(*iou_results)

    iou_score = np.mean(iou_scores)
    print(f"Average IoU Score for {category}: {iou_score:.4f}")

    return iou_score

if __name__ == "__main__":
    # Example usage
    # masks_folder = "./output_masks/bear"  # Adjust this path as needed
    # ground_truth_mask_folder = "../DAVIS/Annotations/480p/bear/"  # Adjust this path as needed
    # category = "bear"

    start_time = time.time()

    directory_masks = "./output_masks/"
    directory_ground_truth = "../DAVIS/Annotations/480p/"

    results_per_category = {}

    num_categories = len(os.listdir(directory_masks))

    for i, category in enumerate(os.listdir(directory_masks)):

        print(f"Processing category {i + 1}/{num_categories}: {category}")

        masks_folder = os.path.join(directory_masks, category)
        ground_truth_mask_folder = os.path.join(directory_ground_truth, category)

        if not os.path.isdir(masks_folder) or not os.path.isdir(ground_truth_mask_folder):
            print(f"Skipping {category} as it is not a valid directory.")
            continue

        score = run_iou(masks_folder, ground_truth_mask_folder, category)

        results_per_category[category] = score

    sorted_results = sorted(results_per_category.items(), key=lambda x: x[1], reverse=True)  # Sort results in descending order by IoU score

    for category, score in sorted_results:
        print("Category:", category, " IoU Score:", score)
    
    # Save results
    with open("iou_results.txt", "w") as f:
        for category, score in sorted_results:
            f.write(f"{category}: {score:.4f}\n")

    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
    
    