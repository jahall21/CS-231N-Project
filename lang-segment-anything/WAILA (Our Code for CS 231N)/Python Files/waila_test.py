"""
# Waila Test Script for LangSAM
# Assume that the best terms given a category are already generated in this script!
# Given an input term and input test category, this script runs LangSAM on the category and computes the best term for the category.
# It combines all masks for the best term and computes the IoU score against the ground truth masks.
"""

import numpy as np
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from lang_sam import LangSAM
import json
import itertools
import argparse

def combine_all_masks(masks):
    """
    Combine all binary masks using logical OR operation.
    """
    if not masks:
        return None
    
    combined_mask = np.logical_or.reduce(np.stack(masks) > 0.5).astype(np.uint8) * 255
    return combined_mask

def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0  # Avoid division by zero
    return intersection / union

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run LangSAM on a set of categories.")
    parser.add_argument("--best_terms", nargs="+", required=True, help="Best term to use for predictions")
    parser.add_argument("--category", type=str, required=True, help="Category to test term on")
    parser.add_argument("--root_output_folder", type=str, required=True, help="Root output folder for saving results")

    
    return parser.parse_args()


if __name__ == "__main__":

    ########################################################################################################################
    
    # VARIABLES TO EXPERIMENT WITH
    args = parse_args()
    best_terms = args.best_terms # Best terms to use for predictions from training
    category = args.category
    root_output_folder = args.root_output_folder
    os.makedirs(root_output_folder, exist_ok=True)
    category_output_folder = os.path.join(root_output_folder, category)
    os.makedirs(category_output_folder, exist_ok=True)

    print(f"Running test with best terms: {best_terms} on category: {category}")

    ########################################################################################################################

    # Initializations
    start_time = time.time()
    model = LangSAM() 

    final_results = {}

    # Loop through all categories in LIST_TEST_CATEGORIES
    category_folder = os.path.join("../DAVIS/JPEGImages/480p", category)
    category_image_names = os.listdir(category_folder)
    category_images = [Image.open(os.path.join(category_folder, name)).convert("RGB") for name in category_image_names]
    
    ground_truth_folder = os.path.join("../DAVIS/Annotations/480p", category)
    ground_truth_image_names = os.listdir(ground_truth_folder)
    ground_truth_images = [Image.open(os.path.join(ground_truth_folder, name)).convert("L") for name in ground_truth_image_names]

    ########################################################################################################################

    avg_iou_scores = []
    
    # Step 7.1
    for i, image in enumerate(category_images):  # Iterate through each frame in the category

        if (i + 1) % 20 == 0:
            print(f"Processing image {i + 1}/{len(category_images)}...")

        ground_truth_mask_i = np.array(ground_truth_images[i])  # Use the ground truth mask for the current frame

        # Step 7.1
        dict_frame_i = {}

        # For each image, run the model with the best combination of terms
        for j, term in enumerate(best_terms):
            # print(f"Processing image {i + 1}/{len(category_images)} with term {j + 1}/{len(best_combination)}: {term}")
            results_frame_i = model.predict([image], [term])  # Run prediction for each term
            dict_frame_i[term] = results_frame_i[0]  # Get first mask for each term of each frame of the category


        # Step 7.2
        # Convert masks to binary format for each term in the current frame
        bw_masks_i = []
        for term, result_i in dict_frame_i.items():

            if result_i["masks"] is None or len(result_i["masks"]) == 0:
                print(f"No mask found for term: {term}")
                continue

            mask_i = result_i["masks"][0]
            bw_mask_i = np.where(mask_i > 0.5, 255, 0).astype(np.uint8)
            bw_masks_i.append(bw_mask_i)

        # If no masks found for frame i, skip to next frame, and return IoU score of 0.0
        if len(bw_masks_i) == 0:
            print(f"No valid masks found for any terms in frame {i}. Return IoU score of 0.0 for this frame.")
            iou_score_i = 0.0
            avg_iou_scores.append(iou_score_i)
            bw_mask_i = np.zeros_like(ground_truth_mask_i, dtype=np.uint8)  # Create an empty mask if no valid masks found
            combined_mask_image = Image.fromarray(bw_mask_i)
            save_path_i = os.path.join(category_output_folder, f"combined_mask_frame_{i}.png")
            combined_mask_image.save(save_path_i)
        
        else:
            # Combine all masks associated with the best combination of terms
            combined_mask_i = combine_all_masks(bw_masks_i)

            # Step 7.3
            iou_score_i = calculate_iou(combined_mask_i > 0, ground_truth_mask_i > 0)
            avg_iou_scores.append(iou_score_i)

            # Step 7.4
            combined_mask_image = Image.fromarray(combined_mask_i)
            save_path_i = os.path.join(category_output_folder, f"combined_mask_frame_{i}.png")
            combined_mask_image.save(save_path_i)
            
            
    score_for_category = np.mean(avg_iou_scores)
    print(f"Average IoU Score for {category} with best combination of terms {best_terms}: {score_for_category:.4f}")
    final_results[category] = {
        "best_terms": best_terms,
        "average_iou_score": score_for_category,
        "iou_scores_per_frame": avg_iou_scores
    }
    # Save results for the current category
    category_results_file = os.path.join(category_output_folder, "results.json")
    os.makedirs(category_output_folder, exist_ok=True)
    with open(category_results_file, "w") as f:
        json.dump(final_results[category], f, indent=4)
    print(f"Results for category {category} saved to {category_results_file}")

    end_time = time.time()
    print(f"Time taken to prepare data: {end_time - start_time:.2f} seconds")

    # Save runtime to a text file
    runtime_file = os.path.join(root_output_folder, "runtime.txt")
    with open(runtime_file, "w") as f:
        f.write(f"Time taken to prepare data: {end_time - start_time:.2f} seconds\n")