import numpy as np
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from lang_sam import LangSAM
from natsort import natsorted # type: ignore
import json
import itertools

LIST_TEST_CATEGORIES = [
        "scooter-gray", "drift-chicane", "parkour", "paragliding-launch", "stroller",
        "motorbike", "kite-walk", "motocross-bumps", "drift-straight", "rollerblade",
        "bmx-bumps", "kite-surf", "hockey", "bmx-trees", "swing",
        "surf", "dance-jump", "hike", "mallard-water", "soapbox"
    ]


def choose_n_terms(similarity_dict, n=5):
    """
    Choose the top n terms based on similarity scores.
    """
    list_of_tuples = similarity_dict
    n_tuples = list_of_tuples[:n]

    terms = []

    for term, _ in n_tuples:
        terms.append(term)
    return terms

def choose_frames_to_sample(total_num_frames, num_frames_to_sample=3):
    """
    Choose frames to sample based on the total number of frames.
    """
    if total_num_frames <= num_frames_to_sample:
        return list(range(total_num_frames))
    
    step = total_num_frames // num_frames_to_sample
    return list(range(0, total_num_frames, step))[:num_frames_to_sample]

def generate_all_mask_combinations(masks, terms):
    """
    Generate all possible combinations of binary masks using logical OR operation.
    """

    if len(masks) != len(terms):
        raise ValueError("The number of masks must match the number of terms.")

    all_combinations = []
    
    # Iterate over all possible combinations
    for r in range(1, len(masks) + 1):
        for indices in itertools.combinations(range(len(masks)), r):
            selected_masks = [masks[i] for i in indices]
            selected_terms = [terms[i] for i in indices]
            
            combined_mask = np.logical_or.reduce(np.stack(selected_masks) > 0.5).astype(np.uint8) * 255
            all_combinations.append((combined_mask, selected_terms))

    return all_combinations

def combine_all_masks(masks):
    """
    Combine all binary masks using logical OR operation.
    """
    if not masks:
        return None
    
    combined_mask = np.logical_or.reduce(np.stack(masks) > 0.5).astype(np.uint8) * 255
    return combined_mask


def compute_best_iou(mask_combinations, ground_truth_mask):
    """
    Compute the best IoU score for each mask combination against the ground truth mask.
    """
    best_iou = 0
    best_combination = None

    for mask, terms in mask_combinations:
        iou_score = calculate_iou(mask > 0, ground_truth_mask > 0)  # Convert to binary
        # print(f"Terms: {terms}, IoU: {iou_score:.4f}")
        if iou_score > best_iou:
            best_iou = iou_score
            best_combination = terms

    return best_combination, best_iou

def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0  # Avoid division by zero
    return intersection / union




if __name__ == "__main__":

    ########################################################################################################################
    
    # VARIABLES TO EXPERIMENT WITH
    n_terms = 4  # Number of similar terms to retrieve (Step 1)
    n_samples = 3  # Number of frames to sample from the category (Step 2)
    word_model = "w2v_similar_words.json"

    ########################################################################################################################

    # Initializations
    start_time = time.time()
    model = LangSAM() 

    # Step 0 (Load in category of interest and ground truth images)
    print("Loading category images for category: ", LIST_TEST_CATEGORIES[0])
    category = LIST_TEST_CATEGORIES[0]

    category_folder = os.path.join("../DAVIS/JPEGImages/480p", category)
    category_image_names = os.listdir(category_folder)
    category_images = [Image.open(os.path.join(category_folder, name)).convert("RGB") for name in category_image_names]
    
    ground_truth_folder = os.path.join("../DAVIS/Annotations/480p", category)
    ground_truth_image_names = os.listdir(ground_truth_folder)
    ground_truth_images = [Image.open(os.path.join(ground_truth_folder, name)).convert("L") for name in ground_truth_image_names]

    ########################################################################################################################

    # Step 1 (Retrieve similar terms for the category)
    dict_of_test_categories = {}
    with open(word_model, "r") as file:
        dict_of_test_categories = json.load(file)

    category_dict = dict_of_test_categories[category]
    similar_terms = choose_n_terms(category_dict, n_terms)
    print("Similar terms: ", similar_terms)
    
    ########################################################################################################################

    # Step 2 (Sample frames from the category)
    total_num_frames = len(category_images)
    frames_to_sample = choose_frames_to_sample(total_num_frames)
    print("Frames to sample: ", frames_to_sample)
    sampled_images = [category_images[i] for i in frames_to_sample]

    ########################################################################################################################

    # Step 3 (Run predictions on sampled images, for now just use the first sampled image, will need to change
    print("Running predictions on sampled images...")

    dict_of_results = {}
    for i, term in enumerate(similar_terms):
        results = model.predict([sampled_images[0]], [term])
        dict_of_results[term] = results[0]
        print(f"Processed term {i + 1}/{len(similar_terms)}: {term}")

    ########################################################################################################################

    # Step 4 (Save the first mask for each term)

    root_output_folder = "./output_masks_test_1"
    category_output_folder = os.path.join(root_output_folder, category)

    # Create both folders (if they don't already exist)
    os.makedirs(category_output_folder, exist_ok=True)


    bw_masks = []  # List of binary masks for combination for later
    for term, result in dict_of_results.items():
        mask = result["masks"][0] # Get the first mask for the term
        bw_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
        bw_masks.append(bw_mask)
        mask_image = Image.fromarray(bw_mask)
        save_path = os.path.join(category_output_folder, f"{term}_mask.png")
        mask_image.save(save_path)
    print(f"Saved masks to {category_output_folder}")

    ########################################################################################################################

    # Step 5 (Iterate through all combinations of masks for each term)
    print("Combining masks...")
    mask_combinations = generate_all_mask_combinations(bw_masks, similar_terms)
    print("Number of mask combinations generated: ", len(mask_combinations))

    # print("Mask combination terms: ", [terms for _, terms in mask_combinations])
    
    ########################################################################################################################

    # Step 6 (Initial IoU step for every mask combination)
    print("Computing IoU for each mask combination...")
    ground_truth_mask = np.array(ground_truth_images[frames_to_sample[0]])  # Use the first ground truth image for IoU calculation
    best_combination, best_iou = compute_best_iou(mask_combinations, ground_truth_mask)
    print(f"Best IoU: {best_iou:.4f} occured with terms: {best_combination}")

    ########################################################################################################################

    # Step 7 (For every frame)
    #   7.1.    Run predictions with all frames in the category with new best combination
    #   7.2.    Combine all masks associated with the best combination of terms
    #   7.3.    Achieving Avg IoU Score across all frames to compare to baseline)
    #   7.4.    Save the masks for each frame in the category
    print("Running predictions on all sampled images with the best combination of terms...")

    avg_iou_scores = []

    # Step 7.1
    for i, image in enumerate(category_images):

        ground_truth_mask_i = np.array(ground_truth_images[i])  # Use the ground truth mask for the current frame

        # Step 7.1
        dict_frame_i = {}

        # For each image, run the model with the best combination of terms
        for j, term in enumerate(best_combination):
            print(f"Processing image {i + 1}/{len(category_images)} with term {j + 1}/{len(best_combination)}: {term}")
            results_frame_i = model.predict([image], [term])  # Run prediction for each term
            dict_frame_i[term] = results_frame_i[0]  # Get first mask for each term of each frame of the category


        # Step 7.2
        # Convert masks to binary format for each term in the current frame
        bw_masks_i = []
        for term, result_i in dict_frame_i.items():
            mask_i = result_i["masks"][0]
            bw_mask_i = np.where(mask_i > 0.5, 255, 0).astype(np.uint8)
            bw_masks_i.append(bw_mask_i)

        # Combine all masks associated with the best combination of terms
        combined_mask_i = combine_all_masks(bw_masks_i)
        

        # Step 7.3
        iou_score_i = calculate_iou(combined_mask_i > 0, ground_truth_mask_i > 0)
        print("IoU score for frame", i + 1, ":", iou_score_i)
        avg_iou_scores.append(iou_score_i)


        # Step 7.4
        combined_mask_image = Image.fromarray(combined_mask_i)
        save_path_i = os.path.join(category_output_folder, f"combined_mask_frame_{i}.png")
        combined_mask_image.save(save_path_i)
    
    score_for_sample_z = np.mean(avg_iou_scores)
    print(f"Average IoU Score for sampled frames with best combination of terms {best_combination}: {score_for_sample_z:.4f}")

    ########################################################################################################################


    end_time = time.time()
    print(f"Time taken to prepare data: {end_time - start_time:.2f} seconds")