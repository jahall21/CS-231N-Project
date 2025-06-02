from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os
import time

def generate_masks_for_all_folders(base_folder, output_base_folder, batch_size=8):
    start_time = time.time()

    # Initialize the SAM model
    model = LangSAM()

    total_images_processed = 0  # Running total of processed images
    multi_mask_dict = {}

    # Iterate over each category folder under the base folder
    for category in os.listdir(base_folder):

        category_folder = os.path.join(base_folder, category)
        if not os.path.isdir(category_folder):
            continue

        # Load all images from the current category folder
        image_names = os.listdir(category_folder)
        images = []
        for name in image_names:
            image_path = os.path.join(category_folder, name)
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        if not images:
            continue

        print(f"Processing folder '{category}' with {len(images)} images.")

        # Use the folder name as a text prompt
        text_prompt = category + "."

        results = []
        # Process the images in batches to optimize prediction speed
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            # Generate predictions for the batch with the same text prompt for every image
            batch_results = model.predict(batch, [text_prompt] * len(batch))
            results.extend(batch_results)
            print(f"Processed images {i} to {i + len(batch) - 1} out of {len(images)} for '{category}'.")

        multi_mask_images = [r for r in results if len(r["masks"]) > 1]
        if multi_mask_images:
            multi_mask_dict[category] = len(multi_mask_images)
            print(f"Category '{category}' has {len(multi_mask_images)} image(s) with multiple masks. May need to investigate further.")
        
        total_images_processed += len(images)
        print(f"Total images processed so far: {total_images_processed}")

        # Create the output folder corresponding to the current category
        output_folder = os.path.join(output_base_folder, category)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Process and save each result as a binary mask
        for i, result in enumerate(results):
            save_path = os.path.join(output_folder, f"{category}_exp1_{i}.png")

            # If no masks are found, create an empty mask and save
            if np.size(result["masks"]) == 0:
                width, height = images[i].size  # PIL uses (width, height) format
                print(f"No masks found for image {i} in category '{category}'. Skipping.")
                empty_mask = np.zeros((height, width), dtype=np.uint8)
                empty_image = Image.fromarray(empty_mask)
                empty_image.save(save_path)

            else:
                # Retrieves first mask for each image for simplicity
                mask = result["masks"][0]
                # Generate a binary mask: pixel values above 0.5 become 255, otherwise 0
                bw_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
                mask_image = Image.fromarray(bw_mask)
                mask_image.save(save_path)

        print(f"Saved {len(results)} masks for folder '{category}'.\n")

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    for category, count in multi_mask_dict.items():
        print(f"{category}: {count} image(s) with multiple masks")


if __name__ == "__main__":
    base_folder = "../DAVIS/JPEGImages/480p"
    output_base_folder = "./output_masks"

    generate_masks_for_all_folders(base_folder, output_base_folder, batch_size=8)
