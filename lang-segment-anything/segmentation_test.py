from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os
import time

start_time = time.time()

# Initialize the model
model = LangSAM()

# Images
bear_folder = "../DAVIS/JPEGImages/480p/bear"
bear_image_names = os.listdir(bear_folder)

bear_images = [Image.open(os.path.join(bear_folder, name)).convert("RGB") for name in bear_image_names]
print("Images loaded: ", len(bear_images))

text_prompt = "bear."

results = []
batch_size = 8  # Adjust batch size as needed
for i in range(0, len(bear_images), batch_size):
    batch = bear_images[i:i + batch_size]
    if not batch:
        break
    batch_results = model.predict(batch, [text_prompt] * len(batch))
    results.extend(batch_results)
    print("Processed Image: ", i, "to", i + len(batch) - 1, "of", len(bear_images))



print("Number of results: ", len(results))
print("Number of masks: ", len(results[0]["masks"]))

# Generate binary masks for each image
if not os.path.exists("./output_masks/bear"):
    os.makedirs("./output_masks/bear")
for i, result in enumerate(results):
    # print(f"Image {i}:")
    # print(f"  Number of masks: {len(result['masks'])}")

    mask = results[i]["masks"][0]  # Get the first mask for each image
    bw_bear_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)

    bear_image = Image.fromarray(bw_bear_mask)
    save_path = os.path.join("./output_masks/bear", f"bear_{i}.png")
    bear_image.save(save_path)


end_time = time.time()
print(f"Processing time: {end_time - start_time:.2f} seconds")
