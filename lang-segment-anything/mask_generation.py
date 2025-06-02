from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os
import time

start_time = time.time()

# Initialize the model
model = LangSAM()

# Images
kite_walk_folder = "../DAVIS/JPEGImages/480p/kite-walk"
kite_walk_image_names = os.listdir(kite_walk_folder)

kite_images = [Image.open(os.path.join(kite_walk_folder, name)).convert("RGB") for name in kite_walk_image_names]
print("Images loaded: ", len(kite_images))

text_prompt = "kite-walk."

results = []
batch_size = 8  # Adjust batch size as needed
for i in range(0, len(kite_images), batch_size):
    batch = kite_images[i:i + batch_size]
    if not batch:
        break
    batch_results = model.predict(batch, [text_prompt] * len(batch))
    results.extend(batch_results)
    print("Processed Image: ", i, "to", i + len(batch) - 1, "of", len(kite_images))



print("Number of results: ", len(results))
print("Number of masks: ", len(results[0]["masks"]))

# Generate binary masks for each image
if not os.path.exists("./output_masks/kite-walk"):
    os.makedirs("./output_masks/kite-walk")
for i, result in enumerate(results):

    mask = results[i]["masks"][0]  # Get the first mask for each image
    bw_kw_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)

    image = Image.fromarray(bw_kw_mask)
    save_path = os.path.join("./output_masks/kite-walk", f"kite-walk_exp1_{i}.png")
    image.save(save_path)


end_time = time.time()
print(f"Processing time: {end_time - start_time:.2f} seconds")
