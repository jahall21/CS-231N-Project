from PIL import Image
from lang_sam import LangSAM

# Initialize the model
model = LangSAM()

# Load and preprocess the image
image_pil = Image.open(r"images\camel\00000.jpg").convert("RGB")
# Or: image_pil = Image.open("images/camel/00000.jpg").convert("RGB")

# Define the text prompt
text_prompt = "camel"

# Generate the mask
results = model.predict([image_pil], [text_prompt])

# Access the mask from the results
mask = results[0]['mask']

# Optionally, save or display the mask
mask.save("segmented_mask.png")