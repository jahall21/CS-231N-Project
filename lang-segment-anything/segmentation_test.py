from PIL import Image
from lang_sam import LangSAM

# Initialize the model
model = LangSAM()

image_pil = Image.open("./assets/car.jpeg").convert("RGB")
image_2_pil = Image.open("./assets/food.jpg").convert("RGB")

text_prompt = "wheel."

results = model.predict([image_pil, image_2_pil], [text_prompt])
print("Results masks:", results[0]["masks"])
print("Results Masks Scores:", results[0]["mask_scores"])
