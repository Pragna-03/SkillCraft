import os
from PIL import Image

# Path to your dataset
dataset_path = r"C:\Users\prana\OneDrive\Desktop\SKILLCRAFT\kagglecatsanddogs_5340\PetImages"

# Step 1: Show what’s inside PetImages
print("Folders inside dataset_path:", os.listdir(dataset_path))

# Step 2: Clean corrupted images
categories = ["Cat", "Dog"]

for category in categories:
    folder = os.path.join(dataset_path, category)
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    print(f"\nChecking images in: {folder}")
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        try:
            with Image.open(img_path) as im:
                im.verify()
        except:
            print(f"Removing corrupted file: {img_path}")
            os.remove(img_path)

# Step 3: Count remaining images
for category in categories:
    folder = os.path.join(dataset_path, category)
    if os.path.exists(folder):
        print(f"{category} images remaining:", len(os.listdir(folder)))
