import os
import shutil
import cv2  # pip install opencv-python

# --- CONFIGURATION ---
source_folder = "unlabelled_images"  # Folder where your raw images are
output_base = "dataset"              # Folder where sorted images will go

# Key mapping: Press 1-5 to move image to the corresponding folder
classes = {
    ord('1'): "Altered Rock",
    ord('2'): "Bedrock",
    ord('3'): "Dunes",
    ord('4'): "Loose Rock",
    ord('5'): "Sedimentary Rock"
}
# ---------------------

# 1. Create the destination folders if they don't exist
if not os.path.exists(source_folder):
    print(f"ERROR: Please create the folder '{source_folder}' and put your images in it.")
    exit()

for class_name in classes.values():
    path = os.path.join(output_base, class_name)
    os.makedirs(path, exist_ok=True)
    print(f"Created/Verified folder: {path}")

# 2. Get list of images
all_files = os.listdir(source_folder)
images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

print(f"\n--- FOUND {len(images)} IMAGES TO SORT ---")
print("Controls: [1-5] to Sort | [s] to Skip | [q] to Quit")
print("---------------------------------------------------")

count = 0
for image_file in images:
    src_path = os.path.join(source_folder, image_file)
    
    # Attempt to read image
    img = cv2.imread(src_path)
    if img is None:
        print(f"Could not read {image_file}, skipping...")
        continue

    # Resize for easier viewing on screen (fits mostly any laptop screen)
    display_img = cv2.resize(img, (800, 600))

    # Add an overlay so you don't forget the keys
    # Text format: "1:Altered 2:Bedrock 3:Dunes 4:Loose 5:Sedimentary"
    overlay_text = "1:Alt 2:Bed 3:Dune 4:Loose 5:Sed"
    
    # Draw a black rectangle background for text readability
    cv2.rectangle(display_img, (0, 0), (800, 50), (0, 0, 0), -1)
    cv2.putText(display_img, overlay_text, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Mars Speed Labeler", display_img)
    
    # Wait for key press
    key = cv2.waitKey(0)

    # Logic
    if key == ord('q'):
        print("Quitting...")
        break
    
    elif key == ord('s'):
        print(f"Skipped: {image_file}")
        continue
        
    elif key in classes:
        target_class = classes[key]
        target_folder = os.path.join(output_base, target_class)
        dst_path = os.path.join(target_folder, image_file)
        
        # Move the file
        try:
            shutil.move(src_path, dst_path)
            count += 1
            print(f"[{count}] Moved to -> {target_class}")
        except Exception as e:
            print(f"Error moving file: {e}")

cv2.destroyAllWindows()
print(f"\nDone! You sorted {count} images.")
