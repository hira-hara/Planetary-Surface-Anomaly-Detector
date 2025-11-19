import splitfolders
import os

# 1. Setup paths
input_folder = "dataset"  # The folder where you put your 6 chosen classes
output_folder = "split_dataset"  # This will be created automatically

# Check if input exists
if not os.path.exists(input_folder):
    print(f"Error: Create a folder named '{input_folder}' and put your class subfolders inside it first!")
else:
    # 2. Split the data
    # Ratio: 70% Train, 15% Val, 15% Test
    splitfolders.ratio(input_folder, output=output_folder,
                       seed=42, ratio=(.7, .15, .15), group_prefix=None)

    print("Success! Data split into 'dataset/train', 'dataset/val', and 'dataset/test'")
    
    # 3. Verify counts (Artifact for Deliverable 2)
    for split in ['train', 'val', 'test']:
        print(f"\n--- {split.upper()} SET ---")
        for label in os.listdir(os.path.join(output_folder, split)):
            path = os.path.join(output_folder, split, label)
            if os.path.isdir(path):
                print(f"{label}: {len(os.listdir(path))} images")
