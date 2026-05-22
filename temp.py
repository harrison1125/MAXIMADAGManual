import os
import shutil
from pathlib import Path

# 1. Define your source directory and your destination summary directory
source_dir = Path('/Users/hpark108/Downloads/Resources 2')
dest_dir = Path("/Users/hpark108/Desktop/all_titanium_plots")

# Create the summary folder if it doesn't exist yet
dest_dir.mkdir(parents=True, exist_ok=True)

print(f"Scanning for plots in: {source_dir}")
print(f"Copying results to: {dest_dir}\n" + "-"*50)

# 2. Walk through all folders and find the target images
count = 0
for filepath in source_dir.rglob("TiAtomicPercent.png"):
    # Extract the unique folder name (e.g., JHAMAC00003-S8R4C3_...) to keep the files distinct
    unique_id = filepath.parent.parent.name
    
    # Create a descriptive new filename: "JHAMAC00003-S8R4C3_..._TiAtomicPercent.png"
    new_filename = f"{unique_id}_TiAtomicPercent.png"
    target_path = dest_dir / new_filename
    
    # Copy the file over
    shutil.copy2(filepath, target_path)
    print(f"Copied: {unique_id} -> {new_filename}")
    count += 1

print("-"*50)
print(f"Done! Successfully collected {count} plots in '{dest_dir}'.")
