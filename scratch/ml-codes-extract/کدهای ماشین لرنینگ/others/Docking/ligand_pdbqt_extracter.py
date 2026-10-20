import os
import shutil
import re

def get_unique_filename(directory, filename):
    """Generate a unique filename if one already exists."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

# Define the main folder (current directory)
main_folder = os.getcwd()

# Optional: create an output folder to collect the pdbqt files
output_folder = os.path.join(main_folder, "collected_pdbqt")
os.makedirs(output_folder, exist_ok=True)

# Loop through all subdirectories
for root, dirs, files in os.walk(main_folder):
    for file in files:
        # Match files like 1.pdbqt, 2.pdbqt, etc.
        if re.match(r'^\d+\.pdbqt$', file):
            full_path = os.path.join(root, file)
            # Skip if source and target are the same
            if os.path.abspath(root) == os.path.abspath(output_folder):
                continue
            # Get a unique name to avoid overwrite
            new_filename = get_unique_filename(output_folder, file)
            dest_path = os.path.join(output_folder, new_filename)
            shutil.copy(full_path, dest_path)
            print(f"Copied: {file} → {new_filename}")

print("✅ Extraction complete.")
