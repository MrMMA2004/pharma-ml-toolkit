import os
import shutil

def main():
    # Get the directory where the script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the output folder inside the same directory
    output_folder = os.path.join(base_dir, 'extracted_dialogs')
    os.makedirs(output_folder, exist_ok=True)

    # Loop through subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'r.dlg':
                folder_name = os.path.basename(os.path.dirname(os.path.join(root, file)))
                new_filename = f"{folder_name}.dlg"
                source_path = os.path.join(root, file)
                destination_path = os.path.join(output_folder, new_filename)

                # Handle duplicate names (optional)
                counter = 1
                while os.path.exists(destination_path):
                    new_filename = f"{folder_name}_{counter}.dlg"
                    destination_path = os.path.join(output_folder, new_filename)
                    counter += 1

                shutil.copy(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")

if __name__ == '__main__':
    main()
