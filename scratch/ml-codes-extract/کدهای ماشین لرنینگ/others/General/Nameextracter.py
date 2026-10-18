import os
import csv

# Set the path to your folder containing the CSVs
folder_path = r"C:\Users\lenovo\Desktop\New folder"

# Get list of all .csv files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

# Remove the ".csv" extension from each filename
file_names = [os.path.splitext(f)[0] for f in csv_files]

# Optional: Sort alphabetically
file_names.sort()

# Define output file path
output_file = os.path.join(folder_path, 'csv_file_names.csv')

# Write to CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['name'])  # header
    for name in file_names:
        writer.writerow([name])

print(f"✅ Saved list of {len(file_names)} file names to {output_file}")
