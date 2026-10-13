import os
import pandas as pd

# Set the path to your main folder
main_folder = os.getcwd()  # or replace with a specific path

# Initialize empty lists to collect DataFrames
csv_dfs = []
excel_dfs = []

# List all files in the folder
for file in os.listdir(main_folder):
    file_path = os.path.join(main_folder, file)
    
    if os.path.isfile(file_path):
        # Merge CSV files
        if file.lower().endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                csv_dfs.append(df)
            except Exception as e:
                print(f"Error reading CSV file {file}: {e}")
        
        # Merge Excel files
        elif file.lower().endswith('.xlsx') and not file.startswith('~$'):  # avoid temp Excel files
            try:
                df = pd.read_excel(file_path)
                excel_dfs.append(df)
            except Exception as e:
                print(f"Error reading Excel file {file}: {e}")

# Save merged CSV
if csv_dfs:
    merged_csv = pd.concat(csv_dfs, ignore_index=True)
    merged_csv.to_csv(os.path.join(main_folder, 'merged_csv_output.csv'), index=False)
    print("CSV files merged successfully.")

# Save merged Excel
if excel_dfs:
    merged_excel = pd.concat(excel_dfs, ignore_index=True)
    merged_excel.to_excel(os.path.join(main_folder, 'merged_excel_output.xlsx'), index=False)
    print("Excel files merged successfully.")
