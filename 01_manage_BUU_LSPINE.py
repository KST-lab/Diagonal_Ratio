import zipfile
import shutil
import os

# Path to the ZIP file
zip_file_path = 'BUU-LSPINE_V2.zip'

# Directory to extract to
extract_to_dir = '.'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_to_dir, exist_ok=True)

# Open the ZIP file
print('Extracting BUU-LSPINE ...')
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the specified directory
    zip_ref.extractall(extract_to_dir)

print(f'Files extracted to {extract_to_dir}')

# Source folder path
source_folder = os.path.join('BUU_LSPINE_V2', 'csv_info')

# Destination folder path
destination_folder = os.path.join('.')
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

# Move the folder
if os.path.exists(source_folder):
    shutil.move(source_folder, destination_folder)
    os.rename('csv_info', 'csv')


print(f'Folder moved from {source_folder} to {destination_folder}')


# Source folder path
source_folder = os.path.join('BUU_LSPINE_V2', 'data')

# Destination folder path
destination_folder = os.path.join('dataset')
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

# Move the folder
if os.path.exists(source_folder):
    shutil.move(source_folder, destination_folder)
    os.rename('dataset/data', 'dataset/BUU_LSPINE_V2')


print(f'Folder moved from {source_folder} to {destination_folder}')