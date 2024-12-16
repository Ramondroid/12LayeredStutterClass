import os
import shutil
import pandas as pd

csv_path = 'StutterFiles.csv'

root_dir = 'full-dataset/clips/stuttering-clips/clips'

destination_dir = 'StutClass/12K-Sorted/NoStutter'

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

df = pd.read_csv(csv_path)

for file_name in df['Audio'].str.strip():
    found = False
    destination_path = os.path.join(destination_dir, file_name)

    if os.path.exists(destination_path):
        print(f"File already exists in the destination: {file_name}")
        continue

    print(f"Searching for: {file_name}")
    
    for subdir, dirs, files in os.walk(root_dir):
        if file_name in files:
            shutil.copy(os.path.join(subdir, file_name), destination_dir)
            print(f"Copied: {file_name}")
            found = True
            break
    
    if not found:
        print(f'File not found: {file_name}')