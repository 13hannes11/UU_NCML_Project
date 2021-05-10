import pandas as pd
import os

base_dir = "../de/"
out_dir = "csv/"
in_dir = "input/"
title_file = "filename_to_titles.csv"


if not os.path.exists(os.path.join(base_dir, out_dir)):
    os.makedirs(os.path.join(base_dir, out_dir))

# Copy titles file and replace file endings
print(f'Copying Title File')
with open(os.path.join(base_dir, in_dir, title_file), 'r') as file:
    file_content = file.read()
    file_content = file_content.replace('.xlsx', '.csv')
with open(os.path.join(base_dir, out_dir, title_file), 'w') as file:
    file.write(file_content)

# Convert xlsx files to csv 
for dirname, _, filenames in os.walk(os.path.join(base_dir, in_dir)):
    for filename in filenames:
        if filename != title_file:
            print(f'Reading {filename}')
            read_file = pd.read_excel (os.path.join(dirname, filename))
            print(f'Saving {filename}')
            read_file.to_csv (os.path.join(base_dir, out_dir, filename.split(".", 1)[0] + ".csv"), index = None, header=True)

