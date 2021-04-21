import pandas as pd
import os

# Convert data to csv
base_dir = "./de/"

out_dir = "csv"
if not os.path.exists(base_dir + out_dir):
    os.makedirs(base_dir + out_dir)

for dirname, _, filenames in os.walk(base_dir + 'input'):
    for filename in filenames:
        read_file = pd.read_excel (os.path.join(dirname, filename))
        read_file.to_csv (os.path.join(base_dir + out_dir, filename.split(".", 1)[0] + ".csv"), index = None, header=True)

