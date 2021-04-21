import pandas as pd
import os

# Convert data to csv
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        read_file = pd.read_excel (os.path.join(dirname, filename))
        read_file.to_csv (os.path.join('csv', filename.split(".", 1)[0] + ".csv"), index = None, header=True)

