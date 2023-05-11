import numpy as np
import sqlite3
from tqdm import tqdm
import zarr

# Load a large NumPy array from a Zarr file
costmap_list = zarr.load('D:/Python/ADWA_RL/10000/costmap_list.zarr')

# Save the NumPy array in an SQLite database
conn = sqlite3.connect("costmap_list_10000.db")
c = conn.cursor()

# Drop the old table if it exists
c.execute("DROP TABLE IF EXISTS costmap_list")

# Create the new table with a BLOB column to store the binary image data
c.execute("CREATE TABLE costmap_list (img_data BLOB)")

# Define the chunk size
chunk_size = 100  # You can adjust this based on available memory and performance

# Insert data in chunks using executemany
rows, img_rows, img_cols = costmap_list.shape
for i in tqdm(range(0, rows, chunk_size)):
    chunk = [
        (costmap_list[x].tobytes(),)  # Serialize the 40x40 image as binary data
        for x in range(i, min(i + chunk_size, rows))
    ]
    c.executemany("INSERT INTO costmap_list VALUES (?)", chunk)

conn.commit()
conn.close()
