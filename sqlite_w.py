import numpy as np
import sqlite3
from tqdm import tqdm
import zarr

# Create a large NumPy array
weights_list = zarr.load('D:/Python/ADWA_RL/500000/weights_list.zarr')
# large_array = np.random.rand(10000, 10000)

# Save the NumPy array in an SQLite database
conn = sqlite3.connect("weights_list_500000.db")
c = conn.cursor()

# Drop the old table if it exists
c.execute("DROP TABLE IF EXISTS weights_list")

# Create the new table
c.execute("CREATE TABLE weights_list (h REAL, o REAL, v REAL)")

# Define the chunk size
chunk_size = 100

# Insert data in chunks using executemany
rows, cols = weights_list.shape
for i in tqdm(range(0, rows, chunk_size)):
    chunk = [
    (weights_list[x, 0], weights_list[x, 1], weights_list[x, 2])
    for x in range(i, min(i + chunk_size, rows))
    ]
    c.executemany("INSERT INTO weights_list VALUES (?, ?, ?)", chunk)

conn.commit()
conn.close()