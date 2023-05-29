import numpy as np
import sqlite3
from tqdm import tqdm
import zarr

# Create a large NumPy array
initial_states_list = zarr.load('D:/Python/ADWA_RL/new_data/500/initial_states_list.zarr')
# large_array = np.random.rand(10000, 10000)

# Save the NumPy array in an SQLite database
conn = sqlite3.connect("initial_states_list_500.db")
c = conn.cursor()

# Drop the old table if it exists
c.execute("DROP TABLE IF EXISTS initial_states_list")

# Create the new table
c.execute("CREATE TABLE initial_states_list (ix REAL, iy REAL, ith REAL, gx REAL, gy REAL, iv REAL, iw REAL)")

# Define the chunk size
chunk_size = 100

# Insert data in chunks using executemany
rows, cols = initial_states_list.shape
for i in tqdm(range(0, rows, chunk_size)):
    chunk = [
        (initial_states_list[x, 0], initial_states_list[x, 1], initial_states_list[x, 2], initial_states_list[x, 3],
        initial_states_list[x, 4], initial_states_list[x, 5], initial_states_list[x, 6])
        for x in range(i, min(i + chunk_size, rows))
    ]
    c.executemany("INSERT INTO initial_states_list VALUES (?, ?, ?, ?, ?, ?, ?)", chunk)

conn.commit()
conn.close()