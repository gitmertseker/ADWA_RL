import sqlite3

# Replace 'your_database_file.db' with the path to your SQLite database file
db_file = 'reward_list_100.db'

# Connect to the SQLite database
conn = sqlite3.connect(db_file)

# Create a cursor object
cursor = conn.cursor()

# Execute an SQL query to fetch data from the database
# Replace 'your_table_name' with the name of the table you want to fetch data from
cursor.execute('SELECT * FROM reward_list')

# Fetch all the results and store them in a list
# data  = cursor.fetchall()

chunk_size = 1000  # Adjust this value based on your memory constraints and performance requirements
result = []

while True:
    data = cursor.fetchmany(chunk_size)
    if not data:
        break

    result.extend(x[0] for x in data)

# Close the cursor and the connection to the database
cursor.close()
conn.close()

# Print the results
print(result)
