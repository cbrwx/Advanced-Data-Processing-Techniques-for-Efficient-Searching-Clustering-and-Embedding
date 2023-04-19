import numpy as np
import sqlite3
from sklearn.cluster import KMeans

# Load the customer purchase history dataset and k-means model
purchases = np.load('purchases.npy')
model = KMeans(n_clusters=5)

# Generate a cluster label for each customer using the k-means model
customer_clusters = model.fit_predict(purchases)

# Create a SQLite database and table to store the customer vectors and clusters
conn = sqlite3.connect('customers.db')
cursor = conn.cursor()
cursor.execute('CREATE TABLE customers (id INTEGER PRIMARY KEY, vector TEXT, cluster INTEGER)')

# Insert the customer vectors and clusters into the SQLite database
for i, purchase in enumerate(purchases):
    vector_str = ', '.join(str(x) for x in purchase)
    cursor.execute('INSERT INTO customers VALUES (?, ?, ?)', (i, vector_str, customer_clusters[i]))

# Retrieve the customers in each cluster from the SQLite database
cursor.execute('SELECT * FROM customers WHERE cluster = 0')
cluster0 = cursor.fetchall()
cursor.execute('SELECT * FROM customers WHERE cluster = 1')
cluster1 = cursor.fetchall()
cursor.execute('SELECT * FROM customers WHERE cluster = 2')
cluster2 = cursor.fetchall()
cursor.execute('SELECT * FROM customers WHERE cluster = 3')
cluster3 = cursor.fetchall()
cursor.execute('SELECT * FROM customers WHERE cluster = 4')
cluster4 = cursor.fetchall()

# Display the customers in each cluster
print('Cluster 0:')
for row in cluster0:
    print(row)
print('Cluster 1:')
for row in cluster1:
    print(row)
print('Cluster 2:')
for row in cluster2:
    print(row)
print('Cluster 3:')
for row in cluster3:
    print(row)
print('Cluster 4:')
for row in cluster4:
    print(row)
