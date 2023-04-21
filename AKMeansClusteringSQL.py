import numpy as np
import sqlite3
from sklearn.cluster import KMeans

def create_customers_table(cursor):
    cursor.execute('CREATE TABLE customers (id INTEGER PRIMARY KEY, vector BLOB, cluster INTEGER)')

def insert_customer_data(cursor, purchases, customer_clusters):
    for i, purchase in enumerate(purchases):
        vector_blob = purchase.tobytes()
        cursor.execute('INSERT INTO customers VALUES (?, ?, ?)', (i, sqlite3.Binary(vector_blob), customer_clusters[i]))

def retrieve_customers_by_cluster(cursor, cluster_id):
    cursor.execute('SELECT * FROM customers WHERE cluster = ?', (cluster_id,))
    return cursor.fetchall()

def display_customers(cluster_id, customers):
    print(f'Cluster {cluster_id}:')
    for row in customers:
        print(row)

def find_closest_customers_to_centroid(cursor, model):
    closest_customers = []
    centroids = model.cluster_centers_
    
    for cluster_id, centroid in enumerate(centroids):
        cursor.execute('SELECT id, vector FROM customers WHERE cluster = ?', (cluster_id,))
        customers = [(row[0], np.frombuffer(row[1], dtype=np.float64)) for row in cursor.fetchall()]
        closest_customer = min(customers, key=lambda x: np.linalg.norm(x[1] - centroid))
        closest_customers.append(closest_customer)
    
    return closest_customers

def main():
    purchases = np.load('purchases.npy')
    model = KMeans(n_clusters=5)
    customer_clusters = model.fit_predict(purchases)

    with sqlite3.connect('customers.db') as conn:
        cursor = conn.cursor()
        create_customers_table(cursor)
        insert_customer_data(cursor, purchases, customer_clusters)

        for cluster_id in range(model.n_clusters):
            customers = retrieve_customers_by_cluster(cursor, cluster_id)
            display_customers(cluster_id, customers)

        closest_customers = find_closest_customers_to_centroid(cursor, model)
        print("\nClosest customers to centroids:")
        for i, customer in enumerate(closest_customers):
            print(f"Cluster {i}: Customer ID {customer[0]}")

if __name__ == "__main__":
    main()
