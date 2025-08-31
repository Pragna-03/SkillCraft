# === Import Required Libraries ===
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === Load Dataset ===
# Replace this path with the correct one where your dataset is located
data_path = "C:/Users/prana/OneDrive/Desktop/SKILLCRAFT/Mall_Customers.csv"
df = pd.read_csv(data_path)

# === Inspect Data ===
print("Sample data:")
print(df.head())

# === Select Features for Clustering ===
# We'll use 'Annual Income' and 'Spending Score' as clustering features
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# === Normalize the Features ===
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# === Determine Optimal Number of Clusters (Elbow Method) ===
distortions = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=1)
    model.fit(features_scaled)
    distortions.append(model.inertia_)

# === Plot Elbow Curve ===
plt.figure(figsize=(7, 5))
plt.plot(k_values, distortions, marker='o', linestyle='--')
plt.title("Elbow Method - Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Distortion)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Apply KMeans with Optimal k ===
optimal_k = 5  # You can adjust this based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=1)
cluster_labels = kmeans.fit_predict(features_scaled)

# === Add Cluster Labels to Original Data ===
df['Cluster'] = cluster_labels

# === Visualize the Clusters ===
plt.figure(figsize=(8, 6))

for cluster in range(optimal_k):
    cluster_points = features_scaled[cluster_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Centroids')

plt.title("Customer Segments Based on Income and Spending")
plt.xlabel("Annual Income (normalized)")
plt.ylabel("Spending Score (normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
