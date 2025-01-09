import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import silhouette_score
from collections import Counter
import random
from collections import deque


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """Train the decision tree."""
        features = list(X.columns)  # Ensure features is a list
        self.tree = self._build_tree(X, y, features, 0)  # Pass depth as a positional argument

    def predict(self, X):
        """Predict the labels for a given dataset."""
        return [self._predict_single(row, self.tree) for _, row in X.iterrows()]

    def _mean_squared_error(self, y):
        """Calculate the mean squared error of the labels.""" 
        return np.mean((y - np.mean(y)) ** 2)

    def _variance_reduction(self, y, y_left, y_right):
        """Calculate the variance reduction from a split."""
        total_variance = self._mean_squared_error(y)
        left_variance = self._mean_squared_error(y_left)
        right_variance = self._mean_squared_error(y_right)
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        return total_variance - (weight_left * left_variance + weight_right * right_variance)

    def _best_split(self, X, y, features):
        """Find the best split for the dataset."""
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature in features:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                left_indices = X[feature] <= threshold
                right_indices = X[feature] > threshold

                y_left = y[left_indices]
                y_right = y[right_indices]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain = self._variance_reduction(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, features, depth):
        """Recursively build the decision tree."""
        if len(set(y)) == 1:
            return np.mean(y)  # Return the mean of y if it's all the same

        if self.max_depth is not None and depth >= self.max_depth:
            return np.mean(y)  # Return the mean if max depth is reached

        best_feature, best_threshold = self._best_split(X, y, features)

        if best_feature is None:
            return np.mean(y)  # If no good split, return the mean

        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold
        new_features = [feature for feature in features if feature != best_feature]

        left_subtree = self._build_tree(X[left_indices], y[left_indices], new_features, depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], new_features, depth + 1)

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def _predict_single(self, row, tree):
        """Predict a single sample."""
        if not isinstance(tree, dict):
            return tree  # Return the predicted value (leaf node)

        if row[tree["feature"]] <= tree["threshold"]:
            return self._predict_single(row, tree["left"])
        else:
            return self._predict_single(row, tree["right"])

    def print_tree(self, tree=None, depth=0):
        """Print the decision tree."""
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            print("  " * depth + f"Leaf: {tree}")
            return

        print("  " * depth + f"[Feature: {tree['feature']}, Threshold: {tree['threshold']}]")
        self.print_tree(tree["left"], depth + 1)
        self.print_tree(tree["right"], depth + 1)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """Generate a bootstrap sample of the dataset."""
        n_samples = len(X)
        indices = np.random.choice(n_samples, self.sample_size or n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        """Train the random forest."""
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """Predict the labels for a given dataset."""
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)

def evaluate_regression(y_true, y_pred):
    metrics = {
        'Mean Absolute Error (MAE)': mean_absolute_error(y_true, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_true, y_pred),
        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R^2 Score': r2_score(y_true, y_pred),
    }
    return metrics


def calculate_total_cost(data, medoids):
    total_cost = 0
    for _, row in data.iterrows():
        min_distance = min(np.linalg.norm(row.values - medoid) for medoid in medoids)
        total_cost += min_distance
    return total_cost

# Function to assign clusters based on medoids
def assign_clusters(data, medoids):
    clusters = {i: [] for i in range(len(medoids))}
    for idx, row in data.iterrows():
        distances = [np.linalg.norm(row.values - medoid) for medoid in medoids]
        nearest_medoid = np.argmin(distances)
        clusters[nearest_medoid].append(idx)
    return clusters

# CLARANS implementation
def clarans(data, k, numlocal, maxneighbor):
    best_medoids = None
    best_cost = float('inf')

    for _ in range(numlocal):
        # Initialize random medoids
        medoids_indices = random.sample(range(len(data)), k)
        medoids = data.iloc[medoids_indices].values
        current_cost = calculate_total_cost(data, medoids)

        for _ in range(maxneighbor):
            # Randomly select a medoid to swap
            medoid_idx = random.choice(range(k))
            non_medoid_indices = [i for i in range(len(data)) if i not in medoids_indices]
            new_medoid_index = random.choice(non_medoid_indices)

            # Swap medoid
            new_medoids = medoids.copy()
            new_medoids[medoid_idx] = data.iloc[new_medoid_index].values

            # Calculate the new cost
            new_cost = calculate_total_cost(data, new_medoids)

            # If new configuration is better, update medoids and cost
            if new_cost < current_cost:
                medoids = new_medoids
                medoids_indices[medoid_idx] = new_medoid_index
                current_cost = new_cost

        # Keep track of the best configuration
        if current_cost < best_cost:
            best_medoids = medoids
            best_cost = current_cost

    # Assign data points to the nearest medoid
    clusters = assign_clusters(data, best_medoids)

    return best_medoids, clusters

def dbscan(data, eps, min_samples):
  
    n = len(data)
    labels = [-1] * n  # Initialize all points as noise (-1)
    cluster_id = 0

    def region_query(point_idx):
        """Find all points within `eps` distance of `point_idx`."""
        neighbors = []
        for idx, row in data.iterrows():
            if np.linalg.norm(data.iloc[point_idx].values - row.values) <= eps:
                neighbors.append(idx)
        return neighbors

    def expand_cluster(point_idx, neighbors):
        """Expand the cluster starting from `point_idx`."""
        nonlocal cluster_id
        labels[point_idx] = cluster_id
        queue = deque(neighbors)
        while queue:
            current_idx = queue.popleft()
            if labels[current_idx] == -1:  # Previously marked as noise
                labels[current_idx] = cluster_id
            if labels[current_idx] != -1:  # Already processed
                continue
            labels[current_idx] = cluster_id
            current_neighbors = region_query(current_idx)
            if len(current_neighbors) >= min_samples:
                queue.extend(current_neighbors)

    for point_idx in range(n):
        if labels[point_idx] != -1:  # Already processed
            continue

        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(point_idx, neighbors)

    return labels


chioces = ['Summer','Spring' , 'Autumn' ,'Winter']

target = 'Winter'


df = pd.read_csv(f'{target}.csv')



df_target= pd.read_csv('target.csv')



def tree_regression(df , df_target , max_depth , target ) :
    
    X = df
    y = df_target[f"Qair_{target}"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  )

    mine = DecisionTree(max_depth = max_depth )   
    mine.fit(X_train, y_train)
    predictions = mine.predict(X_test)

    sickit = DecisionTreeRegressor(max_depth= max_depth)
    sickit.fit(X_train, y_train) 
    predictions_sickit = sickit.predict(X_test)

    my_results = evaluate_regression(y_test , predictions)
    sickit_results = evaluate_regression(y_test , predictions_sickit)

    return my_results , sickit_results , mine , sickit


def forest_regression(df , df_target , max_depth , target ,number_trees) :
    
    X = df
    y = df_target[f"Qair_{target}"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  )

    mine = RandomForest(max_depth = max_depth , n_trees=number_trees)   
    mine.fit(X_train, y_train)
    predictions = mine.predict(X_test)

    sickit = RandomForestRegressor(max_depth= max_depth, n_estimators = number_trees)
    sickit.fit(X_train, y_train) 
    predictions_sickit = sickit.predict(X_test)

    my_results = evaluate_regression(y_test , predictions)
    sickit_results = evaluate_regression(y_test , predictions_sickit)

    return my_results , sickit_results , mine ,sickit


def data_in_2d(df):

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(df)

    plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', alpha=0.5)
    plt.title('PCA - 2D Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.show()


def perform_calarnas(df , k , numlocal , maxneighbor):

    medoids, clusters = clarans(df, k, numlocal, maxneighbor)

    cluster_colors = sns.color_palette("Set2", k)  

    plt.figure(figsize=(8, 6))
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(df)

    for i, (cluster_name, indices) in enumerate(clusters.items()):
        cluster_points = data_2d[indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_name}",
                    color=cluster_colors[i], alpha=0.7, edgecolor='k')

    plt.title('PCA Projection with Clusters', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    def clusters_to_labels(clusters, n_points):
        labels = np.full(n_points, -1)  # Initialize with -1 (unassigned)
        for cluster_id, indices in clusters.items():
            labels[indices] = cluster_id  # Assign cluster IDs to the corresponding indices
        return labels

    n_points = len(df)  # Total number of data points
    labels = clusters_to_labels(clusters, n_points)

    if len(set(labels)) > 1:
        sil_score = silhouette_score(df, labels)
        return sil_score, plt
    
def perform_dbscan ( df ,eps ,min_samples):
    labels = dbscan(df, eps, min_samples)

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(df)
    # Combine PCA-transformed data with clusters
    pca_data_with_clusters = pd.DataFrame(data_2d, columns=["PCA1", "PCA2"])
    pca_data_with_clusters["Cluster"] = labels

    # Plot clusters
    plt.figure(figsize=(8, 6))
    unique_clusters = pca_data_with_clusters["Cluster"].unique()

    num_clusters = len(unique_clusters)

    # Assign distinct colors to clusters
    colors = [mcolors.hsv_to_rgb((i / num_clusters, 0.8, 0.9)) for i in range(num_clusters)]

    # Loop through unique clusters and plot
    for i, cluster in enumerate(unique_clusters):
        cluster_points = pca_data_with_clusters[pca_data_with_clusters["Cluster"] == cluster]
        plt.scatter(cluster_points["PCA1"], cluster_points["PCA2"],
                    label=f"Cluster {cluster}",
                    color=colors[i],
                    alpha=0.7,
                    edgecolor='k')
    plt.title('PCA Projection with Clusters', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title="Clusters")
    plt.grid(alpha=0.3)
    plt.show()

    if len(set(labels)) > 1:
        sil_score = silhouette_score(df, labels)
        return sil_score, plt

def new_instance(PSurf_Autumn, Rainf_Autumn, Snowf_Autumn, Tair_Autumn,
       Wind_Autumn, sand_topsoil, silt_topsoil, clay_topsoil,
       pH_water_topsoil, OC_topsoil, OC_subsoil, N_topsoil,
       N_subsoil, CEC_topsoil, CaCO3_topsoil, CN_topsoil ,trained_model , season):
    
    columns = [f'PSurf_{season}', f'Rainf_{season}', f'Snowf_{season}', f'Tair_{season}',
       f'Wind_{season}', 'sand % topsoil', 'silt % topsoil', 'clay % topsoil',
       'pH water topsoil', 'OC % topsoil', 'OC % subsoil', 'N % topsoil',
       'N % subsoil', 'CEC topsoil', 'CaCO3 % topsoil', 'C/N topsoil']
    
    array = [PSurf_Autumn, Rainf_Autumn, Snowf_Autumn, Tair_Autumn,
       Wind_Autumn, sand_topsoil, silt_topsoil, clay_topsoil,
       pH_water_topsoil, OC_topsoil, OC_subsoil,N_topsoil,
       N_subsoil,CEC_topsoil, CaCO3_topsoil, CN_topsoil]
    

    df = pd.DataFrame([array], columns=columns)

    predictions = trained_model.predict(df)
    return predictions
 



tree = tree_regression(df , df_target , 1 ,'Winter')
forest = forest_regression(df , df_target , 1 ,'Winter', 1)
prediction =  new_instance (5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,tree[3] ,'Winter')
print(prediction)

