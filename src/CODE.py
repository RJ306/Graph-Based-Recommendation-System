import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read Friendship Data from CSV
def load_friendship_data(csv_file):
    df = pd.read_csv(csv_file)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['user1'], row['user2'])
    return G

# Load the graph data from a CSV file
csv_file = 'C:/Users/LENOVO/Desktop/proj/pakistani_friendships.csv'  # Replace with your CSV path
G = load_friendship_data(csv_file)

# Step 2: Prepare Training Data
def prepare_training_data(G):
    positive_pairs = []
    negative_pairs = []
    for user1 in G.nodes:
        for user2 in G.neighbors(user1):
            if user1 < user2:
                positive_pairs.append((user1, user2, 1))
    all_pairs = [(user1, user2) for user1 in G.nodes for user2 in G.nodes if user1 < user2]
    for user1, user2 in all_pairs:
        if user2 not in G.neighbors(user1):
            negative_pairs.append((user1, user2, 0))
    training_data = positive_pairs + negative_pairs
    return pd.DataFrame(training_data, columns=['user1', 'user2', 'label'])

training_data = prepare_training_data(G)

# Step 3: Feature Engineering
def compute_features(df, G):
    features = []
    for _, row in df.iterrows():
        user1, user2 = row['user1'], row['user2']
        common_neighbors_count = len(list(nx.common_neighbors(G, user1, user2)))
        neighbors_user1 = set(G.neighbors(user1))
        neighbors_user2 = set(G.neighbors(user2))
        intersection_size = len(neighbors_user1.intersection(neighbors_user2))
        union_size = len(neighbors_user1.union(neighbors_user2))
        jaccard_similarity = intersection_size / union_size if union_size > 0 else 0
        adamic_adar_index = sum(
            1 / np.log(len(list(G.neighbors(neighbor)))) for neighbor in nx.common_neighbors(G, user1, user2)
        )
        features.append([common_neighbors_count, jaccard_similarity, adamic_adar_index])
    return np.array(features)

X = compute_features(training_data, G)
y = training_data['label'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression(class_weight='balanced', C=0.1)
model.fit(X_train, y_train)

# Step 4: Recommendation Function
def recommend_friends_all_shared_nodes(G, user, model):
    recommendations = []
    for potential_friend in G.nodes:
        if potential_friend != user and potential_friend not in G.neighbors(user):
            common_neighbors_count = len(list(nx.common_neighbors(G, user, potential_friend)))
            if common_neighbors_count > 0:
                neighbors_user1 = set(G.neighbors(user))
                neighbors_user2 = set(G.neighbors(potential_friend))
                intersection_size = len(neighbors_user1.intersection(neighbors_user2))
                union_size = len(neighbors_user1.union(neighbors_user2))
                jaccard_similarity = intersection_size / union_size if union_size > 0 else 0
                adamic_adar_index = sum(
                    1 / np.log(len(list(G.neighbors(neighbor)))) for neighbor in nx.common_neighbors(G, user, potential_friend)
                )
                features = np.array([[common_neighbors_count, jaccard_similarity, adamic_adar_index]])
                prediction_prob = model.predict_proba(features)[0][1]
                recommendations.append((potential_friend, prediction_prob))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

# Visualization Function
def visualize_graph(G, recommendations, user):
    pos = nx.spring_layout(G, k=1.0, iterations=200)
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.2, alpha=0.7, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=10)
    recommended_nodes = [rec[0] for rec in recommendations]
    nx.draw_networkx_nodes(G, pos, nodelist=recommended_nodes, node_size=800, node_color="lightgreen", alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=[user], node_size=900, node_color="orange", alpha=1.0)
    plt.title(f"Friend Recommendations for {user}", fontsize=14)
    plt.axis("off")
    plt.show()

# Sequential Recommendations for All Users
def recommend_and_visualize_for_all_nodes(G, model):
    for user in G.nodes:
        print(f"Generating recommendations for user '{user}':")
        recommendations = recommend_friends_all_shared_nodes(G, user, model)
        print(f"Recommendations for '{user}': {recommendations}")
        visualize_graph(G, recommendations, user)

# Step 5: Evaluation Metrics
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Friends", "Friends"], yticklabels=["Not Friends", "Friends"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="purple", lw=2, label=f"Precision-Recall Curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.show()

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot Evaluation Metrics
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred_proba)
plot_precision_recall_curve(y_test, y_pred_proba)

# Generate and Visualize Recommendations for All Users
recommend_and_visualize_for_all_nodes(G, model)