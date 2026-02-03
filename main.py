import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import matplotlib.pyplot as plt

# STEP 1: Load Dataset
file_path = "emails.csv"
email_df = pd.read_csv(file_path)

# Auto-detect correct text column
possible_cols = ["BodyTokens", "Body", "body", "Message", "message"]
text_col = None

for col in possible_cols:
    if col in email_df.columns:
        text_col = col
        break

if text_col is None:
    raise ValueError("❌ No valid text column found in CSV file!")

print(f"📌 Using text column: {text_col}")

email_texts = email_df[text_col].astype(str)
email_texts = email_texts[email_texts.str.strip() != ""]

print(f"✅ Loaded {len(email_texts)} email records for clustering")

# STEP 2: 80th Percentile Cosine Similarity Threshold
sample_size = min(8000, email_texts.shape[0])
sample_idx = np.random.choice(email_texts.index, sample_size, replace=False)
sample_texts = email_texts.loc[sample_idx]

vectorizer_sample = TfidfVectorizer(stop_words='english', max_features=5000)
X_sample = vectorizer_sample.fit_transform(sample_texts)

subset_size = min(1000, X_sample.shape[0])
subset_indices = random.sample(range(X_sample.shape[0]), subset_size)
X_subset = X_sample[subset_indices]

cos_sim = cosine_similarity(X_subset)
sim_values = cos_sim[np.triu_indices_from(cos_sim, k=1)]

threshold_80 = np.percentile(sim_values, 80)
print(f"📈 80th percentile cosine similarity = {threshold_80:.4f}")

# STEP 3: TF-IDF on Full Dataset
vectorizer_full = TfidfVectorizer(stop_words='english', max_features=5000)
X_full = vectorizer_full.fit_transform(email_texts)


# STEP 4: Determine Clusters Automatically
num_clusters = int(len(email_texts) * (1 - threshold_80) / 9000)
num_clusters = max(5, min(num_clusters, 50))

print(f"🧩 Automatically determined number of clusters: {num_clusters}")

# STEP 5: MiniBatch KMeans Clustering
kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=10000)
email_df['Cluster'] = kmeans.fit_predict(X_full)

# STEP 6: Generate Topic Labels for Clusters
def get_top_keywords(tfidf_matrix, vectorizer, kmeans_model, top_n=5):
    cluster_keywords = {}
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for cluster_id in range(kmeans_model.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[cluster_id, :top_n]]
        cluster_keywords[cluster_id] = " ".join(top_terms).title()

    return cluster_keywords

cluster_labels = get_top_keywords(X_full, vectorizer_full, kmeans, top_n=4)
email_df["ClusterLabel"] = email_df["Cluster"].map(cluster_labels)

print("\n🆕 TOPIC LABELS FOR CLUSTERS:")
for cid, label in cluster_labels.items():
    print(f"Cluster {cid}: {label}")

# STEP 7: Save Clustered CSV
output_file = "clustered_emails_fixed.csv"
email_df.to_csv(output_file, index=False)
print(f"\n✅ Clustered emails saved to: {output_file}")

# STEP 8: Cluster Summary
cluster_counts = email_df['Cluster'].value_counts().sort_index()

print("\n📊 CLUSTER DISTRIBUTION:")
for cid, count in cluster_counts.items():
    print(f"Cluster {cid}: {count} emails")

# Short labels for plotting
def clean_label(label):
    parts = label.split()
    if len(parts) > 2:
        return " ".join(parts[:2])
    return label

email_df["ShortLabel"] = email_df["ClusterLabel"].apply(clean_label)

# STEP 9: User-selected Cluster Justification

#print("\n📌 AVAILABLE CLUSTERS:")
#for cid, label in cluster_labels.items():
 #   print(f"Cluster {cid}: {label}")

# Ask user to choose cluster
try:
    selected_cluster = int(input("\n👉 Enter the Cluster ID you want to view: "))
except ValueError:
    print("❌ Invalid input. Please enter a number.")
    exit()

# Check if cluster exists
if selected_cluster not in cluster_labels:
    print("❌ Cluster ID not found.")
    exit()

# Filter emails from selected cluster
cluster_emails = email_df[email_df["Cluster"] == selected_cluster]

# Select up to 10 emails
sample_rows = cluster_emails.sample(
    n=min(10, len(cluster_emails)), random_state=42
)

cluster_label = cluster_labels[selected_cluster]
top_keywords = cluster_label.split()

print(f"📌 SHOWING 10 EMAILS FROM CLUSTER {selected_cluster}")

for idx, row in sample_rows.iterrows():
    print(f"""
----- EMAIL SAMPLE -----

Email ID      : {idx}
Cluster       : {selected_cluster}
Cluster Label : {cluster_label}

FROM          : {row.get('From', 'N/A')}
TO            : {row.get('To', 'N/A')}
DATE          : {row.get('Date', 'N/A')}
SUBJECT       : {row.get('Subject', 'N/A')}

BODY:
{row[text_col]}

≈ WHY THIS EMAIL BELONGS TO THIS CLUSTER ≈
This email belongs to cluster {selected_cluster} because
its content strongly matches the cluster keywords:
{top_keywords}

----------------------------------------------------
""")

# STEP 10: Visualization With Topic Names (LAST)
cluster_counts = email_df["ShortLabel"].value_counts()

plt.figure(figsize=(14, 8))

ax = cluster_counts.plot(
    kind='bar',
    edgecolor='black'
)

plt.title('Number of Emails per Topic Cluster', fontsize=18, fontweight='bold')
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Number of Emails', fontsize=12)

# Rotate x-labels more to avoid overlap
plt.xticks(rotation=65, ha='right', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.6)

# 🔹 Add counts vertically above bars
max_height = cluster_counts.max()

for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + max_height * 0.01,   # spacing
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=9,
        rotation=90,                 # vertical numbers
        fontweight='bold'
    )

plt.tight_layout()
plt.show()
