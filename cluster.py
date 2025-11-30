import os, sys
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


TEST_CLUSTER_K = 5  #cluster num

amino = list("ACDEFGHIKLMNPQRSTVWY")

def aa_freq_features(seq, prefix):
    seq = str(seq).strip()
    counts = np.zeros(len(amino), dtype=float)
    if len(seq) > 0:
        for ch in seq:
            if ch in amino:
                counts[amino.index(ch)] += 1.0
        counts = counts / len(seq)  # convert to frequencies
    
    return {f"{prefix}_freq_{aa}": val for aa, val in zip(amino, counts)}

#for binding prediction: 1. sequence features only baseline 2. add 3d structure features

def build_seq_features(df):
    rows = []
    for i, row in df.iterrows():
        features = {}

        #strings
        peptide = str(row["Peptide"])
        cdr3a = str(row["CDR3a"])
        cdr3b = str(row["CDR3b"])

        #length-based int
        features["peptide_len"] = len(peptide)
        features["cdr3a_len"] = len(cdr3a)
        features["cdr3b_len"] = len(cdr3b)

        features.update(aa_freq_features(peptide, "pep_"))
        features.update(aa_freq_features(cdr3a, "cdr3a_"))
        features.update(aa_freq_features(cdr3b, "cdr3b_"))

        features["HLA"] = row["HLA"]

        rows.append(features)

    feature_df = pd.DataFrame(rows)

    #one hot encode HLA
    feature_df = pd.get_dummies(feature_df, columns=["HLA"])
    return feature_df

features_baseline = ["Peptide","HLA","Va","Ja","TCRa","CDR1a","CDR2a","CDR3a","CDR3a_extended","Vb","Jb","TCRb","CDR1b","CDR2b","CDR3b","CDR3b_extended","Target"]
join = features_baseline[:-1]  #all but target, to match the test and ground truths


#metric features csv has missing index values (prolly rows that failed processing)
#need to join back to the original vdjdb set to get proper indexing
vdjdb = pd.read_csv("vdjdb_positives.csv")
vdjdb = vdjdb.reset_index()

test = pd.read_csv("test.csv")
test = test.reset_index()

print(vdjdb)
#11312 rows
print(test)
#3484 rows

vdjdb_metrics = pd.read_csv("groundtruth_metrics.csv")
#this is the correct test data with both the original metrics and 3d metrics
train_data = vdjdb.merge(vdjdb_metrics, on="index", how="inner")

test_metrics = pd.read_csv("test_metrics.csv")
test_data = test.merge(test_metrics, on="index", how="inner")

print(train_data)
#6835 rows after merge 

print(test_data)
#2124 rows after merge

#Step 1: build sequence features for train and test

x_train_seq = build_seq_features(train_data[join])
x_test_seq = build_seq_features(test_data[join])
x_test_seq = x_test_seq.reindex(columns=x_train_seq.columns, fill_value=0)

print(x_train_seq)
print(x_train_seq.columns)

print(x_test_seq)
print(x_test_seq.columns)

#cionvert to numpy arrays
x_train_np = x_train_seq.to_numpy()
x_test_np = x_test_seq.to_numpy()

print(x_train_np.shape)
print(x_test_np.shape)

#Step 2: scale the sequence features
#normalize features so the model can train better - mean 0, std 1
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_np)
x_test_scaled = scaler.transform(x_test_np)

print(x_train_scaled)
print(x_test_scaled)

#Step 3: fit kNN on positive vdjdb data
knn = NearestNeighbors(n_neighbors=TEST_CLUSTER_K, metric='euclidean')
knn.fit(x_train_scaled)

#Step 4: compute distances to kNN for test data (binding likeness scores)
test_distances, indices = knn.kneighbors(x_test_scaled)
test_mean_distances = test_distances.mean(axis=1)

raw_scores = -test_mean_distances

#normalization between 0 and 1
min_score = raw_scores.min()
max_score = raw_scores.max()
norm_scores = (raw_scores - min_score) / (max_score - min_score)

#Step 6: attach scores to test data 
test_df = test_data.copy()
test_df["seq_only_score"] = norm_scores

test_df.to_csv("test_with_seq_only_scores.csv", index=False)


''' Now cluster with both sequence and 3d features'''

#above was the seq only baseline, now cluster with both seq and 3d features 
features_3d = ["dG_separated","dSASA_hphobic", "hbonds_int","plddt","pae","cdr3b_plddt"]

#3d feature arrays
x_train_3d = train_data[features_3d].fillna(train_data[features_3d].mean())
x_test_3d = test_data[features_3d].fillna(test_data[features_3d].mean())

x_test_3d = x_test_3d[features_3d].to_numpy()
x_train_3d = x_train_3d[features_3d].to_numpy()

print(x_train_3d)
print(x_test_3d)

x_train_combined = np.hstack([x_train_np, x_train_3d])
x_test_combined = np.hstack([x_test_np, x_test_3d])

print(x_train_combined.shape)
print(x_test_combined.shape)

#scale combined features
scaler_combined = StandardScaler()
x_train_combined_scaled = scaler_combined.fit_transform(x_train_combined)
x_test_combined_scaled = scaler_combined.transform(x_test_combined)

#fit kNN on combined features
knn_combined = NearestNeighbors(n_neighbors=TEST_CLUSTER_K, metric='euclidean')
knn_combined.fit(x_train_combined_scaled)

test_distances_combined, indices_combined = knn_combined.kneighbors(x_test_combined_scaled)
test_mean_distances_combined = test_distances_combined.mean(axis=1) 

raw_scores_combined = -test_mean_distances_combined
min_score_combined = raw_scores_combined.min()
max_score_combined = raw_scores_combined.max()
norm_scores_combined = (raw_scores_combined - min_score_combined) / (max_score_combined - min_score_combined)   

test_df = test_data.copy()
test_df["seq_3d_score"] = norm_scores_combined
test_df.to_csv("test_with_seq_3d_scores.csv", index=False)



output_df = test_data.copy()
output_df["seq_only_score"] = norm_scores            
output_df["seq_3d_score"]   = norm_scores_combined   
output_df.to_csv("both_test_scores.csv", index=False)
print("Wrote both_test_scores.csv")

# #cluster the test data set with 3d features
# X_struct = test_data[features_3d]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_struct)

# kmeans = KMeans(n_clusters=TEST_CLUSTER_K, random_state=0)
# cluster_ids = kmeans.fit_predict(X_scaled)
# test_data["cluster_id"] = cluster_ids
# print(test_data[["index", "cluster_id"] + features_3d])