import os, sys
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

TEST_CLUSTER_K = 3  #cluster num


features_3d = ["index","dG_separated","dSASA_hphobic", "hbonds_int","plddt","pae","cdr3b_plddt"]
features_baseline = ["Peptide","HLA","Va","Ja","TCRa","CDR1a","CDR2a","CDR3a","CDR3a_extended","Vb","Jb","TCRb","CDR1b","CDR2b","CDR3b","CDR3b_extended","Target"]
join = features_baseline[:-1]  #all but target, to match the test and ground truths

test_df = pd.read_csv("test.csv")             
pos_df  = pd.read_csv("vdjdb_positives.csv")


#test out clustering on non-3d metrics first



# #cluster the test data set with 3d features
# X_struct = test_data[features_3d]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_struct)

# kmeans = KMeans(n_clusters=TEST_CLUSTER_K, random_state=0)
# cluster_ids = kmeans.fit_predict(X_scaled)
# test_data["cluster_id"] = cluster_ids
# print(test_data[["index", "cluster_id"] + features_3d])