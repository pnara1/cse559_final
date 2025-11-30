import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("both_test_scores.csv")

#printable info
print(df[["seq_only_score", "seq_3d_score"]].describe())
print(df[["seq_only_score", "seq_3d_score"]].corr())


#histogram of both scores
plt.figure(figsize=(8, 5))
plt.hist(df["seq_only_score"], bins=40, alpha=0.5, label="Seq only")
plt.hist(df["seq_3d_score"],   bins=40, alpha=0.5, label="Seq + 3D")
plt.xlabel("Score")
plt.ylabel("Count")
plt.title("Score distributions: Sequence-only vs Sequence + 3D")
plt.legend()
plt.savefig("histogram_scores.png")

#scatter plot comparing both scores
#above diagonal means seq+3d score is higher
#below diagonal means seq-only score is higher
plt.figure(figsize=(6, 6))
plt.scatter(df["seq_only_score"], df["seq_3d_score"], s=10, alpha=0.5)
plt.xlabel("Seq-only score")
plt.ylabel("Seq + 3D score")
plt.title("Per-sample comparison of scores")
min_val = min(df["seq_only_score"].min(), df["seq_3d_score"].min())
max_val = max(df["seq_only_score"].max(), df["seq_3d_score"].max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")
plt.tight_layout()
plt.savefig("scatter_scores.png")


