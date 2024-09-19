from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def visualize_pca(N, pca):
# plot the explained variances
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color = 'tab:blue'
    ax1.bar(1+np.arange(N), pca.explained_variance_ratio_, color=color)
    ax1.set_xticks(1+np.arange(N, step=2))
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel("Explained variance ratio", color=color)
    ax1.set_xlabel("Generated feature")

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(1+np.arange(N), np.cumsum(pca.explained_variance_ratio_), color=color)
    ax2.set_ylabel("Cumulative explained variance ratio", color=color)
    fig.tight_layout()
    # plt.show()
    plt.savefig('pca.png')
