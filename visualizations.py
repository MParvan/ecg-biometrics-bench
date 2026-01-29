import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

class Visualizer:
    def __init__(self):
        pass

    def plot_embeddings(self, embeddings, labels, title="T-SNE of ECG Embeddings", save_path=None):
        """
        Projects high-dim embeddings (e.g., 128D) to 2D using T-SNE.
        """
        print("[INFO] Computing T-SNE... this may take a moment.")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        emb_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        # Plot only top 20 classes to avoid clutter if many subjects
        unique_labels = np.unique(labels)
        if len(unique_labels) > 20:
            subset_labels = unique_labels[:20]
            mask = np.isin(labels, subset_labels)
            sns.scatterplot(x=emb_2d[mask, 0], y=emb_2d[mask, 1], hue=labels[mask], palette="tab10", legend="full")
        else:
            sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels, palette="tab10", legend="full")
            
        plt.title(title)
        plt.xlabel("T-SNE Dim 1")
        plt.ylabel("T-SNE Dim 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"[INFO] Plot saved to {save_path}")
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes=None, normalize=True, save_path=None):
        """
        Plots a heatmap of the confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
        plt.title("Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_cmc(self, similarities, y_true, title="CMC Curve", save_path=None):
        """
        Plots Cumulative Match Characteristic (CMC) curve.
        
        Args:
            similarities: (N_query, N_gallery) matrix of similarity scores.
            y_true: (N_query,) array of indices of the correct gallery match.
                    (Assumes the gallery is sorted such that column J corresponds to subject J).
        """
        n_query = similarities.shape[0]
        n_gallery = similarities.shape[1]
        
        # Sort predictions (descending order of similarity)
        # indices of sorted matches
        sorted_indices = np.argsort(-similarities, axis=1)
        
        ranks = np.zeros(n_gallery)
        
        for i in range(n_query):
            # Find where the true label appears in the sorted list
            # logic depends on how y_true maps to columns. 
            # Assuming y_true[i] is the column index of the correct template:
            rank_pos = np.where(sorted_indices[i] == y_true[i])[0][0]
            ranks[rank_pos:] += 1
            
        cmc_scores = ranks / n_query
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, n_gallery + 1), cmc_scores, linewidth=2)
        plt.title(title)
        plt.xlabel("Rank (k)")
        plt.ylabel("Identification Rate (Rank-k)")
        plt.grid(True)
        plt.xlim([1, min(20, n_gallery)]) # Focus on Top-20
        plt.ylim([0, 1.05])
        
        if save_path:
            plt.savefig(save_path)
        plt.show()