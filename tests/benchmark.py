
# benchmark.py
import time
import pandas as pd
import numpy as np
from main import MinHashLSHClustering
import matplotlib.pyplot as plt


def generate_test_data(n_docs: int, n_clusters: int = 5) -> list:
    """Generate synthetic test data with known clusters."""
    np.random.seed(42)

    # Base templates for each cluster
    templates = [
        "machine learning artificial intelligence data science algorithms",
        "weather sunny cloudy rainy temperature forecast meteorology",
        "programming python java coding software development",
        "finance banking investment stocks market economy",
        "healthcare medical hospital doctor patient treatment"
    ]

    texts = []
    for i in range(n_docs):
        cluster_idx = i % n_clusters
        base = templates[cluster_idx]

        # Add some variation
        words = base.split()
        np.random.shuffle(words)

        # Random subset and add noise
        subset_size = max(3, len(words) - np.random.randint(0, 2))
        selected_words = words[:subset_size]

        if np.random.random() > 0.8:  # 20% noise
            noise_words = ["random", "noise", "word", "text", "document"]
            selected_words.extend(np.random.choice(noise_words, size=2))

        texts.append(" ".join(selected_words))

    return texts


def benchmark_clustering_performance():
    """Benchmark clustering performance across different dataset sizes."""
    sizes = [100, 500, 1000, 2000, 5000]
    thresholds = [0.1, 0.3, 0.5, 0.7]

    results = []

    for size in sizes:
        print(f"Testing dataset size: {size}")
        texts = generate_test_data(size)

        for threshold in thresholds:
            print(f"  Threshold: {threshold}")

            clusterer = MinHashLSHClustering(threshold=threshold)

            start_time = time.time()
            clustered_docs = clusterer.cluster_documents(texts)
            end_time = time.time()

            processing_time = end_time - start_time
            num_clusters = len(set(doc.cluster_id for doc in clustered_docs))
            avg_certainty = np.mean([doc.certainty for doc in clustered_docs])

            results.append({
                'size': size,
                'threshold': threshold,
                'time': processing_time,
                'clusters': num_clusters,
                'avg_certainty': avg_certainty
            })

            print(f"    Time: {processing_time:.2f}s, Clusters: {num_clusters}, Avg Certainty: {avg_certainty:.3f}")

    return pd.DataFrame(results)


def plot_benchmark_results(results_df: pd.DataFrame):
    """Plot benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Processing time vs dataset size
    for threshold in results_df['threshold'].unique():
        data = results_df[results_df['threshold'] == threshold]
        axes[0, 0].plot(data['size'], data['time'], marker='o', label=f'Threshold {threshold}')

    axes[0, 0].set_xlabel('Dataset Size')
    axes[0, 0].set_ylabel('Processing Time (s)')
    axes[0, 0].set_title('Processing Time vs Dataset Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Number of clusters vs threshold
    for size in [1000, 2000, 5000]:
        if size in results_df['size'].values:
            data = results_df[results_df['size'] == size]
            axes[0, 1].plot(data['threshold'], data['clusters'], marker='s', label=f'Size {size}')

    axes[0, 1].set_xlabel('Jaccard Threshold')
    axes[0, 1].set_ylabel('Number of Clusters')
    axes[0, 1].set_title('Clusters vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Average certainty vs threshold
    for size in [1000, 2000, 5000]:
        if size in results_df['size'].values:
            data = results_df[results_df['size'] == size]
            axes[1, 0].plot(data['threshold'], data['avg_certainty'], marker='^', label=f'Size {size}')

    axes[1, 0].set_xlabel('Jaccard Threshold')
    axes[1, 0].set_ylabel('Average Certainty')
    axes[1, 0].set_title('Average Certainty vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Time complexity analysis
    largest_threshold_data = results_df[results_df['threshold'] == 0.3]
    axes[1, 1].loglog(largest_threshold_data['size'], largest_threshold_data['time'], 'bo-')
    axes[1, 1].set_xlabel('Dataset Size (log scale)')
    axes[1, 1].set_ylabel('Processing Time (log scale)')
    axes[1, 1].set_title('Time Complexity Analysis')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_quality_evaluation():
    """Evaluate clustering quality on known ground truth."""
    print("Running clustering quality evaluation...")

    # Generate data with known clusters
    texts = generate_test_data(1000, n_clusters=5)
    true_labels = [i % 5 for i in range(1000)]

    thresholds = np.arange(0.1, 0.8, 0.1)
    quality_metrics = []

    for threshold in thresholds:
        clusterer = MinHashLSHClustering(threshold=threshold)
        clustered_docs = clusterer.cluster_documents(texts)

        predicted_labels = [doc.cluster_id for doc in clustered_docs]

        # Calculate adjusted rand index (ARI)
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        num_clusters = len(set(predicted_labels))
        avg_certainty = np.mean([doc.certainty for doc in clustered_docs])

        quality_metrics.append({
            'threshold': threshold,
            'ari': ari,
            'nmi': nmi,
            'num_clusters': num_clusters,
            'avg_certainty': avg_certainty
        })

        print(f"Threshold {threshold:.1f}: ARI={ari:.3f}, NMI={nmi:.3f}, Clusters={num_clusters}")

    return pd.DataFrame(quality_metrics)


if __name__ == "__main__":
    print("Starting MinHash-LSH Clustering Benchmark...")

    # Run performance benchmark
    print("\n=== Performance Benchmark ===")
    perf_results = benchmark_clustering_performance()

    # Run quality evaluation
    print("\n=== Quality Evaluation ===")
    quality_results = run_quality_evaluation()

    # Plot results
    print("\n=== Generating Plots ===")
    plot_benchmark_results(perf_results)

    # Save results
    perf_results.to_csv('benchmark_performance.csv', index=False)
    quality_results.to_csv('benchmark_quality.csv', index=False)

    print("Benchmark completed! Results saved to CSV files and plots.")

