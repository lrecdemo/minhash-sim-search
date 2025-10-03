import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import adjusted_rand_score, v_measure_score
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = '0'

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from app.minhash_clustering.cluster_in_mem import MemMinhashLSHClustering
from app.minhash_clustering.cluster_streaming import StreamingMinHashLSHClustering

CSV_PATH = os.path.join(parent_dir, 'app', 'demo_data', 'paper_verses.csv')


def purity_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    clusters = np.unique(y_pred)
    total_correct = 0

    for cluster in clusters:
        cluster_mask = y_pred == cluster
        cluster_labels = y_true[cluster_mask]
        if len(cluster_labels) > 0:
            most_frequent_label = np.bincount(cluster_labels).argmax()
            total_correct += np.sum(cluster_labels == most_frequent_label)

    return total_correct / len(y_true)


def calculate_cluster_purities(labels, pred_labels):
    df_temp = pd.DataFrame({'idgroup': labels, 'cluster_id': pred_labels})

    group_purity = df_temp.groupby('idgroup').apply(
        lambda x: x['cluster_id'].value_counts().max() / len(x)
    )
    mean_group_purity = group_purity.mean()

    cluster_purity = df_temp.groupby('cluster_id').apply(
        lambda x: x['idgroup'].value_counts().max() / len(x)
    )
    mean_cluster_purity = cluster_purity.mean()

    num_true_groups = len(df_temp['idgroup'].unique())
    num_pred_clusters = len(df_temp['cluster_id'].unique())
    over_clustering_ratio = num_pred_clusters / num_true_groups

    perfect_group_purity_pct = (group_purity == 1.0).mean()
    perfect_cluster_purity_pct = (cluster_purity == 1.0).mean()

    return {
        'mean_group_purity': mean_group_purity,
        'mean_cluster_purity': mean_cluster_purity,
        'over_clustering_ratio': over_clustering_ratio,
        'num_pred_clusters': num_pred_clusters,
        'num_true_groups': num_true_groups,
        'perfect_group_purity_pct': perfect_group_purity_pct,
        'perfect_cluster_purity_pct': perfect_cluster_purity_pct
    }


def create_comprehensive_heatmaps(plot_data, experiment_name, results_dir):
    print(f"  Creating comprehensive heatmap for {experiment_name}...")

    metrics_list = [
        ('ari', 'ARI'),
        ('v_measure', 'V-measure'),
        ('purity', 'Purity'),
        ('mean_group_purity', 'Group Purity'),
        ('mean_cluster_purity', 'Cluster Purity'),
        ('over_clustering_ratio', 'Overclust. Ratio'),
        ('num_pred_clusters', 'Num Clusters'),
        ('perfect_group_purity_pct', 'Perfect Groups %')
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Clustering Quality Metrics - {experiment_name}', fontsize=16, fontweight='bold')

    for idx, (metric, title) in enumerate(metrics_list):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        pivot_data = plot_data.pivot(index='shingle_size', columns='jaccard_threshold', values=metric)

        if metric in ['over_clustering_ratio', 'num_pred_clusters']:
            cmap = 'RdYlGn_r'
        else:
            cmap = 'RdYlGn'

        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, ax=ax, cbar_kws={'label': title})
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Jaccard Threshold', fontsize=10)
        ax.set_ylabel('Shingle Size', fontsize=10)

    plt.tight_layout()
    filename = f'comprehensive_metrics_{experiment_name.lower().replace(" ", "_")}.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def create_single_heatmap(data, metric_type, preprocessing_suffix, results_dir):
    print(f"  Creating single heatmap: {metric_type} - {preprocessing_suffix}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if metric_type == 'ari_vmeasure':
        pivot_ari = data.pivot(index='shingle_size', columns='jaccard_threshold', values='ari')
        pivot_vm = data.pivot(index='shingle_size', columns='jaccard_threshold', values='v_measure')

        sns.heatmap(pivot_ari, annot=False, cmap='RdYlGn', ax=ax,
                    cbar_kws={'label': 'ARI Score'}, vmin=0, vmax=1)

        for i, shingle in enumerate(pivot_ari.index):
            for j, threshold in enumerate(pivot_ari.columns):
                ari_val = pivot_ari.loc[shingle, threshold]
                vm_val = pivot_vm.loc[shingle, threshold]
                text = f'{ari_val:.2f}\n{vm_val:.2f}'
                text_color = 'white' if ari_val < 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                        color=text_color, fontsize=10, weight='bold', linespacing=0.9)

        ax.set_title(f'ARI (top) / V-measure (bottom) - {preprocessing_suffix}',
                     fontweight='bold', fontsize=12, pad=6)
        filename = f'single_ari_vmeasure_{preprocessing_suffix}.png'

    else:
        pivot_purity = data.pivot(index='shingle_size', columns='jaccard_threshold',
                                  values='mean_group_purity')
        pivot_over = data.pivot(index='shingle_size', columns='jaccard_threshold',
                                values='over_clustering_ratio')

        sns.heatmap(pivot_purity, annot=False, cmap='RdYlGn', ax=ax,
                    cbar_kws={'label': 'Group Purity'}, vmin=0, vmax=1)

        for i, shingle in enumerate(pivot_purity.index):
            for j, threshold in enumerate(pivot_purity.columns):
                purity_val = pivot_purity.loc[shingle, threshold]
                over_val = pivot_over.loc[shingle, threshold]
                text = f'{purity_val:.2f}\n{over_val:.2f}'
                text_color = 'white' if purity_val < 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                        color=text_color, fontsize=10, weight='bold', linespacing=0.9)

        ax.set_title(f'Group Purity (top) / Overclust. Ratio (bottom) - {preprocessing_suffix}',
                     fontweight='bold', fontsize=12, pad=6)
        filename = f'single_purity_overclustering_{preprocessing_suffix}.png'

    ax.set_xlabel('Jaccard Threshold', fontsize=11)
    ax.set_ylabel('Shingle Size', fontsize=11)

    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def run_clustering_experiment_enhanced(texts, labels, experiment_name, method='optimized',
                                       preprocess_options=None):
    print(f"\nRunning {experiment_name} experiment with {method} method...")

    shingle_sizes = [2, 3, 4, 5, 6]
    jaccard_thresholds = [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

    plot_data = []

    for shingle_size in shingle_sizes:
        for threshold in jaccard_thresholds:
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)

            print(f"  Processing shingle_size={shingle_size}, threshold={threshold}")

            if method == 'optimized':
                clustering = MemMinhashLSHClustering(
                    threshold=threshold,
                    shingle_size=shingle_size,
                    preprocess_options=preprocess_options
                )
                clustered_docs = clustering.cluster_documents(texts)
                pred_labels = [doc.cluster_id for doc in clustered_docs]

            else:
                clustering = StreamingMinHashLSHClustering(
                    threshold=threshold,
                    shingle_size=shingle_size,
                    chunk_size=min(5000, max(1000, len(texts) // 20)),
                    preprocess_options=preprocess_options
                )
                cluster_results = clustering.cluster_streaming(texts)
                pred_labels = [cluster_results.get(i, 0) for i in range(len(texts))]
                clustering.cleanup()

            unique_clusters = len(set(pred_labels))
            print(f"    {method}: {unique_clusters} clusters, first 5 labels: {pred_labels[:5]}")

            ari = adjusted_rand_score(labels, pred_labels)
            v_measure = v_measure_score(labels, pred_labels)
            purity = purity_score(labels, pred_labels)

            cluster_metrics = calculate_cluster_purities(labels, pred_labels)

            plot_data.append({
                'shingle_size': shingle_size,
                'jaccard_threshold': threshold,
                'ari': ari,
                'v_measure': v_measure,
                'purity': purity,
                **cluster_metrics
            })

    return pd.DataFrame(plot_data)


def create_side_by_side_combined_heatmaps(data_dict, results_dir, suffix=''):
    print(f"\nCreating combined heatmaps{' (' + suffix + ')' if suffix else ''}...")

    methods = list(data_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(8 * n_methods, 6))
    if n_methods == 1:
        axes = [axes]

    for idx, (method, data) in enumerate(data_dict.items()):
        pivot_ari = data.pivot(index='shingle_size', columns='jaccard_threshold', values='ari')
        pivot_vm = data.pivot(index='shingle_size', columns='jaccard_threshold', values='v_measure')

        sns.heatmap(pivot_ari, annot=False, cmap='RdYlGn', ax=axes[idx],
                    cbar_kws={'label': 'ARI Score'}, vmin=0, vmax=1)

        for i, shingle in enumerate(pivot_ari.index):
            for j, threshold in enumerate(pivot_ari.columns):
                ari_val = pivot_ari.loc[shingle, threshold]
                vm_val = pivot_vm.loc[shingle, threshold]
                text = f'{ari_val:.2f}\n{vm_val:.2f}'
                text_color = 'white' if ari_val < 0.5 else 'black'
                axes[idx].text(j + 0.5, i + 0.5, text, ha='center', va='center',
                               color=text_color, fontsize=10, weight='bold', linespacing=0.9)

        axes[idx].set_title(f'{method}\nARI (top) / V-measure (bottom)',
                            fontweight='bold', fontsize=12, pad=6)
        axes[idx].set_xlabel('Jaccard Threshold', fontsize=11)
        axes[idx].set_ylabel('Shingle Size', fontsize=11)

    plt.tight_layout(pad=0.2)
    filename = f'sidebyside_ari_vmeasure{"_" + suffix if suffix else ""}.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, n_methods, figsize=(8 * n_methods, 6))
    if n_methods == 1:
        axes = [axes]

    for idx, (method, data) in enumerate(data_dict.items()):
        pivot_purity = data.pivot(index='shingle_size', columns='jaccard_threshold',
                                  values='mean_group_purity')
        pivot_over = data.pivot(index='shingle_size', columns='jaccard_threshold',
                                values='over_clustering_ratio')

        sns.heatmap(pivot_purity, annot=False, cmap='RdYlGn', ax=axes[idx],
                    cbar_kws={'label': 'Group Purity'}, vmin=0, vmax=1)

        for i, shingle in enumerate(pivot_purity.index):
            for j, threshold in enumerate(pivot_purity.columns):
                purity_val = pivot_purity.loc[shingle, threshold]
                over_val = pivot_over.loc[shingle, threshold]
                text = f'{purity_val:.2f}\n{over_val:.2f}'
                text_color = 'white' if purity_val < 0.5 else 'black'
                axes[idx].text(j + 0.5, i + 0.5, text, ha='center', va='center',
                               color=text_color, fontsize=10, weight='bold', linespacing=0.9)

        axes[idx].set_title(f'{method}\nGroup Purity (top) / Overclust. Ratio (bottom)',
                            fontweight='bold', fontsize=12, pad=6)
        axes[idx].set_xlabel('Jaccard Threshold', fontsize=11)
        axes[idx].set_ylabel('Shingle Size', fontsize=11)

    plt.tight_layout(pad=0.2)
    filename = f'sidebyside_purity_overclustering{"_" + suffix if suffix else ""}.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    print("Combined heatmaps created successfully!")


def create_method_difference_heatmaps(optimized_data, streaming_data, preprocessing_suffix, results_dir):
    print(f"  Creating method difference heatmaps for {preprocessing_suffix}...")

    metrics_to_compare = [
        ('ari', 'ARI Difference'),
        ('v_measure', 'V-measure Difference'),
        ('mean_group_purity', 'Group Purity Difference'),
        ('over_clustering_ratio', 'Overclustering Ratio Difference')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Streaming vs Optimized Method Differences - {preprocessing_suffix.title()}',
                 fontsize=16, fontweight='bold')

    for idx, (metric, title) in enumerate(metrics_to_compare):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        pivot_opt = optimized_data.pivot(index='shingle_size', columns='jaccard_threshold', values=metric)
        pivot_stream = streaming_data.pivot(index='shingle_size', columns='jaccard_threshold', values=metric)

        difference = pivot_stream - pivot_opt

        max_abs_diff = max(abs(difference.min().min()), abs(difference.max().max()))
        vmin = -max_abs_diff if max_abs_diff > 0 else -0.01
        vmax = max_abs_diff if max_abs_diff > 0 else 0.01

        sns.heatmap(difference, annot=True, fmt='.3f', cmap='RdBu_r', ax=ax,
                    center=0, vmin=vmin, vmax=vmax,
                    cbar_kws={'label': 'Streaming - Optimized'})

        ax.set_title(f'{title}\n(Streaming - Optimized)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Jaccard Threshold', fontsize=10)
        ax.set_ylabel('Shingle Size', fontsize=10)

    plt.tight_layout()
    filename = f'method_differences_{preprocessing_suffix}.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    mean_abs_differences = {}
    for metric, _ in metrics_to_compare:
        pivot_opt = optimized_data.pivot(index='shingle_size', columns='jaccard_threshold', values=metric)
        pivot_stream = streaming_data.pivot(index='shingle_size', columns='jaccard_threshold', values=metric)
        difference = pivot_stream - pivot_opt
        mean_abs_differences[metric] = abs(difference).values.mean()

    return mean_abs_differences


def save_comprehensive_summary(results_dict, results_dir, sample_info):
    summary_file = os.path.join(results_dir, 'clustering_comparison_summary.txt')

    with open(summary_file, 'w') as f:
        f.write("COMPREHENSIVE CLUSTERING QUALITY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total verses: {sample_info['sample_size']}\n")
        f.write(f"Number of Versegroups: {sample_info['num_groups']}\n")
        f.write(f"Random Seed: {sample_info['seed']}\n")
        f.write(f"Source File: {sample_info['source_file']}\n\n")

        for preprocessing_type, methods in results_dict.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"{preprocessing_type.upper()}\n")
            f.write(f"{'=' * 80}\n\n")

            for method_name, data in methods.items():
                f.write(f"\n{'-' * 60}\n")
                f.write(f"{method_name}\n")
                f.write(f"{'-' * 60}\n")

                best_ari = data.loc[data['ari'].idxmax()]
                best_vm = data.loc[data['v_measure'].idxmax()]
                best_purity = data.loc[data['purity'].idxmax()]

                f.write(f"\nBest ARI: {best_ari['ari']:.3f} ")
                f.write(f"(shingle={best_ari['shingle_size']}, threshold={best_ari['jaccard_threshold']})\n")

                f.write(f"Best V-measure: {best_vm['v_measure']:.3f} ")
                f.write(f"(shingle={best_vm['shingle_size']}, threshold={best_vm['jaccard_threshold']})\n")

                f.write(f"Best Purity: {best_purity['purity']:.3f} ")
                f.write(f"(shingle={best_purity['shingle_size']}, threshold={best_purity['jaccard_threshold']})\n")

                f.write(f"\nMean ARI: {data['ari'].mean():.3f}\n")
                f.write(f"Mean V-measure: {data['v_measure'].mean():.3f}\n")
                f.write(f"Mean Purity: {data['purity'].mean():.3f}\n")
                f.write(f"Mean Group Purity: {data['mean_group_purity'].mean():.3f}\n")
                f.write(f"Mean Overclustering Ratio: {data['over_clustering_ratio'].mean():.3f}\n")

        f.write(f"\n\n{'=' * 80}\n")
        f.write("COMPARATIVE ANALYSIS\n")
        f.write(f"{'=' * 80}\n\n")

        for preprocessing_type, methods in results_dict.items():
            if len(methods) > 1:
                f.write(f"\n{preprocessing_type}:\n")
                method_names = list(methods.keys())
                for i, method1 in enumerate(method_names):
                    for method2 in method_names[i + 1:]:
                        data1 = methods[method1]
                        data2 = methods[method2]

                        ari_diff = data2['ari'].mean() - data1['ari'].mean()
                        vm_diff = data2['v_measure'].mean() - data1['v_measure'].mean()
                        purity_diff = data2['purity'].mean() - data1['purity'].mean()

                        f.write(f"\n{method2} vs {method1}:\n")
                        f.write(f"  ARI difference: {ari_diff:+.3f}\n")
                        f.write(f"  V-measure difference: {vm_diff:+.3f}\n")
                        f.write(f"  Purity difference: {purity_diff:+.3f}\n")

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), 'benchmark_quality_results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"CSV file: {CSV_PATH}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 80)

    print("\nLoading data...")
    df = pd.read_csv(CSV_PATH)

    text_columns = [col for col in df.columns if 'text' in col.lower()]
    if not text_columns:
        raise ValueError("No 'text' column found in CSV")

    texts = df[text_columns[0]].dropna().astype(str).tolist()
    labels = df['idgroup'].tolist() if 'idgroup' in df.columns else list(range(len(texts)))

    print(f"\nDataset: {len(texts)} texts with {len(set(labels))} unique groups")

    sample_info = {
        'sample_size': len(texts),
        'num_groups': len(set(labels)),
        'seed': RANDOM_SEED,
        'source_file': CSV_PATH
    }

    no_preprocessing = {
        'lowercase': False,
        'remove_diacritics': False,
        'remove_punctuation': False
    }

    full_preprocessing = {
        'lowercase': True,
        'remove_diacritics': True,
        'remove_punctuation': True
    }

    all_results = {
        'Original Text (No Preprocessing)': {},
        'Preprocessed Text (Full Preprocessing)': {}
    }

    print("\n" + "=" * 80)
    print("ORIGINAL TEXT - NO PREPROCESSING")
    print("=" * 80)
    all_results['Original Text (No Preprocessing)']['Optimized'] = run_clustering_experiment_enhanced(
        texts, labels, "Original Text - Optimized",
        method='optimized', preprocess_options=no_preprocessing
    )
    all_results['Original Text (No Preprocessing)']['Streaming'] = run_clustering_experiment_enhanced(
        texts, labels, "Original Text - Streaming",
        method='streaming', preprocess_options=no_preprocessing
    )

    print("\n" + "=" * 80)
    print("PREPROCESSED TEXT - FULL PREPROCESSING")
    print("=" * 80)
    all_results['Preprocessed Text (Full Preprocessing)']['Optimized'] = run_clustering_experiment_enhanced(
        texts, labels, "Preprocessed Text - Optimized",
        method='optimized', preprocess_options=full_preprocessing
    )
    all_results['Preprocessed Text (Full Preprocessing)']['Streaming'] = run_clustering_experiment_enhanced(
        texts, labels, "Preprocessed Text - Streaming",
        method='streaming', preprocess_options=full_preprocessing
    )

    print("\nCreating comprehensive metric heatmaps...")
    for preprocessing_type, methods in all_results.items():
        for method_name, data in methods.items():
            create_comprehensive_heatmaps(data, f"{preprocessing_type} - {method_name}", results_dir)

    print("\nCreating side-by-side comparison visualizations...")
    create_side_by_side_combined_heatmaps(
        all_results['Original Text (No Preprocessing)'],
        results_dir,
        suffix='original'
    )
    create_side_by_side_combined_heatmaps(
        all_results['Preprocessed Text (Full Preprocessing)'],
        results_dir,
        suffix='preprocessed'
    )

    print("\nCreating single heatmaps for optimized method only...")
    create_single_heatmap(
        all_results['Original Text (No Preprocessing)']['Optimized'],
        'ari_vmeasure',
        'original',
        results_dir
    )
    create_single_heatmap(
        all_results['Original Text (No Preprocessing)']['Optimized'],
        'purity_overclustering',
        'original',
        results_dir
    )
    create_single_heatmap(
        all_results['Preprocessed Text (Full Preprocessing)']['Optimized'],
        'ari_vmeasure',
        'preprocessed',
        results_dir
    )
    create_single_heatmap(
        all_results['Preprocessed Text (Full Preprocessing)']['Optimized'],
        'purity_overclustering',
        'preprocessed',
        results_dir
    )

    print("\nCreating method difference heatmaps...")
    diff_original = create_method_difference_heatmaps(
        all_results['Original Text (No Preprocessing)']['Optimized'],
        all_results['Original Text (No Preprocessing)']['Streaming'],
        'original',
        results_dir
    )
    diff_preprocessed = create_method_difference_heatmaps(
        all_results['Preprocessed Text (Full Preprocessing)']['Optimized'],
        all_results['Preprocessed Text (Full Preprocessing)']['Streaming'],
        'preprocessed',
        results_dir
    )

    print("\nMethod Difference Summary:")
    print("\nOriginal Text (No Preprocessing):")
    for metric, mean_abs_diff in diff_original.items():
        print(f"  Mean absolute {metric} difference: {mean_abs_diff:.4f}")

    print("\nPreprocessed Text (Full Preprocessing):")
    for metric, mean_abs_diff in diff_preprocessed.items():
        print(f"  Mean absolute {metric} difference: {mean_abs_diff:.4f}")

    save_comprehensive_summary(all_results, results_dir, sample_info)

    print(f"\nAll results saved to: {results_dir}")