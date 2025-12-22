"""
s_05_visualize.py

Generate comprehensive visualizations for RAG experiment analysis.

Creates:
1. Precision@K comparison across pipelines
2. Recall@K comparison across pipelines
3. Query type performance (Factual vs Conceptual)
4. Heatmap of success rates
5. K value impact analysis
6. Pipeline comparison bar charts

All plots saved to: outputs/analysis/plots/
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ANALYSIS_DIR = os.path.join(BASE_DIR, "outputs", "analysis")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_analysis_data() -> tuple:
    """Load analysis results from JSON files."""
    
    with open(os.path.join(ANALYSIS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
    
    with open(os.path.join(ANALYSIS_DIR, "aggregates.json"), "r") as f:
        aggregates = json.load(f)
    
    with open(os.path.join(ANALYSIS_DIR, "comparisons.json"), "r") as f:
        comparisons = json.load(f)
    
    return metrics, aggregates, comparisons


def metrics_to_dataframe(metrics: List[Dict]) -> pd.DataFrame:
    """Convert metrics list to pandas DataFrame for easy plotting."""
    return pd.DataFrame(metrics)


# ============================================================
# Plot 1: Precision@K by Pipeline
# ============================================================

def plot_precision_at_k(df: pd.DataFrame):
    """Plot precision at different K values for each pipeline."""
    
    plt.figure(figsize=(12, 6))
    
    # Group by pipeline and K, compute mean precision
    grouped = df.groupby(['pipeline', 'k'])['file_precision'].mean().reset_index()
    
    sns.lineplot(
        data=grouped,
        x='k',
        y='file_precision',
        hue='pipeline',
        marker='o',
        markersize=8,
        linewidth=2.5
    )
    
    plt.title('Precision@K Comparison Across Pipelines', fontsize=16, fontweight='bold')
    plt.xlabel('K (Number of Retrieved Chunks)', fontsize=12)
    plt.ylabel('File Precision', fontsize=12)
    plt.legend(title='Pipeline', fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOTS_DIR, 'precision_at_k.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created: precision_at_k.png")


# ============================================================
# Plot 2: Recall@K by Pipeline
# ============================================================

def plot_recall_at_k(df: pd.DataFrame):
    """Plot recall at different K values for each pipeline."""
    
    plt.figure(figsize=(12, 6))
    
    grouped = df.groupby(['pipeline', 'k'])['file_recall'].mean().reset_index()
    
    sns.lineplot(
        data=grouped,
        x='k',
        y='file_recall',
        hue='pipeline',
        marker='s',
        markersize=8,
        linewidth=2.5
    )
    
    plt.title('Recall@K Comparison Across Pipelines', fontsize=16, fontweight='bold')
    plt.xlabel('K (Number of Retrieved Chunks)', fontsize=12)
    plt.ylabel('File Recall', fontsize=12)
    plt.legend(title='Pipeline', fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOTS_DIR, 'recall_at_k.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created: recall_at_k.png")


# ============================================================
# Plot 3: Query Type Performance
# ============================================================

def plot_query_type_performance(df: pd.DataFrame):
    """Compare factual vs conceptual query performance by representation."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Precision by query type
    sns.boxplot(
        data=df,
        x='query_type',
        y='file_precision',
        hue='representation',
        ax=axes[0]
    )
    axes[0].set_title('Precision: Factual vs Conceptual Queries', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Query Type', fontsize=12)
    axes[0].set_ylabel('File Precision', fontsize=12)
    axes[0].legend(title='Representation')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Recall by query type
    sns.boxplot(
        data=df,
        x='query_type',
        y='file_recall',
        hue='representation',
        ax=axes[1]
    )
    axes[1].set_title('Recall: Factual vs Conceptual Queries', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Query Type', fontsize=12)
    axes[1].set_ylabel('File Recall', fontsize=12)
    axes[1].legend(title='Representation')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'query_type_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created: query_type_performance.png")


# ============================================================
# Plot 4: Success Rate Heatmap
# ============================================================

def plot_success_heatmap(df: pd.DataFrame):
    """Heatmap showing file retrieval success rate by chunking and representation."""
    
    # Compute success rate (queries where at least 1 correct file was retrieved)
    df['success'] = (df['correct_files'] > 0).astype(int)
    
    pivot = df.groupby(['chunking', 'representation'])['success'].mean().reset_index()
    pivot_table = pivot.pivot(index='chunking', columns='representation', values='success')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        cbar_kws={'label': 'Success Rate'},
        vmin=0,
        vmax=1,
        linewidths=1,
        linecolor='white'
    )
    
    plt.title('File Retrieval Success Rate by Pipeline', fontsize=16, fontweight='bold')
    plt.xlabel('Representation', fontsize=12)
    plt.ylabel('Chunking Method', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOTS_DIR, 'success_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created: success_heatmap.png")


# ============================================================
# Plot 5: K Value Impact (Dual Axis)
# ============================================================

def plot_k_impact_dual_axis(df: pd.DataFrame):
    """Show how precision and recall change with K (dual y-axis)."""
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    grouped = df.groupby('k').agg({
        'file_precision': 'mean',
        'file_recall': 'mean'
    }).reset_index()
    
    # Precision on left axis
    color = 'tab:blue'
    ax1.set_xlabel('K (Number of Retrieved Chunks)', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12, color=color)
    ax1.plot(grouped['k'], grouped['file_precision'], 
             marker='o', linewidth=2.5, markersize=10, color=color, label='Precision')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Recall on right axis
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Recall', fontsize=12, color=color)
    ax2.plot(grouped['k'], grouped['file_recall'], 
             marker='s', linewidth=2.5, markersize=10, color=color, label='Recall')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Precision-Recall Trade-off vs K', fontsize=16, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'k_impact_dual_axis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created: k_impact_dual_axis.png")


# ============================================================
# Plot 6: Pipeline Comparison Bar Chart
# ============================================================

def plot_pipeline_comparison(aggregates: Dict[str, Any]):
    """Bar chart comparing overall performance of each pipeline."""
    
    pipelines = aggregates['by_pipeline']
    
    data = {
        'Pipeline': list(pipelines.keys()),
        'Precision': [v['avg_file_precision'] for v in pipelines.values()],
        'Recall': [v['avg_file_recall'] for v in pipelines.values()]
    }
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['Precision'], width, label='Precision', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, df['Recall'], width, label='Recall', color='darkorange', alpha=0.8)
    
    ax.set_xlabel('Pipeline', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Pipeline Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Pipeline'], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for i, (prec, rec) in enumerate(zip(df['Precision'], df['Recall'])):
        ax.text(i - width/2, prec + 0.02, f'{prec:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, rec + 0.02, f'{rec:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'pipeline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created: pipeline_comparison.png")


# ============================================================
# Plot 7: Chunking Method Comparison
# ============================================================

def plot_chunking_comparison(df: pd.DataFrame):
    """Compare fixed vs semantic chunking across different metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precision by chunking
    sns.barplot(data=df, x='chunking', y='file_precision', ax=axes[0, 0], 
                palette='Set2', ci=95)
    axes[0, 0].set_title('Precision by Chunking Method', fontweight='bold')
    axes[0, 0].set_ylabel('File Precision')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Recall by chunking
    sns.barplot(data=df, x='chunking', y='file_recall', ax=axes[0, 1], 
                palette='Set2', ci=95)
    axes[0, 1].set_title('Recall by Chunking Method', fontweight='bold')
    axes[0, 1].set_ylabel('File Recall')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Correct files by chunking
    sns.barplot(data=df, x='chunking', y='correct_files', ax=axes[1, 0], 
                palette='Set2', ci=95)
    axes[1, 0].set_title('Average Correct Files Retrieved', fontweight='bold')
    axes[1, 0].set_ylabel('Correct Files')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Answer length by chunking
    sns.boxplot(data=df, x='chunking', y='answer_length', ax=axes[1, 1], 
                palette='Set2')
    axes[1, 1].set_title('Answer Length Distribution', fontweight='bold')
    axes[1, 1].set_ylabel('Answer Length (chars)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Fixed vs Semantic Chunking: Comprehensive Comparison', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'chunking_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created: chunking_comparison.png")


# ============================================================
# Plot 8: Representation Comparison (BM25 vs Dense)
# ============================================================

def plot_representation_comparison(df: pd.DataFrame):
    """Compare BM25 vs Dense embeddings for factual and conceptual queries."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Factual queries
    factual = df[df['query_type'] == 'factual']
    fact_grouped = factual.groupby('representation').agg({
        'file_precision': 'mean',
        'file_recall': 'mean'
    }).reset_index()
    
    x = np.arange(len(fact_grouped))
    width = 0.35
    axes[0].bar(x - width/2, fact_grouped['file_precision'], width, 
                label='Precision', color='steelblue', alpha=0.8)
    axes[0].bar(x + width/2, fact_grouped['file_recall'], width, 
                label='Recall', color='darkorange', alpha=0.8)
    axes[0].set_title('Factual Queries: BM25 vs Dense', fontweight='bold', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fact_grouped['representation'])
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 1.1)
    
    # Conceptual queries
    conceptual = df[df['query_type'] == 'conceptual']
    conc_grouped = conceptual.groupby('representation').agg({
        'file_precision': 'mean',
        'file_recall': 'mean'
    }).reset_index()
    
    x = np.arange(len(conc_grouped))
    axes[1].bar(x - width/2, conc_grouped['file_precision'], width, 
                label='Precision', color='steelblue', alpha=0.8)
    axes[1].bar(x + width/2, conc_grouped['file_recall'], width, 
                label='Recall', color='darkorange', alpha=0.8)
    axes[1].set_title('Conceptual Queries: BM25 vs Dense', fontweight='bold', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conc_grouped['representation'])
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 1.1)
    
    plt.suptitle('BM25 vs Dense Embeddings by Query Type', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'representation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created: representation_comparison.png")


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(PLOTS_DIR)
    
    print("Loading analysis data...")
    metrics, aggregates, comparisons = load_analysis_data()
    
    print("Converting to DataFrame...")
    df = metrics_to_dataframe(metrics)
    
    print("\nGenerating visualizations...")
    print("-" * 60)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Generate all plots
    plot_precision_at_k(df)
    plot_recall_at_k(df)
    plot_query_type_performance(df)
    plot_success_heatmap(df)
    plot_k_impact_dual_axis(df)
    plot_pipeline_comparison(aggregates)
    plot_chunking_comparison(df)
    plot_representation_comparison(df)
    
    print("-" * 60)
    print(f"\n✅ All visualizations created successfully!")
    print(f"📁 Saved to: {PLOTS_DIR}")
    print("\nGenerated plots:")
    print("  1. precision_at_k.png - Precision across K values")
    print("  2. recall_at_k.png - Recall across K values")
    print("  3. query_type_performance.png - Factual vs Conceptual")
    print("  4. success_heatmap.png - Success rate heatmap")
    print("  5. k_impact_dual_axis.png - Precision-Recall trade-off")
    print("  6. pipeline_comparison.png - Overall pipeline performance")
    print("  7. chunking_comparison.png - Fixed vs Semantic")
    print("  8. representation_comparison.png - BM25 vs Dense")


if __name__ == "__main__":
    main()