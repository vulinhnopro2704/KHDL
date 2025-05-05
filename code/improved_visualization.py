import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler

# Path to the input data file
input_file = '../data/clean/feature_engineering_encoding.csv'

# Path for saving visualizations 
viz_dir = '../visualizations/improved'
os.makedirs(viz_dir, exist_ok=True)

# Utility functions for visualization
def create_heatmap_encoding_comparison(df, cat_col, encoded_col, target_col='price', n_groups=10):
    """
    Create a heatmap showing the relationship between categories, their encoded values,
    and the target variable.
    """
    # Group categories by their frequency
    value_counts = df[cat_col].value_counts()
    
    # Get top n_groups-1 categories and group the rest as "Others"
    top_categories = value_counts.nlargest(n_groups-1).index.tolist()
    
    # Create a copy of dataframe with grouped categories
    df_grouped = df.copy()
    df_grouped[f'{cat_col}_grouped'] = df_grouped[cat_col].apply(
        lambda x: x if x in top_categories else 'Others'
    )
    
    # Calculate average encoded value and target value for each category
    grouped_stats = df_grouped.groupby(f'{cat_col}_grouped').agg({
        encoded_col: 'mean',
        target_col: 'mean',
        cat_col: 'count'  # count for sizing
    }).reset_index()
    
    # Sort by encoded value
    grouped_stats = grouped_stats.sort_values(by=encoded_col)
    
    # Normalize sizes for bubble chart
    if len(grouped_stats) > 1:
        scaler = MinMaxScaler(feature_range=(50, 300))
        grouped_stats['size'] = scaler.fit_transform(
            grouped_stats[cat_col].values.reshape(-1, 1)
        ).flatten()
    else:
        grouped_stats['size'] = 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with size representing frequency
    scatter = ax.scatter(
        grouped_stats[encoded_col], 
        grouped_stats[target_col],
        s=grouped_stats['size'],
        c=range(len(grouped_stats)),
        cmap='viridis',
        alpha=0.7
    )
    
    # Add category labels
    for i, row in grouped_stats.iterrows():
        ax.annotate(
            row[f'{cat_col}_grouped'],
            (row[encoded_col], row[target_col]),
            ha='center', va='center',
            fontsize=9,
            color='white' if i > len(grouped_stats) / 2 else 'black'
        )
    
    # Add title and labels
    ax.set_title(f'Relationship between {cat_col}, its encoded values, and {target_col}', fontsize=14)
    ax.set_xlabel(f'{encoded_col} (Target Encoded Value)', fontsize=12)
    ax.set_ylabel(f'{target_col}', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Category Index')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    return fig

def create_multiple_small_plots(df, cat_col, encoded_col, target_col='price', n_per_plot=20, max_plots=5):
    """
    Create multiple small plots to show all categories across several charts
    """
    # Get all unique categories sorted by frequency
    categories = df[cat_col].value_counts().index.tolist()
    
    # Calculate number of plots needed
    n_plots = min(max_plots, (len(categories) + n_per_plot - 1) // n_per_plot)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5 * n_plots))
    
    # Create GridSpec for better control over subplot sizes
    gs = gridspec.GridSpec(n_plots, 1, height_ratios=[1] * n_plots)
    
    for i in range(n_plots):
        # Get subset of categories for this plot
        start_idx = i * n_per_plot
        end_idx = min(start_idx + n_per_plot, len(categories))
        plot_categories = categories[start_idx:end_idx]
        
        # Filter dataframe to only these categories
        df_subset = df[df[cat_col].isin(plot_categories)]
        
        # Create subplot
        ax = plt.subplot(gs[i])
        
        # Create boxplot
        sns.boxplot(x=cat_col, y=target_col, data=df_subset, ax=ax, palette='viridis')
        
        # Plot encoded values as points
        means = df_subset.groupby(cat_col)[encoded_col].mean()
        for j, cat in enumerate(plot_categories):
            if cat in means:
                ax.scatter(j, means[cat], color='red', s=100, marker='*', 
                           label='Encoded Value' if i == 0 and j == 0 else "")
        
        # Formatting
        ax.set_title(f'Categories {start_idx+1} to {end_idx} of {len(categories)}', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        
        if i == 0:
            ax.legend()
        
        # Only add x-axis label to the bottom plot
        if i == n_plots - 1:
            ax.set_xlabel(cat_col, fontsize=12)
        else:
            ax.set_xlabel('')
            
        ax.set_ylabel(target_col, fontsize=12)
    
    plt.tight_layout()
    
    return fig

def create_hierarchical_clustering_heatmap(df, cat_col, encoded_col, target_col='price', max_categories=50):
    """
    Create a hierarchical clustering heatmap of categories and their relationship with target
    """
    # Get top categories by frequency
    top_categories = df[cat_col].value_counts().nlargest(max_categories).index.tolist()
    
    # Filter dataframe to only these categories
    df_subset = df[df[cat_col].isin(top_categories)]
    
    # Calculate the mean values for each category
    pivot_data = df_subset.groupby(cat_col).agg({
        encoded_col: 'mean',
        target_col: 'mean',
        cat_col: 'count'
    }).rename(columns={cat_col: 'count'})
    
    # Normalize the data for better visualization
    normalized_data = pivot_data.copy()
    for col in normalized_data.columns:
        if col != 'count':
            normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                                  (normalized_data[col].max() - normalized_data[col].min())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#76a7e5', '#1e4c99'])
    
    # Create heatmap with hierarchical clustering
    sns.clustermap(normalized_data.drop('count', axis=1),
                   figsize=(14, 10),
                   cmap=cmap,
                   method='average',
                   metric='euclidean',
                   row_cluster=True,
                   col_cluster=False,
                   linewidths=0.5,
                   linecolor='gray',
                   dendrogram_ratio=0.2,
                   cbar_pos=(0.02, 0.8, 0.05, 0.18))
    
    plt.suptitle(f'Hierarchical Clustering of {cat_col} by Encoded Value and {target_col}', 
                 fontsize=16, y=1.02)
    
    return fig

def visualize_value_distribution(df, cat_col, encoded_col, target_col='price'):
    """
    Visualize the distribution of values in a categorical variable and their encoded values
    """
    # Create a dataframe with counts and means
    value_stats = df.groupby(cat_col).agg({
        cat_col: 'count',
        encoded_col: 'mean',
        target_col: 'mean'
    }).rename(columns={cat_col: 'count'}).reset_index()
    
    # Sort by frequency
    value_stats = value_stats.sort_values('count', ascending=False)
    
    # Create three subplots: frequency, encoded value, and target value
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # 1. Plot frequency distribution (histogram)
    sns.barplot(x=cat_col, y='count', data=value_stats.head(30), ax=axes[0], palette='Greens_r')
    axes[0].set_title(f'Top 30 Most Frequent Values of {cat_col}', fontsize=14)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_xlabel('')
    
    # 2. Plot encoded value by category
    sns.barplot(x=cat_col, y=encoded_col, data=value_stats.head(30), ax=axes[1], palette='Blues_r')
    axes[1].set_title(f'Target Encoded Values for Top 30 {cat_col} Categories', fontsize=14)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
    axes[1].set_ylabel(f'{encoded_col}', fontsize=12)
    axes[1].set_xlabel('')
    
    # 3. Plot target value by category
    sns.barplot(x=cat_col, y=target_col, data=value_stats.head(30), ax=axes[2], palette='Reds_r')
    axes[2].set_title(f'Average {target_col} for Top 30 {cat_col} Categories', fontsize=14)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=90)
    axes[2].set_ylabel(f'Average {target_col}', fontsize=12)
    axes[2].set_xlabel(cat_col, fontsize=12)
    
    plt.tight_layout()
    
    return fig

def visualize_encoding_correlation(df, encoded_cols, target_col='price'):
    """
    Visualize correlation between encoded features and target
    """
    # Select encoded columns and target
    cols_to_include = encoded_cols + [target_col]
    corr_matrix = df[cols_to_include].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, ax=ax, vmin=-1, vmax=1, center=0)
    
    # Add title
    ax.set_title('Correlation between Encoded Features and Target', fontsize=14)
    
    plt.tight_layout()
    
    return fig

# Main script execution
if __name__ == "__main__":
    # Read the data
    print(f"Reading dataset from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found. Make sure to run target_encoding.py first.")
        exit(1)
    
    # List of categorical columns and their encoded versions
    cat_encoded_pairs = [
        ('chip_model', 'chip_model_encoded'),
        ('brand', 'brand_encoded'),
        ('screen_tech', 'screen_tech_encoded'),
        ('screen_resolution_k', 'screen_resolution_k_encoded')
    ]
    
    # List to store all encoded columns
    encoded_cols = [pair[1] for pair in cat_encoded_pairs]
    
    # Create visualizations
    for cat_col, encoded_col in cat_encoded_pairs:
        print(f"\nCreating visualizations for {cat_col}...")
        
        # 1. Create bubbleplot with 'Others' category
        print(f"  Creating bubble plot with 'Others' category...")
        fig_bubble = create_heatmap_encoding_comparison(
            df, cat_col, encoded_col, 'price', 
            n_groups=min(df[cat_col].nunique(), 20)
        )
        fig_bubble.savefig(os.path.join(viz_dir, f"{cat_col}_bubble_plot.png"))
        plt.close(fig_bubble)
        
        # 2. Create multiple small plots (if many categories)
        if df[cat_col].nunique() > 20:
            print(f"  Creating multiple small plots for all categories...")
            fig_multi = create_multiple_small_plots(
                df, cat_col, encoded_col, 'price', 
                n_per_plot=20, max_plots=min(5, (df[cat_col].nunique() + 19) // 20)
            )
            fig_multi.savefig(os.path.join(viz_dir, f"{cat_col}_multiple_plots.png"))
            plt.close(fig_multi)
        
        # 3. Create hierarchical clustering (if many categories)
        if df[cat_col].nunique() >= 10:
            print(f"  Creating hierarchical clustering heatmap...")
            fig_cluster = create_hierarchical_clustering_heatmap(
                df, cat_col, encoded_col, 'price', 
                max_categories=min(df[cat_col].nunique(), 50)
            )
            fig_cluster.savefig(os.path.join(viz_dir, f"{cat_col}_hierarchical_cluster.png"))
            plt.close(fig_cluster)
        
        # 4. Create value distribution visualization
        print(f"  Creating value distribution visualization...")
        fig_dist = visualize_value_distribution(df, cat_col, encoded_col, 'price')
        fig_dist.savefig(os.path.join(viz_dir, f"{cat_col}_value_distribution.png"))
        plt.close(fig_dist)
    
    # 5. Create correlation heatmap for all encoded features
    print("\nCreating correlation heatmap for encoded features...")
    fig_corr = visualize_encoding_correlation(df, encoded_cols, 'price')
    fig_corr.savefig(os.path.join(viz_dir, "encoding_correlation_heatmap.png"))
    plt.close(fig_corr)
    
    print(f"\nAll visualizations have been saved to {viz_dir}")
    print("\nSummary of visualizations created:")
    print("  1. Bubble plots - Categories grouped with 'Others' category")
    print("  2. Multiple small plots - All categories shown across multiple charts")
    print("  3. Hierarchical clustering - Categories clustered by encoding similarity")
    print("  4. Value distribution - Frequency, encoding, and target value by category")
    print("  5. Correlation heatmap - Correlation between encoded features and target")