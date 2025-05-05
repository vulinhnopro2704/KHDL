import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

def target_encode_with_regularization(df, column, target, alpha=5, min_samples=10):
    """
    Performs target encoding with regularization based on category frequency.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    column : str
        The name of the categorical column to encode
    target : str
        The name of the target column
    alpha : float, default=5
        Regularization parameter - higher means more regularization
    min_samples : int, default=10
        Minimum number of samples for a category to be considered reliable
        
    Returns:
    --------
    pandas Series
        The encoded values for the column
    """
    # Calculate the global mean
    global_mean = df[target].mean()
    
    # Calculate category-level statistics
    stats = df.groupby(column)[target].agg(['mean', 'count']).reset_index()
    
    # Apply regularization formula:
    # encoded_value = (count * category_mean + alpha * global_mean) / (count + alpha)
    stats['encoded'] = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)
    
    # Create a mapping dictionary
    encoding_map = dict(zip(stats[column], stats['encoded']))
    
    # For rare categories with fewer samples than min_samples, use global mean
    for cat in encoding_map:
        count = stats.loc[stats[column] == cat, 'count'].values[0]
        if count < min_samples:
            encoding_map[cat] = global_mean
    
    # Apply the mapping to the original data
    encoded_values = df[column].map(encoding_map).fillna(global_mean)
    
    return encoded_values

def target_encode_kfold(df, column, target, n_fold=5, alpha=5):
    """
    Performs target encoding with k-fold cross-validation to prevent data leakage.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    column : str
        The name of the categorical column to encode
    target : str
        The name of the target column
    n_fold : int, default=5
        Number of folds for cross-validation
    alpha : float, default=5
        Regularization parameter
        
    Returns:
    --------
    pandas Series
        The encoded values for the column
    """
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Calculate the global mean
    global_mean = df_copy[target].mean()
    
    # Create the output Series
    encoded = pd.Series(index=df_copy.index)
    
    # Split the data into n_fold parts
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(df_copy):
        # Calculate the mean target for each category in the training fold
        means = df_copy.iloc[train_idx].groupby(column)[target].agg(['mean', 'count'])
        
        # Apply regularization
        smoothed_means = ((means['mean'] * means['count']) + (global_mean * alpha)) / (means['count'] + alpha)
        
        # Map the means to the test fold
        encoded.iloc[test_idx] = df_copy.iloc[test_idx][column].map(smoothed_means)
    
    # Handle any categories that weren't seen in the training data (use global mean)
    encoded.fillna(global_mean, inplace=True)
    
    return encoded

def plot_encoding_impact(df, original_col, encoded_col, target, n_samples=10):
    """
    Plot the relationship between the original categories, their encoded values,
    and the target variable.
    """
    # Sample a subset of data for visualization
    if len(df) > n_samples:
        df_sample = df.sample(n_samples, random_state=42)
    else:
        df_sample = df
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original categories vs Target
    sns.boxplot(x=original_col, y=target, data=df_sample, ax=ax1)
    ax1.set_title(f'{original_col} vs {target}')
    ax1.set_xlabel(original_col)
    ax1.set_ylabel(target)
    ax1.tick_params(axis='x', rotation=90)
    
    # Plot 2: Encoded values vs Target
    sns.scatterplot(x=encoded_col, y=target, data=df_sample, ax=ax2)
    ax2.set_title(f'{encoded_col} vs {target}')
    ax2.set_xlabel(encoded_col)
    ax2.set_ylabel(target)
    
    plt.tight_layout()
    return fig

# Path to the input data file
input_file = '../data/clean/feature_engineering.csv'

# Path to output file
output_dir = '../data/clean/'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'feature_engineering_encoding.csv')

# Path for saving visualizations
viz_dir = '../visualizations'
os.makedirs(viz_dir, exist_ok=True)

# Read the dataset
print(f"Reading dataset from {input_file}...")
df = pd.read_csv(input_file)
print(f"Dataset shape: {df.shape}")

# Check the number of distinct values in each column
print("\nNumber of distinct values in each column:")
print(f"Chip model: {df['chip_model'].nunique()} distinct values")
print(f"Brand: {df['brand'].nunique()} distinct values")
print(f"Screen tech: {df['screen_tech'].nunique()} distinct values")
print(f"Screen resolution k: {df['screen_resolution_k'].nunique()} distinct values")

print("\nApplying target encoding to categorical columns...")

# Split the data into train and test sets to demonstrate proper encoding workflow
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Target encoding for chip_model (high cardinality)
print("\nEncoding chip_model...")
# Using k-fold encoding on training data
train_df['chip_model_encoded'] = target_encode_kfold(
    train_df, 'chip_model', 'price', n_fold=5, alpha=20
)

# Apply encoding to test data using the mapping learned from training data
chip_model_mapping = train_df.groupby('chip_model')['chip_model_encoded'].mean().to_dict()
test_df['chip_model_encoded'] = test_df['chip_model'].map(chip_model_mapping).fillna(train_df['price'].mean())

# Combine train and test data
df['chip_model_encoded'] = pd.concat([train_df['chip_model_encoded'], test_df['chip_model_encoded']])

# Target encoding for brand (medium cardinality)
print("Encoding brand...")
train_df['brand_encoded'] = target_encode_kfold(
    train_df, 'brand', 'price', n_fold=5, alpha=10
)
brand_mapping = train_df.groupby('brand')['brand_encoded'].mean().to_dict()
test_df['brand_encoded'] = test_df['brand'].map(brand_mapping).fillna(train_df['price'].mean())
df['brand_encoded'] = pd.concat([train_df['brand_encoded'], test_df['brand_encoded']])

# Target encoding for screen_tech (high cardinality)
print("Encoding screen_tech...")
train_df['screen_tech_encoded'] = target_encode_kfold(
    train_df, 'screen_tech', 'price', n_fold=5, alpha=15
)
screen_tech_mapping = train_df.groupby('screen_tech')['screen_tech_encoded'].mean().to_dict()
test_df['screen_tech_encoded'] = test_df['screen_tech'].map(screen_tech_mapping).fillna(train_df['price'].mean())
df['screen_tech_encoded'] = pd.concat([train_df['screen_tech_encoded'], test_df['screen_tech_encoded']])

# Target encoding for screen_resolution_k (low cardinality)
print("Encoding screen_resolution_k...")
train_df['screen_resolution_k_encoded'] = target_encode_kfold(
    train_df, 'screen_resolution_k', 'price', n_fold=5, alpha=5
)
resolution_mapping = train_df.groupby('screen_resolution_k')['screen_resolution_k_encoded'].mean().to_dict()
test_df['screen_resolution_k_encoded'] = test_df['screen_resolution_k'].map(resolution_mapping).fillna(train_df['price'].mean())
df['screen_resolution_k_encoded'] = pd.concat([train_df['screen_resolution_k_encoded'], test_df['screen_resolution_k_encoded']])

# Create visualizations for each encoded feature
print("\nCreating visualizations to show encoding impact...")
for col in ['brand', 'chip_model', 'screen_tech', 'screen_resolution_k']:
    encoded_col = f"{col}_encoded"
    
    # Create a subset for visualization (up to 10 categories)
    top_categories = df[col].value_counts().nlargest(10).index.tolist()
    df_viz = df[df[col].isin(top_categories)].copy()
    
    # Plot the relationship
    fig = plot_encoding_impact(df_viz, col, encoded_col, 'price', n_samples=min(100, len(df_viz)))
    
    # Save the figure
    fig_path = os.path.join(viz_dir, f"{col}_encoding_impact.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved visualization to {fig_path}")

# Calculate correlation with price
print("\nCorrelation with target (price):")
encoded_cols = ['chip_model_encoded', 'brand_encoded', 'screen_tech_encoded', 'screen_resolution_k_encoded']
correlations = df[encoded_cols + ['price']].corr()['price'].drop('price').sort_values(ascending=False)
print(correlations)

# Save the processed dataset
print(f"\nSaving encoded dataset to {output_file}...")
df.to_csv(output_file, index=False)

print("\nDone!")

# Summary of encoding approach for each feature
print("\nEncoding Summary:")
print("1. chip_model (205 distinct values): Used k-fold target encoding with higher regularization (alpha=20)")
print("2. brand (21 distinct values): Used k-fold target encoding with medium regularization (alpha=10)")
print("3. screen_tech (99 distinct values): Used k-fold target encoding with medium-high regularization (alpha=15)")
print("4. screen_resolution_k (4 distinct values): Used k-fold target encoding with low regularization (alpha=5)")
print("\nThe encoded features preserve the relationship with price while converting categorical variables to numerical values.")