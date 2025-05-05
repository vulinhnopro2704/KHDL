import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from matplotlib.gridspec import GridSpec

# Set the style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

def load_data():
    """Load the dataset from the specified path."""
    data_path = os.path.join('..', 'data', 'clean', 'feature_engineering.csv')
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        # Try alternative path
        data_path = os.path.join('..', 'data', 'datasets', 'price_label_dataset.csv')
        try:
            df = pd.read_csv(data_path)
            print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print("Error: Dataset not found. Please check the file path.")
            return None

def explore_categorical_features(df):
    """Identify categorical features and their value counts."""
    categorical_features = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            print(f"Feature: {col}, Unique values: {df[col].nunique()}")
            categorical_features.append(col)
    
    return categorical_features

def stratified_split_comparison(df, stratify_columns):
    """
    Compare the distributions of features when stratified by different columns
    
    Args:
        df: DataFrame containing the data
        stratify_columns: List of columns to use for stratification
    
    Returns:
        Dictionary with train and test datasets for each stratification approach
    """
    results = {}
    
    # Create a directory for saving visualizations if it doesn't exist
    vis_dir = os.path.join('..', 'visualizations', 'train_test_split')
    os.makedirs(vis_dir, exist_ok=True)
    
    for stratify_col in stratify_columns:
        print(f"\nStratifying by {stratify_col}:")
        
        # Handle missing values in stratification column
        if df[stratify_col].isnull().any():
            print(f"Warning: NaN values found in {stratify_col}. Filling with 'Unknown'")
            df[stratify_col] = df[stratify_col].fillna('Unknown')
        
        # For numerical columns with many unique values, bin them for stratification
        if df[stratify_col].dtype != 'object' and df[stratify_col].nunique() > 20:
            print(f"Binning {stratify_col} for stratification as it has {df[stratify_col].nunique()} unique values")
            df[f'{stratify_col}_bin'] = pd.qcut(df[stratify_col], 10, duplicates='drop')
            stratify_values = df[f'{stratify_col}_bin']
        else:
            stratify_values = df[stratify_col]
        
        # Handle rare classes (with only 1 member)
        # First identify classes with only 1 member
        value_counts = stratify_values.value_counts()
        rare_classes = value_counts[value_counts == 1].index.tolist()
        
        if rare_classes:
            print(f"Found {len(rare_classes)} rare classes with only 1 member in {stratify_col}")
            
            # Create mask for samples to be always included in training set
            rare_mask = stratify_values.isin(rare_classes)
            train_forced = df[rare_mask].copy()
            
            # Remove these samples from the stratification process
            df_stratify = df[~rare_mask].copy()
            stratify_values_filtered = stratify_values[~rare_mask]
            
            # Perform stratified split on remaining data
            if len(df_stratify) > 0:
                train, test = train_test_split(
                    df_stratify, 
                    test_size=0.2,
                    random_state=42,
                    stratify=stratify_values_filtered
                )
                
                # Add rare class samples to training set
                train = pd.concat([train, train_forced])
            else:
                # If all data is rare classes, put everything in train
                train = train_forced
                test = pd.DataFrame(columns=df.columns)
                
            print(f"Added {len(train_forced)} rare class samples directly to training set")
        else:
            # Standard stratified split if no rare classes
            try:
                train, test = train_test_split(
                    df, 
                    test_size=0.2,
                    random_state=42,
                    stratify=stratify_values
                )
            except ValueError as e:
                if "The least populated class in y has only 1 member" in str(e):
                    print(f"Error: {e}")
                    print("Retrying with a different approach for rare classes...")
                    
                    # Identify classes with fewer than 5 members
                    value_counts = stratify_values.value_counts()
                    rare_classes = value_counts[value_counts < 5].index.tolist()
                    
                    # Create mask for samples to be always included in training set
                    rare_mask = stratify_values.isin(rare_classes)
                    train_forced = df[rare_mask].copy()
                    
                    # Remove these samples from the stratification process
                    df_stratify = df[~rare_mask].copy()
                    stratify_values_filtered = stratify_values[~rare_mask]
                    
                    # Perform stratified split on remaining data
                    train, test = train_test_split(
                        df_stratify, 
                        test_size=0.2,
                        random_state=42,
                        stratify=stratify_values_filtered
                    )
                    
                    # Add rare class samples to training set
                    train = pd.concat([train, train_forced])
                    
                    print(f"Added {len(train_forced)} rare class samples directly to training set")
                else:
                    raise
        
        results[stratify_col] = {'train': train, 'test': test}
        
        # Calculate and print distribution comparison
        train_dist = train[stratify_col].value_counts(normalize=True)
        test_dist = test[stratify_col].value_counts(normalize=True)
        
        print(f"Training set size: {train.shape[0]}")
        print(f"Test set size: {test.shape[0]}")
        
        # Calculate KL divergence for numerical features to measure distribution similarity
        feature_balance_scores = {}
        
        for feature in df.select_dtypes(include=np.number).columns:
            if feature == stratify_col or len(test) == 0:
                continue
                
            # Create histograms with the same bins for train and test
            hist_bins = np.histogram_bin_edges(df[feature], bins='auto')
            train_hist, _ = np.histogram(train[feature], bins=hist_bins, density=True)
            test_hist, _ = np.histogram(test[feature], bins=hist_bins, density=True)
            
            # Avoid division by zero and log(0)
            train_hist = np.clip(train_hist, 1e-10, None)
            test_hist = np.clip(test_hist, 1e-10, None)
            
            # Calculate KL divergence: lower is better (more similar distributions)
            kl_div = np.sum(test_hist * np.log(test_hist / train_hist))
            feature_balance_scores[feature] = kl_div
        
        if feature_balance_scores:
            avg_balance_score = np.mean(list(feature_balance_scores.values()))
            print(f"Average KL divergence across numerical features: {avg_balance_score:.4f}")
        else:
            print("Warning: Could not calculate feature balance scores (empty test set or all features excluded)")
        
        # Visualize the distributions of train and test sets
        if len(test) > 0:  # Only visualize if we have test data
            visualize_distributions(train, test, stratify_col, vis_dir)
        
    return results

def visualize_distributions(train, test, stratify_col, save_dir):
    """
    Visualize the distributions of train and test sets for different features
    """
    # Select a subset of important features to visualize
    features_to_plot = ['price', 'ram', 'internal_memory', 'screen_size', 'brand', 'chip_model']
    features_to_plot = [f for f in features_to_plot if f in train.columns and f != stratify_col]
    
    # Add the stratification column to the list of features to plot
    if stratify_col not in features_to_plot:
        features_to_plot.insert(0, stratify_col)
    
    # Determine grid dimensions
    n_features = len(features_to_plot)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    fig = plt.figure(figsize=(15, n_rows * 5))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    for i, feature in enumerate(features_to_plot):
        row, col = i // n_cols, i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        if train[feature].dtype == 'object' or train[feature].nunique() < 20:
            # For categorical features, plot bar chart of value counts
            train_counts = train[feature].value_counts(normalize=True)
            test_counts = test[feature].value_counts(normalize=True)
            
            # Get the union of indices
            all_indices = list(set(train_counts.index) | set(test_counts.index))
            
            # Create a DataFrame with both distributions
            compare_df = pd.DataFrame(index=all_indices)
            compare_df['Train'] = train_counts
            compare_df['Test'] = test_counts
            compare_df.fillna(0, inplace=True)
            
            # Sort by frequency in train set
            compare_df = compare_df.sort_values('Train', ascending=False)
            
            # Plot top 10 categories if there are many
            if len(compare_df) > 10:
                compare_df = compare_df.head(10)
                
            compare_df.plot(kind='bar', ax=ax)
            ax.set_title(f'{feature} Distribution in Train vs Test')
            ax.set_ylabel('Proportion')
            ax.tick_params(axis='x', rotation=45)
            
        else:
            # For numerical features, plot histograms
            bins = np.histogram_bin_edges(pd.concat([train[feature], test[feature]]), bins='auto')
            
            ax.hist(train[feature], bins=bins, alpha=0.5, label='Train', density=True)
            ax.hist(test[feature], bins=bins, alpha=0.5, label='Test', density=True)
            
            ax.set_title(f'{feature} Distribution in Train vs Test')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'stratify_by_{stratify_col}_distributions.png'))
    plt.close()
    
    # Create a heatmap to visualize the correlation between numerical features
    plt.figure(figsize=(12, 10))
    sns.heatmap(train.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlations (Train Set - Stratified by {stratify_col})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'stratify_by_{stratify_col}_correlations.png'))
    plt.close()

def select_best_stratification(results):
    """
    Select the best stratification approach based on balance metrics
    """
    balance_scores = {}
    
    for strat_col, datasets in results.items():
        train = datasets['train']
        test = datasets['test']
        
        # Calculate balance scores across all numerical features
        feature_scores = {}
        for feature in train.select_dtypes(include=np.number).columns:
            if feature.endswith('_bin'):
                continue
                
            # Create histograms with same bins
            hist_bins = np.histogram_bin_edges(pd.concat([train[feature], test[feature]]), bins='auto')
            train_hist, _ = np.histogram(train[feature], bins=hist_bins, density=True)
            test_hist, _ = np.histogram(test[feature], bins=hist_bins, density=True)
            
            # Avoid division by zero and log(0)
            train_hist = np.clip(train_hist, 1e-10, None)
            test_hist = np.clip(test_hist, 1e-10, None)
            
            # KL divergence - lower means more similar distributions
            kl_div = np.sum(test_hist * np.log(test_hist / train_hist))
            feature_scores[feature] = kl_div
        
        # Average score across features
        balance_scores[strat_col] = np.mean(list(feature_scores.values()))
    
    best_col = min(balance_scores, key=balance_scores.get)
    
    print("\nBalance scores for each stratification approach (lower is better):")
    for col, score in balance_scores.items():
        print(f"{col}: {score:.4f}")
        
    print(f"\nBest stratification column: {best_col} with score: {balance_scores[best_col]:.4f}")
    
    return best_col, results[best_col]

def save_best_splits(best_split, best_col):
    """
    Save the best train-test split to CSV files
    """
    train = best_split['train']
    test = best_split['test']
    
    # Create directory if it doesn't exist
    out_dir = os.path.join('..', 'data', 'datasets')
    os.makedirs(out_dir, exist_ok=True)
    
    # Save files
    train_file = os.path.join(out_dir, f'train_stratified_by_{best_col}.csv')
    test_file = os.path.join(out_dir, f'test_stratified_by_{best_col}.csv')
    
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    
    print(f"\nSaved the best train-test split (stratified by {best_col}):")
    print(f"Train set: {train_file}")
    print(f"Test set: {test_file}")

def main():
    # Load the dataset
    df = load_data()
    if df is None:
        return
    
    print("\nExploring dataset characteristics...\n")
    print(f"Dataset overview:\n{df.info()}")
    print(f"\nSample of data:\n{df.head()}")
    
    # Identify categorical features for stratification
    print("\nIdentifying categorical features for stratification...")
    categorical_features = explore_categorical_features(df)
    
    # Add numerical features that might be good for stratification
    stratify_candidates = categorical_features.copy()
    for col in df.columns:
        if col not in stratify_candidates and df[col].nunique() < 50 and df[col].nunique() > 1:
            stratify_candidates.append(col)
    
    # If there are too many candidates, select the most promising ones
    if len(stratify_candidates) > 5:
        print("\nToo many stratification candidates, selecting the most promising ones...")
        # Prioritize features like brand, price range, chip_model, screen_tech
        priority_features = ['brand', 'chip_model', 'screen_tech', 'screen_resolution_k']
        selected_candidates = [col for col in stratify_candidates if col in priority_features]
        
        # Add a few more if needed
        if len(selected_candidates) < 3:
            additional_features = [col for col in stratify_candidates if col not in selected_candidates][:3-len(selected_candidates)]
            selected_candidates.extend(additional_features)
        
        stratify_candidates = selected_candidates
    
    print(f"\nSelected stratification candidates: {stratify_candidates}")
    
    # Compare different stratification approaches
    results = stratified_split_comparison(df, stratify_candidates)
    
    # Select the best stratification approach
    best_col, best_split = select_best_stratification(results)
    
    # Save the best train-test split
    save_best_splits(best_split, best_col)
    
    print("\nTrain-test split analysis complete!")

if __name__ == "__main__":
    main()