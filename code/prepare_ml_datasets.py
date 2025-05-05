import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Path to the input data files
category_file = '../data/datasets/category_label_dataset.csv'
price_file = '../data/datasets/price_label_dataset.csv'

# Path to output directory
output_dir = '../data/ml_ready/'
os.makedirs(output_dir, exist_ok=True)

def prepare_dataset(input_file, label_column, test_size=0.2, random_state=42):
    """
    Prepare a dataset for machine learning by splitting it into training and testing sets
    """
    print(f"Processing {input_file}...")
    
    # Read the dataset
    df = pd.read_csv(input_file)
    print(f"Dataset shape: {df.shape}")
    
    # Extract features and labels
    X = df.drop(label_column, axis=1)
    y = df[label_column]
    
    # Convert categorical columns to numeric if needed
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        X[col] = pd.factorize(X[col])[0]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

# Process the category label dataset
print("\nPreparing price category dataset...")
X_train_cat, X_test_cat, y_train_cat, y_test_cat = prepare_dataset(
    category_file, 'price_category'
)

# Process the price label dataset
print("\nPreparing price dataset...")
X_train_price, X_test_price, y_train_price, y_test_price = prepare_dataset(
    price_file, 'price'
)

# Save the processed datasets
print("\nSaving processed datasets...")

# Save category datasets
X_train_cat.to_csv(os.path.join(output_dir, 'X_train_category.csv'), index=False)
X_test_cat.to_csv(os.path.join(output_dir, 'X_test_category.csv'), index=False)
y_train_cat.to_csv(os.path.join(output_dir, 'y_train_category.csv'), index=False)
y_test_cat.to_csv(os.path.join(output_dir, 'y_test_category.csv'), index=False)

# Save price datasets
X_train_price.to_csv(os.path.join(output_dir, 'X_train_price.csv'), index=False)
X_test_price.to_csv(os.path.join(output_dir, 'X_test_price.csv'), index=False)
y_train_price.to_csv(os.path.join(output_dir, 'y_train_price.csv'), index=False)
y_test_price.to_csv(os.path.join(output_dir, 'y_test_price.csv'), index=False)

print("All datasets have been successfully processed and saved!")