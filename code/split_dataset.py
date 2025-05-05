import pandas as pd
import os

# Path to the input data file
input_file = '../data/clean/feature_engineering.csv'

# Path to output files
output_dir = '../data/datasets/'
os.makedirs(output_dir, exist_ok=True)

# Read the dataset
print(f"Reading dataset from {input_file}...")
df = pd.read_csv(input_file)
print(f"Dataset shape: {df.shape}")
print(f"Dataset columns: {df.columns.tolist()}")

# Create dataset with price_category as label
print("Creating dataset with price_category as label...")
df_category = df.copy()
# Move price_category to the first column to make it clear it's the label
columns = ['price_category'] + [col for col in df_category.columns if col != 'price_category']
df_category = df_category[columns]

# Create dataset with price as label
print("Creating dataset with price as label...")
df_price = df.copy()
# Move price to the first column to make it clear it's the label
columns = ['price'] + [col for col in df_price.columns if col != 'price']
df_price = df_price[columns]

# Save the datasets
category_output = os.path.join(output_dir, 'category_label_dataset.csv')
price_output = os.path.join(output_dir, 'price_label_dataset.csv')

df_category.to_csv(category_output, index=False)
df_price.to_csv(price_output, index=False)

print(f"Saved dataset with price_category as label to {category_output}")
print(f"Saved dataset with price as label to {price_output}")
print("Done!")