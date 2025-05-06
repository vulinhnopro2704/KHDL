import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# Function to perform target encoding with regularization
def target_encode_with_regularization(train_df, test_df, column, target_col, alpha=5):
    """
    Performs target encoding with regularization based on category frequency.
    
    Parameters:
    -----------
    train_df : pandas DataFrame
        Training data
    test_df : pandas DataFrame
        Test data
    column : str
        The name of the categorical column to encode
    target_col : str
        The name of the target column
    alpha : float, default=5
        Regularization parameter - higher means more regularization
        
    Returns:
    --------
    train_encoded, test_encoded : tuple of pandas Series
        The encoded values for the column in train and test sets
    encoding_map : dict
        The mapping dictionary for future encoding
    """
    # Calculate the global mean from training data
    global_mean = train_df[target_col].mean()
    
    # Calculate category-level statistics from training data
    stats = train_df.groupby(column)[target_col].agg(['mean', 'count']).reset_index()
    
    # Apply regularization formula:
    # encoded_value = (count * category_mean + alpha * global_mean) / (count + alpha)
    stats['encoded'] = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)
    
    # Create a mapping dictionary
    encoding_map = dict(zip(stats[column], stats['encoded']))
    
    # Apply the mapping to both train and test data
    train_encoded = train_df[column].map(encoding_map).fillna(global_mean)
    test_encoded = test_df[column].map(encoding_map).fillna(global_mean)
    
    return train_encoded, test_encoded, encoding_map

def evaluate_models(X_train, y_train, kfold=5):
    """
    Evaluate multiple models using k-fold cross-validation
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Features for training
    y_train : pandas Series
        Target variable
    kfold : int, default=5
        Number of folds for cross-validation
        
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics for each model
    """
    # Models to evaluate
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'SGDRegressor': SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    }
    
    # Initialize results dictionary
    results = {
        model_name: {'rmse': [], 'mae': [], 'r2': []} 
        for model_name in models.keys()
    }
    
    # Create k-fold cross-validation
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    
    fold = 1
    # Perform k-fold validation
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Scale the features
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)
        
        print(f"\nFold {fold}/{kfold}")
        fold += 1
        
        # Evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(X_train_fold_scaled, y_train_fold)
            
            # Make predictions
            y_pred = model.predict(X_val_fold_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            mae = mean_absolute_error(y_val_fold, y_pred)
            r2 = r2_score(y_val_fold, y_pred)
            
            # Store results
            results[name]['rmse'].append(rmse)
            results[name]['mae'].append(mae)
            results[name]['r2'].append(r2)
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    return results

def main():
    # Path to the dataset
    data_path = '../../data/datasets/train_stratified_by_chip_model.csv'
    
    # Create directory for saving models if it doesn't exist
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Create directory for visualizations if it doesn't exist
    viz_dir = '../visualizations/model_selection'
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Display basic information
    print("\nFeatures available:")
    print(df.columns.tolist())
    
    print("\nDescriptive statistics:")
    print(df.describe().T)
    
    # Define target variable
    target_column = 'price'
    try:
        # Split the data into features and target
        y = df[target_column]
        X = df.drop(target_column, axis=1)
    except KeyError:
        print(f"Target column '{target_column}' not found in the dataset.")
        return
        
    
    # Identify categorical features
    categorical_cols = ['chip_model', 'brand', 'screen_tech', 'screen_resolution_k']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Create KFold for model evaluation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to store encoders and evaluation metrics
    encoders = {}
    all_metrics = []
    
    fold = 1
    # Perform encoding and model evaluation for each fold
    for train_index, test_index in kfold.split(X):
        print(f"\nProcessing Fold {fold}/5")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Store original data for fold
        fold_data = {
            'X_train': X_train.copy(),
            'X_test': X_test.copy(),
            'y_train': y_train.copy(),
            'y_test': y_test.copy()
        }
        
        # Create a new dataframe for encoded features
        X_train_encoded = X_train[numerical_cols].copy()
        X_test_encoded = X_test[numerical_cols].copy()
        
        # Perform target encoding for each categorical feature
        fold_encoders = {}
        for col in categorical_cols:
            print(f"Encoding {col}...")
            alpha = 20 if col == 'chip_model' else 10 if col == 'brand' else 15 if col == 'screen_tech' else 5
            train_encoded, test_encoded, encoding_map = target_encode_with_regularization(
                X_train, X_test, col, target_column, alpha=alpha
            )
            
            # Add encoded features to the dataframe
            encoded_col_name = f"{col}_encoded"
            X_train_encoded[encoded_col_name] = train_encoded
            X_test_encoded[encoded_col_name] = test_encoded
            
            # Store the encoder
            fold_encoders[col] = {
                'encoding_map': encoding_map,
                'global_mean': y_train.mean(),
                'alpha': alpha
            }
        
        # Store the encoders for this fold
        encoders[f"fold_{fold}"] = fold_encoders
        
        # Evaluate models using the encoded features
        print("\nEvaluating models with cross-validation on this fold's training data...")
        fold_results = evaluate_models(X_train_encoded, y_train)
        
        # Scale features for final evaluation on test set
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)
        
        # Evaluate each model on the test set
        for model_name in fold_results.keys():
            if model_name == 'LinearRegression':
                model = LinearRegression()
            elif model_name == 'Ridge':
                model = Ridge(alpha=1.0)
            elif model_name == 'Lasso':
                model = Lasso(alpha=0.1)
            else:  # SGDRegressor
                model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
            
            # Train on the entire training set
            model.fit(X_train_scaled, y_train)
            
            # Predict on test set
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store metrics
            all_metrics.append({
                'fold': fold,
                'model': model_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
            
            # Save the model and its associated components
            model_components = {
                'model': model,
                'encoders': fold_encoders,
                'scaler': scaler,
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols,
                'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2}
            }
            
            # Save the model components
            model_filename = os.path.join(models_dir, f"{model_name}_fold_{fold}.pkl")
            with open(model_filename, 'wb') as f:
                pickle.dump(model_components, f)
            
            print(f"{model_name} - Test RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        fold += 1
    
    # Convert all metrics to a DataFrame for analysis
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate average metrics for each model
    avg_metrics = metrics_df.groupby('model').mean().reset_index()
    print("\nAverage metrics across all folds:")
    print(avg_metrics)
    
    # Find the best model based on RMSE (lower is better)
    best_model_name = avg_metrics.loc[avg_metrics['rmse'].idxmin(), 'model']
    best_rmse = avg_metrics.loc[avg_metrics['rmse'].idxmin(), 'rmse']
    best_mae = avg_metrics.loc[avg_metrics['rmse'].idxmin(), 'mae']
    best_r2 = avg_metrics.loc[avg_metrics['rmse'].idxmin(), 'r2']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Average RMSE: {best_rmse:.2f}")
    print(f"Average MAE: {best_mae:.2f}")
    print(f"Average R²: {best_r2:.4f}")
    
    # Train the best model on the entire dataset
    print("\nTraining the best model on the entire dataset...")
    
    # Encode the entire dataset
    X_encoded = X[numerical_cols].copy()
    
    # Create a final encoder for each categorical feature
    final_encoders = {}
    for col in categorical_cols:
        print(f"Encoding {col}...")
        alpha = 20 if col == 'chip_model' else 10 if col == 'brand' else 15 if col == 'screen_tech' else 5
        train_encoded, _, encoding_map = target_encode_with_regularization(
            X, X, col, target_column, alpha=alpha
        )
        
        # Add encoded features to the dataframe
        encoded_col_name = f"{col}_encoded"
        X_encoded[encoded_col_name] = train_encoded
        
        # Store the encoder
        final_encoders[col] = {
            'encoding_map': encoding_map,
            'global_mean': y.mean(),
            'alpha': alpha
        }
    
    # Scale features
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_encoded)
    
    # Initialize and train the best model
    if best_model_name == 'LinearRegression':
        best_model = LinearRegression()
    elif best_model_name == 'Ridge':
        best_model = Ridge(alpha=1.0)
    elif best_model_name == 'Lasso':
        best_model = Lasso(alpha=0.1)
    else:  # SGDRegressor
        best_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    
    best_model.fit(X_scaled, y)
    
    # Save the final model and associated components
    final_model_components = {
        'model': best_model,
        'encoders': final_encoders,
        'scaler': final_scaler,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'model_name': best_model_name
    }
    
    final_model_filename = os.path.join(models_dir, 'best_model.pkl')
    with open(final_model_filename, 'wb') as f:
        pickle.dump(final_model_components, f)
    
    print(f"\nFinal {best_model_name} model saved to {final_model_filename}")
    
    # Create visualization of model performance
    plt.figure(figsize=(12, 8))
    
    # Box plot of RMSE for each model
    plt.subplot(2, 2, 1)
    sns.boxplot(x='model', y='rmse', data=metrics_df)
    plt.title('RMSE by Model')
    plt.xticks(rotation=45)
    
    # Box plot of MAE for each model
    plt.subplot(2, 2, 2)
    sns.boxplot(x='model', y='mae', data=metrics_df)
    plt.title('MAE by Model')
    plt.xticks(rotation=45)
    
    # Box plot of R² for each model
    plt.subplot(2, 2, 3)
    sns.boxplot(x='model', y='r2', data=metrics_df)
    plt.title('R² by Model')
    plt.xticks(rotation=45)
    
    # Bar chart of average metrics
    plt.subplot(2, 2, 4)
    avg_metrics_melted = pd.melt(avg_metrics, id_vars=['model'], value_vars=['rmse', 'mae', 'r2'])
    sns.barplot(x='model', y='value', hue='variable', data=avg_metrics_melted)
    plt.title('Average Metrics by Model')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'model_comparison.png'))
    
    print(f"Model comparison visualization saved to {os.path.join(viz_dir, 'model_comparison.png')}")
    
    # Create a visualization of feature importances for linear models
    if best_model_name in ['LinearRegression', 'Ridge', 'Lasso']:
        feature_names = numerical_cols + [f"{col}_encoded" for col in categorical_cols]
        
        # Get feature coefficients
        coefficients = best_model.coef_
        
        # Create a DataFrame for visualization
        coeffs_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })
        
        # Sort by absolute coefficient value
        coeffs_df['AbsCoefficient'] = np.abs(coeffs_df['Coefficient'])
        coeffs_df = coeffs_df.sort_values('AbsCoefficient', ascending=False)
        
        # Visualize the top 10 coefficients
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=coeffs_df.head(10))
        plt.title(f'Top 10 Feature Importances - {best_model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importances.png'))
        
        print(f"Feature importance visualization saved to {os.path.join(viz_dir, 'feature_importances.png')}")

if __name__ == "__main__":
    main()