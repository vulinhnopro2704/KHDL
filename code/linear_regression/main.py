import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_model():
    """Load the best trained model"""
    model_path = '../models/best_model.pkl'
    try:
        with open(model_path, 'rb') as f:
            model_components = pickle.load(f)
        print(f"Successfully loaded model: {model_components['model_name']}")
        return model_components
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Please run model_selection.py first.")
        return None

def predict_price(model_components, sample_data):
    """Make predictions using the trained model"""
    if model_components is None:
        return None
    
    # Extract components
    model = model_components['model']
    encoders = model_components['encoders']
    scaler = model_components['scaler']
    categorical_cols = model_components['categorical_cols']
    numerical_cols = model_components['numerical_cols']
    
    # Create a copy of input data
    data = sample_data.copy()
    
    # Prepare the features
    X = data.copy()
    if 'price' in X.columns:
        X = X.drop('price', axis=1)
    
    # Extract numerical features
    X_encoded = X[numerical_cols].copy()
    
    # Perform encoding for categorical features
    for col in categorical_cols:
        if col in X.columns:
            # Get the encoder for this column
            encoder = encoders[col]
            encoding_map = encoder['encoding_map']
            global_mean = encoder['global_mean']
            
            # Apply encoding
            encoded_col_name = f"{col}_encoded"
            X_encoded[encoded_col_name] = X[col].map(encoding_map).fillna(global_mean)
    
    # Scale features
    X_scaled = scaler.transform(X_encoded)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions

def evaluate_model_on_test_data():
    """Evaluate the model on the test dataset"""
    # Load the model components
    model_components = load_model()
    if model_components is None:
        return
    
    # Load test data
    test_file = '../data/datasets/test_stratified_by_chip_model.csv'
    if not os.path.exists(test_file):
        print(f"Test file not found at {test_file}")
        return
    
    test_df = pd.read_csv(test_file)
    print(f"Loaded test dataset with {test_df.shape[0]} samples")
    
    # Make predictions
    if 'price' in test_df.columns:
        y_true = test_df['price']
        y_pred = predict_price(model_components, test_df)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print("\nModel Performance on Test Data:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        
        # Visualize actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs. Predicted Prices')
        
        viz_dir = '../visualizations/model_selection'
        os.makedirs(viz_dir, exist_ok=True)
        plt.savefig(os.path.join(viz_dir, 'actual_vs_predicted.png'))
        print(f"Visualization saved to {os.path.join(viz_dir, 'actual_vs_predicted.png')}")
    else:
        print("No 'price' column in test data for evaluation")
        y_pred = predict_price(model_components, test_df)
        
    return y_pred

def print_model_info():
    """Print information about the trained model"""
    model_components = load_model()
    if model_components is None:
        return
    
    print("\nModel Information:")
    print(f"Model Type: {model_components['model_name']}")
    
    if model_components['model_name'] in ['LinearRegression', 'Ridge', 'Lasso']:
        model = model_components['model']
        feature_names = model_components['numerical_cols'] + [f"{col}_encoded" for col in model_components['categorical_cols']]
        
        # Get feature coefficients
        coefficients = model.coef_
        
        # Create a DataFrame for display
        coeffs_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })
        
        # Sort by absolute coefficient value
        coeffs_df['AbsCoefficient'] = np.abs(coeffs_df['Coefficient'])
        coeffs_df = coeffs_df.sort_values('AbsCoefficient', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(coeffs_df.head(10)[['Feature', 'Coefficient']])
    
    # Print encoding information
    print("\nEncoding Parameters:")
    for col, encoder in model_components['encoders'].items():
        print(f"{col}: alpha={encoder['alpha']}, global_mean={encoder['global_mean']:.2f}")

def main():
    """Main function to run the program"""
    print("\n==== Smartphone Price Prediction Model ====\n")
    
    # Check if model has been trained
    model_path = '../models/best_model.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Running model selection first...")
        import model_selection
        model_selection.main()
    
    # Print information about the model
    print_model_info()
    
    # Evaluate the model on test data
    print("\nEvaluating model on test data...")
    evaluate_model_on_test_data()
    
    print("\nDone!")

if __name__ == "__main__":
    main()