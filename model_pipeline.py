import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_samples=10000):
    """Generates synthetic ad impression data."""
    print(f"Generating {num_samples} synthetic ad impressions...")
    
    # Features
    ad_formats = ['Banner', 'Video', 'Native', 'Rich Media']
    page_positions = ['Top', 'Middle', 'Bottom', 'Sidebar']
    device_types = ['Desktop', 'Mobile', 'Tablet']
    
    data = {
        'ad_format': np.random.choice(ad_formats, num_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'page_position': np.random.choice(page_positions, num_samples, p=[0.3, 0.2, 0.3, 0.2]),
        'device_type': np.random.choice(device_types, num_samples, p=[0.5, 0.4, 0.1]),
        'clutter_score': np.random.uniform(0, 10, num_samples), # 0 = clean, 10 = very cluttered
        'time_in_view_sec': np.random.exponential(5, num_samples), # Most ads viewed for short time
        'scroll_velocity': np.random.normal(50, 20, num_samples), # pixels per second
        'domain_quality_score': np.random.uniform(0, 100, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Cap time in view and scroll velocity to realistic values
    df['time_in_view_sec'] = np.clip(df['time_in_view_sec'], 0, 60)
    df['scroll_velocity'] = np.clip(df['scroll_velocity'], 0, 150)
    
    # Generate Target: Attention Score (simulating AU metric)
    # Higher quality domain, less clutter, longer time in view -> higher attention
    # Video and Native tend to get more attention than banners
    
    format_multiplier = {'Banner': 0.8, 'Native': 1.2, 'Video': 1.5, 'Rich Media': 1.3}
    pos_multiplier = {'Top': 1.2, 'Middle': 1.0, 'Bottom': 0.6, 'Sidebar': 0.7}
    device_multiplier = {'Desktop': 1.1, 'Mobile': 0.9, 'Tablet': 1.0}
    
    df['attention_score_base'] = (
        (df['domain_quality_score'] * 0.4) + 
        ((10 - df['clutter_score']) * 3) + 
        (df['time_in_view_sec'] * 2) - 
        (df['scroll_velocity'] * 0.1)
    )
    
    df['attention_score'] = (
        df['attention_score_base'] * 
        df['ad_format'].map(format_multiplier) * 
        df['page_position'].map(pos_multiplier) *
        df['device_type'].map(device_multiplier)
    )
    
    # Add some noise
    df['attention_score'] += np.random.normal(0, 10, num_samples)
    
    # Normalize attention score to roughly 0-100 for readability
    df['attention_score'] = np.clip(df['attention_score'], 0, 100)
    
    # Generate Target: Conversion (Causal Impact simulation)
    # Higher attention significantly increases conversion probability
    base_conv_prob = 0.01 # 1% base
    attention_impact = df['attention_score'] / 100 * 0.05 # up to 5% bonus from attention
    
    # Confounders for Causal Inference: Domain Quality affects both Attention and Conversion natively
    domain_impact = df['domain_quality_score'] / 100 * 0.02
    
    conv_prob = base_conv_prob + attention_impact + domain_impact
    df['conversion'] = np.random.binomial(1, np.clip(conv_prob, 0, 1))
    
    # Drop intermediate columns
    df = df.drop(columns=['attention_score_base'])
    
    return df

def train_and_save_model(df, model_path='attention_model.pkl'):
    print("Training XGBoost Model...")
    
    # Categorical encoding
    df_encoded = pd.get_dummies(df, columns=['ad_format', 'page_position', 'device_type'])
    
    features = [col for col in df_encoded.columns if col not in ['attention_score', 'conversion']]
    X = df_encoded[features]
    y = df_encoded['attention_score']
    
    # Train Random Forest
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X, y)
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X, y)
    
    # Save models and feature names
    joblib.dump({
        'rf_model': rf_model, 
        'xgb_model': xgb_model, 
        'features': features
    }, model_path)
    print(f"Models saved to {model_path}")
    
    return {'rf_model': rf_model, 'xgb_model': xgb_model}, features

if __name__ == "__main__":
    df = generate_synthetic_data(20000)
    df.to_csv('synthetic_ad_data.csv', index=False)
    print("Saved synthetic data to synthetic_ad_data.csv")
    
    train_and_save_model(df)
    print("Pipeline complete.")
