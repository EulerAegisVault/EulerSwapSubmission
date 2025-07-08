#!/usr/bin/env python3
"""
Create Unichain ML Risk Model
Fixed version that creates the model in the correct location
"""

import requests
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def create_mock_features():
    """Create realistic mock features for DeFi protocols"""
    return {
        'avg_value': np.random.uniform(0.001, 0.1),
        'std_value': np.random.uniform(0.00001, 0.01),
        'max_value': np.random.uniform(0.1, 1.0),
        'avg_gas_used': np.random.uniform(50000, 200000),
        'std_gas_used': np.random.uniform(5000, 50000),
        'tx_count': np.random.randint(50, 1000),
        'unique_users': np.random.randint(10, 500),
        'avg_time_between_tx': np.random.uniform(1, 100),
        'max_time_between_tx': np.random.uniform(100, 10000),
        'failed_tx_ratio': np.random.uniform(0, 0.05),
        'high_value_tx_ratio': np.random.uniform(0, 0.3),
        'repeat_user_ratio': np.random.uniform(0.1, 0.8),
        'weekend_activity_ratio': np.random.uniform(0.1, 0.4),
        'night_activity_ratio': np.random.uniform(0.05, 0.3),
        'avg_tx_per_user': np.random.uniform(1, 10),
        'max_tx_per_user': np.random.uniform(5, 50),
        'gini_coefficient': np.random.uniform(0.2, 0.8),
        'activity_burst_score': np.random.uniform(0.5, 2.0)
    }

def main():
    print("🔬 Creating Unichain ML Risk Model")
    print("=" * 40)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    model_path = os.path.join(models_dir, "anomaly_risk_model.joblib")
    
    try:
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        print(f"📁 Models directory: {models_dir}")
        
        # Create mock baseline data (10 protocols for better training)
        print("📊 Generating training data...")
        baseline_data = [create_mock_features() for _ in range(10)]
        df = pd.DataFrame(baseline_data)
        
        print(f"✅ Created training data: {len(df)} protocols, {len(df.columns)} features")
        
        # Train model
        print("🤖 Training anomaly detection model...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        
        model = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        
        # Get baseline scores for reference
        baseline_scores = model.decision_function(X_scaled).tolist()
        print(f"📈 Baseline score range: {min(baseline_scores):.3f} to {max(baseline_scores):.3f}")
        
        # Save with correct format
        model_data = {
            'model': model,
            'scaler': scaler,
            'baseline_scores': baseline_scores,
            'feature_names': df.columns.tolist()
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Verify model works
        print("🧪 Testing model...")
        test_features = create_mock_features()
        test_df = pd.DataFrame([test_features])
        test_scaled = scaler.transform(test_df)
        test_score = model.decision_function(test_scaled)[0]
        
        print(f"✅ Test score: {test_score:.3f}")
        print("🎉 Unichain ML Risk Model created successfully!")
        
        return baseline_scores
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()