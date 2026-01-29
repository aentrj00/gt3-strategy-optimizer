"""
Safety Car Prediction Model
==========================
XGBoost model to predict Safety Car probability during races.

Features used:
- Circuit characteristics
- Race timing (minute, hour)
- Weather conditions
- Number of cars on track
- Historical SC patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
from pathlib import Path
import logging

# Optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyCarPredictor:
    """
    Machine learning model to predict Safety Car probability.
    
    Uses historical race data to predict the likelihood of a
    Safety Car or Full Course Yellow in upcoming laps.
    """
    
    # Circuit risk profiles (historical SC rates)
    CIRCUIT_RISK = {
        'le_mans': 0.12,
        'spa': 0.20,
        'monza': 0.10,
        'nurburgring': 0.18,
        'daytona': 0.25,
        'sebring': 0.22,
        'bathurst': 0.28,
        'laguna_seca': 0.20,
        'cota': 0.15,
        'fuji': 0.16,
        'bahrain': 0.10,
        'default': 0.15
    }
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to load pre-trained model
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features for the model.
        
        Args:
            df: Raw data DataFrame
            is_training: Whether this is for training (fit encoders)
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Time-based features
        if 'elapsed_time_seconds' in df.columns:
            features['race_minute'] = df['elapsed_time_seconds'] / 60
            features['race_hour'] = df['elapsed_time_seconds'] / 3600
        elif 'lap_number' in df.columns and 'lap_time_seconds' in df.columns:
            features['race_minute'] = (df['lap_number'] * df['lap_time_seconds'].mean()) / 60
            features['race_hour'] = features['race_minute'] / 60
        
        # Lap features
        if 'lap_number' in df.columns:
            features['lap_number'] = df['lap_number']
            features['laps_remaining'] = df.get('total_laps', 100) - df['lap_number']
            features['race_progress'] = df['lap_number'] / df.get('total_laps', 100)
        
        # Circuit encoding
        if 'circuit' in df.columns:
            if is_training:
                self.label_encoders['circuit'] = LabelEncoder()
                features['circuit_encoded'] = self.label_encoders['circuit'].fit_transform(df['circuit'])
            else:
                features['circuit_encoded'] = self.label_encoders['circuit'].transform(df['circuit'])
            
            # Add circuit risk factor
            features['circuit_risk'] = df['circuit'].map(
                lambda x: self.CIRCUIT_RISK.get(x, self.CIRCUIT_RISK['default'])
            )
        
        # Weather encoding
        if 'weather' in df.columns:
            if is_training:
                self.label_encoders['weather'] = LabelEncoder()
                features['weather_encoded'] = self.label_encoders['weather'].fit_transform(
                    df['weather'].fillna('dry')
                )
            else:
                features['weather_encoded'] = self.label_encoders['weather'].transform(
                    df['weather'].fillna('dry')
                )
            
            # Weather risk factor
            weather_risk = {'dry': 1.0, 'damp': 1.5, 'wet': 2.0, 'heavy_rain': 2.5}
            features['weather_risk'] = df['weather'].map(
                lambda x: weather_risk.get(x, 1.0)
            )
        
        # Car count on track (if available)
        if 'cars_on_track' in df.columns:
            features['cars_on_track'] = df['cars_on_track']
        elif 'car_number' in df.columns:
            # Estimate from data
            features['cars_on_track'] = df.groupby('lap_number')['car_number'].transform('nunique')
        
        # Class distribution (multi-class affects SC probability)
        if 'class' in df.columns:
            class_counts = df.groupby('lap_number')['class'].value_counts().unstack(fill_value=0)
            if 'Hypercar' in class_counts.columns:
                features['hypercars_on_track'] = df['lap_number'].map(
                    class_counts.get('Hypercar', pd.Series())
                ).fillna(0)
            features['class_variety'] = df.groupby('lap_number')['class'].transform('nunique')
        
        # Historical pattern features
        if 'track_status' in df.columns:
            # Previous SC in this race
            df['is_sc'] = df['track_status'].isin(['SC', 'FCY', 'safety_car']).astype(int)
            features['previous_sc_count'] = df.groupby('race_id' if 'race_id' in df.columns else 'lap_number')['is_sc'].transform('cumsum')
            features['laps_since_last_sc'] = df.groupby('race_id' if 'race_id' in df.columns else 'lap_number').apply(
                lambda x: (x['is_sc'] == 0).cumsum()
            ).reset_index(drop=True)
        
        # Track temperature (affects grip -> incidents)
        if 'track_temp_c' in df.columns:
            features['track_temp'] = df['track_temp_c']
            features['temp_extreme'] = ((df['track_temp_c'] < 20) | (df['track_temp_c'] > 40)).astype(int)
        
        # Tire condition aggregate (low tire life = more incidents)
        if 'tire_life_pct' in df.columns:
            features['avg_tire_life'] = df.groupby('lap_number')['tire_life_pct'].transform('mean') / 100
            features['min_tire_life'] = df.groupby('lap_number')['tire_life_pct'].transform('min') / 100
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Store feature columns
        if is_training:
            self.feature_columns = features.columns.tolist()
        else:
            # Ensure same columns as training
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.feature_columns]
        
        return features
    
    def prepare_target(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Prepare target variable: SC in next N laps.
        
        Args:
            df: Data with track_status column
            window: Number of laps to look ahead
            
        Returns:
            Binary series (1 = SC in next N laps)
        """
        if 'track_status' not in df.columns:
            logger.warning("No track_status column - generating synthetic target")
            # Generate synthetic target based on circuit risk
            if 'circuit' in df.columns:
                base_prob = df['circuit'].map(
                    lambda x: self.CIRCUIT_RISK.get(x, 0.15) / 10  # Per-lap probability
                )
            else:
                base_prob = 0.015  # Default
            return (np.random.random(len(df)) < base_prob).astype(int)
        
        # Real target: SC occurs in next N laps
        df['is_sc'] = df['track_status'].isin(['SC', 'FCY', 'safety_car']).astype(int)
        
        # Look ahead
        target = df.groupby('race_id' if 'race_id' in df.columns else 'car_number').apply(
            lambda x: x['is_sc'].rolling(window=window, min_periods=1).max().shift(-window)
        ).reset_index(drop=True)
        
        return target.fillna(0).astype(int)
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        use_time_split: bool = True
    ) -> Dict:
        """
        Train the Safety Car prediction model.
        
        Args:
            df: Training data DataFrame
            test_size: Fraction for test set
            use_time_split: Use time-based split (recommended for race data)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Preparing features for training...")
        
        # Prepare features and target
        X = self.prepare_features(df, is_training=True)
        y = self.prepare_target(df)
        
        # Align data
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Training data: {len(X)} samples, {len(self.feature_columns)} features")
        logger.info(f"Positive class ratio: {y.mean():.2%}")
        
        # Split data
        if use_time_split:
            tscv = TimeSeriesSplit(n_splits=5)
            train_idx, test_idx = list(tscv.split(X))[-1]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if XGB_AVAILABLE:
            logger.info("Training XGBoost model...")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train) / (2 * y_train.sum() + 1),  # Handle imbalance
                random_state=42,
                use_label_encoder=False,
                eval_metric='auc'
            )
        elif LGB_AVAILABLE:
            logger.info("Training LightGBM model...")
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            logger.warning("Neither XGBoost nor LightGBM available. Using sklearn.")
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else 0,
            'samples': len(X),
            'positive_rate': y.mean(),
            'test_positive_rate': y_test.mean()
        }
        
        logger.info(f"Model trained. Accuracy: {metrics['accuracy']:.2%}, AUC: {metrics['roc_auc']:.3f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            metrics['feature_importance'] = importance
            logger.info(f"\nTop features:\n{importance.head(5).to_string(index=False)}")
        
        return metrics
    
    def predict_probability(
        self,
        circuit: str,
        race_minute: float,
        weather: str = 'dry',
        cars_on_track: int = 30,
        track_temp: float = 25.0,
        previous_sc_count: int = 0
    ) -> Dict:
        """
        Predict SC probability for current race conditions.
        
        Args:
            circuit: Circuit identifier
            race_minute: Current race minute
            weather: Weather condition
            cars_on_track: Number of cars still racing
            track_temp: Track temperature (°C)
            previous_sc_count: SCs already in this race
            
        Returns:
            Dictionary with probabilities
        """
        if not self.is_trained:
            # Use heuristic model if not trained
            return self._heuristic_prediction(
                circuit, race_minute, weather, cars_on_track, track_temp
            )
        
        # Build feature vector
        features = pd.DataFrame([{
            'race_minute': race_minute,
            'race_hour': race_minute / 60,
            'lap_number': int(race_minute * 60 / 100),  # Estimate
            'circuit': circuit,
            'weather': weather,
            'cars_on_track': cars_on_track,
            'track_temp_c': track_temp,
            'previous_sc_count': previous_sc_count
        }])
        
        X = self.prepare_features(features, is_training=False)
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0, 1]
        
        return {
            'probability': prob,
            'risk_level': self._risk_level(prob),
            'recommendation': self._recommendation(prob)
        }
    
    def _heuristic_prediction(
        self,
        circuit: str,
        race_minute: float,
        weather: str,
        cars_on_track: int,
        track_temp: float
    ) -> Dict:
        """Fallback heuristic prediction when model not trained."""
        # Base probability from circuit
        base_prob = self.CIRCUIT_RISK.get(circuit, self.CIRCUIT_RISK['default'])
        
        # Adjust for time (SC more likely in first hour)
        if race_minute < 60:
            time_factor = 1.5
        elif race_minute < 120:
            time_factor = 1.0
        else:
            time_factor = 0.8
        
        # Weather factor
        weather_factor = {'dry': 1.0, 'damp': 1.5, 'wet': 2.0, 'heavy_rain': 2.5}.get(weather, 1.0)
        
        # Car count factor
        car_factor = 1.0 + (cars_on_track - 25) * 0.01
        
        # Temperature factor (extremes increase risk)
        temp_factor = 1.0
        if track_temp < 20 or track_temp > 40:
            temp_factor = 1.2
        
        # Calculate probability per 10-minute window
        prob = base_prob * time_factor * weather_factor * car_factor * temp_factor / 6
        prob = min(prob, 0.5)  # Cap at 50%
        
        return {
            'probability': prob,
            'risk_level': self._risk_level(prob),
            'recommendation': self._recommendation(prob),
            'model': 'heuristic'
        }
    
    def _risk_level(self, prob: float) -> str:
        """Convert probability to risk level."""
        if prob < 0.1:
            return 'Low'
        elif prob < 0.25:
            return 'Medium'
        elif prob < 0.4:
            return 'High'
        else:
            return 'Very High'
    
    def _recommendation(self, prob: float) -> str:
        """Generate strategy recommendation based on probability."""
        if prob < 0.1:
            return "Proceed with normal strategy"
        elif prob < 0.25:
            return "Consider extending stint if fuel allows"
        elif prob < 0.4:
            return "Hold pit stop if possible - SC likely soon"
        else:
            return "High SC probability - pit under SC if it occurs"
    
    def predict_race_windows(
        self,
        circuit: str,
        race_duration_minutes: int,
        weather: str = 'dry',
        interval_minutes: int = 10
    ) -> pd.DataFrame:
        """
        Predict SC probability throughout the race.
        
        Args:
            circuit: Circuit identifier
            race_duration_minutes: Total race duration
            weather: Weather condition
            interval_minutes: Prediction interval
            
        Returns:
            DataFrame with time-based predictions
        """
        predictions = []
        
        for minute in range(0, race_duration_minutes, interval_minutes):
            pred = self.predict_probability(
                circuit=circuit,
                race_minute=minute,
                weather=weather
            )
            
            predictions.append({
                'minute': minute,
                'hour': minute / 60,
                'probability': pred['probability'],
                'risk_level': pred['risk_level']
            })
        
        return pd.DataFrame(predictions)
    
    def save_model(self, path: str) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            logger.warning("Model not trained - nothing to save")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model from file."""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        
        logger.info(f"Model loaded from {path}")


def main():
    """Demo safety car predictor."""
    print("Safety Car Predictor Demo")
    print("=" * 50)
    
    predictor = SafetyCarPredictor()
    
    # Test heuristic prediction (no training data needed)
    print("\nHeuristic Predictions:")
    print("-" * 40)
    
    circuits = ['spa', 'daytona', 'le_mans', 'bahrain']
    
    for circuit in circuits:
        pred = predictor.predict_probability(
            circuit=circuit,
            race_minute=120,  # 2 hours in
            weather='dry',
            cars_on_track=28
        )
        print(f"{circuit.upper():15} | Prob: {pred['probability']:.1%} | Risk: {pred['risk_level']}")
    
    # Weather impact
    print("\nWeather Impact (Spa at 2h):")
    print("-" * 40)
    
    for weather in ['dry', 'damp', 'wet', 'heavy_rain']:
        pred = predictor.predict_probability(
            circuit='spa',
            race_minute=120,
            weather=weather,
            cars_on_track=28
        )
        print(f"{weather:15} | Prob: {pred['probability']:.1%} | {pred['recommendation']}")
    
    # Race timeline prediction
    print("\n" + "=" * 50)
    print("Race Timeline Prediction (6h at Spa):")
    
    timeline = predictor.predict_race_windows(
        circuit='spa',
        race_duration_minutes=360,
        weather='dry',
        interval_minutes=30
    )
    
    print(timeline.to_string(index=False))
    
    # Generate synthetic training data and train model
    print("\n" + "=" * 50)
    print("Training model on synthetic data...")
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 5000
    
    synthetic_data = pd.DataFrame({
        'lap_number': np.random.randint(1, 100, n_samples),
        'elapsed_time_seconds': np.random.uniform(0, 21600, n_samples),
        'circuit': np.random.choice(['spa', 'le_mans', 'daytona', 'monza'], n_samples),
        'weather': np.random.choice(['dry', 'damp', 'wet'], n_samples, p=[0.7, 0.2, 0.1]),
        'cars_on_track': np.random.randint(20, 35, n_samples),
        'track_temp_c': np.random.uniform(15, 40, n_samples),
        'track_status': np.random.choice(['Green', 'SC', 'FCY'], n_samples, p=[0.95, 0.03, 0.02]),
        'tire_life_pct': np.random.uniform(20, 100, n_samples),
        'class': np.random.choice(['Hypercar', 'LMP2', 'LMGT3'], n_samples),
        'race_id': np.random.choice(['race_1', 'race_2', 'race_3'], n_samples)
    })
    
    # Train
    metrics = predictor.train(synthetic_data)
    
    print(f"\nModel Performance:")
    print(f"  - Accuracy: {metrics['accuracy']:.2%}")
    print(f"  - ROC AUC: {metrics['roc_auc']:.3f}")
    
    # Test trained model
    print("\nPrediction with trained model:")
    pred = predictor.predict_probability(
        circuit='spa',
        race_minute=150,
        weather='wet',
        cars_on_track=25
    )
    print(f"  Probability: {pred['probability']:.1%}")
    print(f"  Risk Level: {pred['risk_level']}")
    print(f"  Recommendation: {pred['recommendation']}")


if __name__ == '__main__':
    main()
