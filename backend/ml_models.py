"""
Machine Learning Models for GutMind Explorer
Real ML models for microbiome-mental health prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
from pathlib import Path

from data_loader import get_research_dataset, get_bacteria_columns, PSYCHOBIOTIC_EFFECTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicrobiomeMentalHealthModel:
    """
    ML model for predicting mental health outcomes from microbiome data.
    Uses ensemble of Random Forest and Gradient Boosting.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.feature_names = []
        self.is_trained = False
        self.training_stats = {}
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare features from dataframe."""
        bacteria_cols = get_bacteria_columns(df)
        self.feature_names = bacteria_cols
        
        X = df[bacteria_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        return X
    
    def train(self, df: pd.DataFrame, target: str = 'anxiety_level') -> Dict[str, Any]:
        """
        Train the model on the research dataset.
        
        Args:
            df: DataFrame with bacteria abundances and target variable
            target: Column name for target ('anxiety_level' or 'depression_level')
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training model for {target}...")
        
        X = self.prepare_features(df)
        y = (df[target] == 'high').astype(int).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_proba = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Train Gradient Boosting
        self.gb_model.fit(X_train_scaled, y_train)
        gb_pred = self.gb_model.predict(X_test_scaled)
        gb_proba = self.gb_model.predict_proba(X_test_scaled)[:, 1]
        
        # Ensemble prediction (average probabilities)
        ensemble_proba = (rf_proba + gb_proba) / 2
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        # Calculate metrics
        self.training_stats = {
            'n_samples': len(df),
            'n_features': len(self.feature_names),
            'target': target,
            'class_distribution': {
                'high': int(y.sum()),
                'low': int(len(y) - y.sum())
            },
            'random_forest': {
                'accuracy': round(accuracy_score(y_test, rf_pred), 3),
                'precision': round(precision_score(y_test, rf_pred), 3),
                'recall': round(recall_score(y_test, rf_pred), 3),
                'f1': round(f1_score(y_test, rf_pred), 3),
                'auc_roc': round(roc_auc_score(y_test, rf_proba), 3)
            },
            'gradient_boosting': {
                'accuracy': round(accuracy_score(y_test, gb_pred), 3),
                'precision': round(precision_score(y_test, gb_pred), 3),
                'recall': round(recall_score(y_test, gb_pred), 3),
                'f1': round(f1_score(y_test, gb_pred), 3),
                'auc_roc': round(roc_auc_score(y_test, gb_proba), 3)
            },
            'ensemble': {
                'accuracy': round(accuracy_score(y_test, ensemble_pred), 3),
                'precision': round(precision_score(y_test, ensemble_pred), 3),
                'recall': round(recall_score(y_test, ensemble_pred), 3),
                'f1': round(f1_score(y_test, ensemble_pred), 3),
                'auc_roc': round(roc_auc_score(y_test, ensemble_proba), 3)
            },
            'feature_importance': self._get_feature_importance()
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        self.training_stats['cv_auc_mean'] = round(cv_scores.mean(), 3)
        self.training_stats['cv_auc_std'] = round(cv_scores.std(), 3)
        
        self.is_trained = True
        logger.info(f"Training complete. Ensemble AUC: {self.training_stats['ensemble']['auc_roc']}")
        
        return self.training_stats
    
    def _get_feature_importance(self) -> List[Dict[str, Any]]:
        """Get feature importance from the trained model."""
        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_
        
        # Average importance
        avg_importance = (rf_importance + gb_importance) / 2
        
        # Combine with known effects
        importance_list = []
        for i, name in enumerate(self.feature_names):
            effect_info = PSYCHOBIOTIC_EFFECTS.get(name, {'effect': 'unknown', 'confidence': 'low'})
            importance_list.append({
                'bacteria': name,
                'importance': round(float(avg_importance[i]), 4),
                'rf_importance': round(float(rf_importance[i]), 4),
                'gb_importance': round(float(gb_importance[i]), 4),
                'known_effect': effect_info.get('effect', 'unknown'),
                'confidence': effect_info.get('confidence', 'low')
            })
        
        # Sort by importance
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        return importance_list
    
    def predict(self, user_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction for user's microbiome profile.
        
        Args:
            user_data: DataFrame with bacteria abundances (single row or multiple)
        
        Returns:
            Prediction results with probabilities and explanations
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Align features with training data
        X = np.zeros((len(user_data), len(self.feature_names)))
        for i, name in enumerate(self.feature_names):
            if name in user_data.columns:
                X[:, i] = user_data[name].values
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        gb_proba = self.gb_model.predict_proba(X_scaled)[:, 1]
        
        # Ensemble
        ensemble_proba = (rf_proba + gb_proba) / 2
        
        results = []
        for i in range(len(user_data)):
            prob = float(ensemble_proba[i])
            prediction = 'high' if prob >= 0.5 else 'low'
            confidence = abs(prob - 0.5) * 2  # 0-1 scale
            
            # Get top contributing features for this prediction
            contributing_factors = self._explain_prediction(X_scaled[i])
            
            results.append({
                'prediction': prediction,
                'probability': round(prob, 3),
                'confidence': round(confidence, 3),
                'risk_percentile': self._calculate_percentile(prob),
                'contributing_factors': contributing_factors
            })
        
        return results[0] if len(results) == 1 else results
    
    def _explain_prediction(self, x_scaled: np.ndarray) -> List[Dict[str, Any]]:
        """Explain which features contributed most to this prediction."""
        # Simple approach: feature value * importance
        importance = (self.rf_model.feature_importances_ + self.gb_model.feature_importances_) / 2
        contributions = x_scaled * importance
        
        factors = []
        for i, name in enumerate(self.feature_names):
            if abs(contributions[i]) > 0.01:  # Only significant contributions
                effect_info = PSYCHOBIOTIC_EFFECTS.get(name, {})
                factors.append({
                    'bacteria': name,
                    'contribution': round(float(contributions[i]), 3),
                    'direction': 'increases risk' if contributions[i] > 0 else 'decreases risk',
                    'known_effect': effect_info.get('effect', 'unknown')
                })
        
        # Sort by absolute contribution
        factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return factors[:5]  # Top 5 factors
    
    def _calculate_percentile(self, probability: float) -> int:
        """Calculate where this probability falls in the training distribution."""
        # Simplified: map probability to percentile
        return int(probability * 100)


class MicrobiomeAnalyzer:
    """Statistical analysis of microbiome data."""
    
    @staticmethod
    def calculate_correlations(df: pd.DataFrame, target: str = 'anxiety_score') -> List[Dict[str, Any]]:
        """Calculate correlations between bacteria and mental health score."""
        bacteria_cols = get_bacteria_columns(df)
        
        correlations = []
        for col in bacteria_cols:
            if col in df.columns and target in df.columns:
                # Pearson correlation
                r, p_value = stats.pearsonr(df[col].fillna(0), df[target].fillna(0))
                
                # Spearman correlation (more robust)
                rho, p_spearman = stats.spearmanr(df[col].fillna(0), df[target].fillna(0))
                
                effect_info = PSYCHOBIOTIC_EFFECTS.get(col, {})
                
                correlations.append({
                    'bacteria': col,
                    'pearson_r': round(r, 3),
                    'pearson_p': round(p_value, 4),
                    'spearman_rho': round(rho, 3),
                    'spearman_p': round(p_spearman, 4),
                    'significant': p_value < 0.05,
                    'direction': 'positive' if r > 0 else 'negative',
                    'known_effect': effect_info.get('effect', 'unknown')
                })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x['pearson_r']), reverse=True)
        
        return correlations
    
    @staticmethod
    def run_pca(df: pd.DataFrame, n_components: int = 2) -> Dict[str, Any]:
        """Run PCA on microbiome data."""
        bacteria_cols = get_bacteria_columns(df)
        X = df[bacteria_cols].fillna(0).values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=min(n_components, len(bacteria_cols)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Prepare results
        result = {
            'coordinates': X_pca.tolist(),
            'explained_variance': [round(v, 3) for v in pca.explained_variance_ratio_],
            'cumulative_variance': [round(sum(pca.explained_variance_ratio_[:i+1]), 3) 
                                   for i in range(len(pca.explained_variance_ratio_))],
            'loadings': {}
        }
        
        # Top loadings for each component
        for i in range(pca.n_components_):
            loadings = list(zip(bacteria_cols, pca.components_[i]))
            loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            result['loadings'][f'PC{i+1}'] = [
                {'bacteria': name, 'loading': round(float(val), 3)} 
                for name, val in loadings[:10]
            ]
        
        return result
    
    @staticmethod
    def run_clustering(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """Run K-means clustering on microbiome data."""
        bacteria_cols = get_bacteria_columns(df)
        X = df[bacteria_cols].fillna(0).values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            mask = clusters == i
            cluster_df = df[mask]
            
            stats_dict = {
                'cluster': i,
                'size': int(mask.sum()),
                'pct_high_anxiety': round(float((cluster_df['anxiety_level'] == 'high').mean() * 100), 1) if 'anxiety_level' in df.columns else None,
                'pct_high_depression': round(float((cluster_df['depression_level'] == 'high').mean() * 100), 1) if 'depression_level' in df.columns else None,
                'mean_anxiety': round(float(cluster_df['anxiety_score'].mean()), 1) if 'anxiety_score' in df.columns else None,
            }
            
            # Top bacteria in this cluster
            top_bacteria = []
            for col in bacteria_cols[:10]:  # Top 10
                top_bacteria.append({
                    'bacteria': col,
                    'mean': round(float(cluster_df[col].mean()), 2),
                    'std': round(float(cluster_df[col].std()), 2)
                })
            stats_dict['top_bacteria'] = top_bacteria
            
            cluster_stats.append(stats_dict)
        
        return {
            'clusters': clusters.tolist(),
            'pca_coordinates': X_pca.tolist(),
            'cluster_stats': cluster_stats,
            'inertia': round(float(kmeans.inertia_), 2)
        }
    
    @staticmethod
    def compare_to_population(user_data: pd.DataFrame, population_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare user's profile to the research population."""
        bacteria_cols = get_bacteria_columns(population_df)
        
        comparisons = []
        for col in bacteria_cols:
            if col in user_data.columns and col in population_df.columns:
                user_val = float(user_data[col].iloc[0])
                pop_mean = float(population_df[col].mean())
                pop_std = float(population_df[col].std())
                
                # Calculate z-score
                z_score = (user_val - pop_mean) / pop_std if pop_std > 0 else 0
                
                # Calculate percentile
                percentile = stats.percentileofscore(population_df[col].dropna(), user_val)
                
                effect_info = PSYCHOBIOTIC_EFFECTS.get(col, {})
                
                comparisons.append({
                    'bacteria': col,
                    'user_value': round(user_val, 2),
                    'population_mean': round(pop_mean, 2),
                    'population_std': round(pop_std, 2),
                    'z_score': round(z_score, 2),
                    'percentile': round(percentile, 1),
                    'status': 'high' if z_score > 1 else ('low' if z_score < -1 else 'normal'),
                    'known_effect': effect_info.get('effect', 'unknown')
                })
        
        # Sort by absolute z-score
        comparisons.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        return {
            'comparisons': comparisons,
            'summary': {
                'n_high': sum(1 for c in comparisons if c['status'] == 'high'),
                'n_low': sum(1 for c in comparisons if c['status'] == 'low'),
                'n_normal': sum(1 for c in comparisons if c['status'] == 'normal'),
                'notable_findings': [c for c in comparisons if abs(c['z_score']) > 1.5][:5]
            }
        }


# Singleton model instance
_model_instance = None

def get_trained_model() -> MicrobiomeMentalHealthModel:
    """Get or create a trained model instance."""
    global _model_instance
    
    if _model_instance is None or not _model_instance.is_trained:
        _model_instance = MicrobiomeMentalHealthModel()
        df = get_research_dataset()
        _model_instance.train(df, target='anxiety_level')
    
    return _model_instance
