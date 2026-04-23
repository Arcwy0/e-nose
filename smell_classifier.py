"""
Enhanced Smell Classification Module - Standalone Version with 22 Features Support

This module provides the complete standalone smell classifier for 22 features
(R1-R17 + T, H, CO2, H2S, CH2O) with comprehensive visualization, data analysis,
and testing capabilities.

Key Changes:
- Support for 22 sensors (17 resistance + 5 environmental)
- Environmental sensors are NOT scaled/preprocessed
- Enhanced environmental sensor analysis and visualization
- Console testing with 22 features
- Backward compatibility removed - retrained for 22 features

Author: Enhanced for Robotics Project
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import os
import time
import argparse
import warnings
import datetime
from typing import Optional, Dict, List, Tuple, Any

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class SmellClassifier:
    """
    High-performance e-nose smell classifier with 22 features support.
    
    Features: R1-R17 (resistance sensors) + T, H, CO2, H2S, CH2O (environmental)
    - Resistance sensors: scaled and preprocessed
    - Environmental sensors: kept as raw values (no scaling)
    """
    
    # Define sensor column categories
    RESISTANCE_SENSORS = [f"R{i}" for i in range(1, 18)]  # R1-R17
    ENVIRONMENTAL_SENSORS = ["T", "H", "CO2", "H2S", "CH2O"]
    ALL_SENSORS = RESISTANCE_SENSORS + ENVIRONMENTAL_SENSORS
    
    def __init__(self, model_type='sgd', random_state=42, online_learning=True):
        """Initialize the smell classifier with 22-feature support."""
        self.model_type = model_type
        self.random_state = random_state
        self.online_learning = online_learning
        
        # Preprocessing components - separate for resistance and environmental
        self.resistance_scaler = StandardScaler()  # Only for R1-R17
        self.variance_threshold = None
        self.feature_selector = None
        
        # Model and training state
        self.model = None
        self.classes_ = None
        self.class_weights_ = None
        self.is_fitted = False
        
        # Feature tracking
        self.original_features = None
        self.active_features = None
        self.selected_features = None
        
        # Training history
        self.training_history = {
            'accuracy': [],
            'timestamps': [],
            'sample_counts': []
        }
        
        # Data storage for analysis
        self.last_training_data = None
        self.last_training_labels = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ML model with parameters for 22-feature support."""
        if self.model_type == 'sgd':
            self.model = SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=0.001,
                max_iter=10000,
                tol=1e-4,
                random_state=self.random_state,
                learning_rate='optimal',
                warm_start=True
            )
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced'
            )
            self.online_learning = False
        elif self.model_type == 'svc':
            self.model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            )
            self.online_learning = False
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def augment_sample(self, sample, noise_std=0.0015):
        """
        Create augmented version of a sample.
        Only augment resistance sensors, keep environmental sensors unchanged.
        """
        if isinstance(sample, pd.Series):
            augmented = sample.copy()
            
            # Only augment resistance sensors (R1-R17)
            for sensor in self.RESISTANCE_SENSORS:
                if sensor in sample.index and sample[sensor] != 0:
                    noise = np.random.normal(0, noise_std)
                    augmented[sensor] = max(0, sample[sensor] + noise)
            
            # Environmental sensors remain unchanged
            return augmented
            
        elif isinstance(sample, dict):
            augmented = sample.copy()
            
            # Only augment resistance sensors
            for sensor in self.RESISTANCE_SENSORS:
                if sensor in sample and isinstance(sample[sensor], (int, float)) and sample[sensor] != 0:
                    noise = np.random.normal(0, noise_std)
                    augmented[sensor] = max(0, sample[sensor] + noise)
            
            # Environmental sensors remain unchanged
            return augmented
        else:
            raise ValueError("Sample must be pandas Series or dict")
    
    def augment_data(self, X, y, n_augmentations=3, noise_std=0.0015):
        """Augment training data with noise only on resistance sensors."""
        if n_augmentations <= 0:
            return X, y
        
        print(f"Augmenting data: {n_augmentations} samples per original (noise_std={noise_std})")
        print("Note: Only resistance sensors (R1-R17) are augmented, environmental sensors unchanged")
        
        X_aug_list = [X]  # Start with original data
        y_aug_list = [y]
        
        for _ in range(n_augmentations):
            X_new = X.apply(lambda row: self.augment_sample(row, noise_std), axis=1)
            X_aug_list.append(X_new)
            y_aug_list.append(y.copy())
        
        X_combined = pd.concat(X_aug_list, ignore_index=True)
        y_combined = pd.concat(y_aug_list, ignore_index=True)
        
        print(f"Data augmented: {len(X)} → {len(X_combined)} samples")
        return X_combined, y_combined
    
    def load_data(self, file_path, target_column='Gas name'):
        """Load and parse 22-feature e-nose data from CSV file."""
        print(f"Loading data from {file_path}...")
        
        # Try different delimiters
        try:
            data = pd.read_csv(file_path, sep=';')
            if len(data.columns) <= 1:
                data = pd.read_csv(file_path, sep=',')
        except:
            data = pd.read_csv(file_path, sep=',')
        
        # Clean column names
        data.columns = [col.rstrip(';').strip() for col in data.columns]
        
        # Find target column
        possible_targets = ['Gas name', 'smell', 'smell_label', 'smell;']
        if target_column not in data.columns:
            for target in possible_targets:
                if target in data.columns:
                    target_column = target
                    break
            else:
                raise ValueError(f"Target column not found. Available: {data.columns.tolist()}")
        
        # Remove unnamed columns
        unnamed_cols = [col for col in data.columns if 'Unnamed' in col]
        if unnamed_cols:
            data = data.drop(columns=unnamed_cols)
            print(f"Removed unnamed columns: {unnamed_cols}")
        
        # Separate features and target
        X = data.drop(columns=[target_column, 'Gas label', 'timestamp'], errors='ignore')
        y = data[target_column]
        
        # Ensure labels are strings
        y = y.astype(str)
        
        # Verify we have the expected 22 features
        expected_sensors = self.ALL_SENSORS
        missing_sensors = [s for s in expected_sensors if s not in X.columns]
        if missing_sensors:
            print(f"Warning: Missing expected sensors: {missing_sensors}")
            # Add missing sensors with zeros
            for sensor in missing_sensors:
                X[sensor] = 0.0
        
        # Select only the 22 expected sensors in correct order
        X = X[expected_sensors]
        
        # Store feature names
        self.original_features = X.columns.tolist()
        self.classes_ = np.array([str(c) for c in np.unique(y)])
        
        print(f"Loaded: {X.shape[1]} features ({len(self.RESISTANCE_SENSORS)} resistance + {len(self.ENVIRONMENTAL_SENSORS)} environmental)")
        print(f"Samples: {X.shape[0]}, Classes: {len(self.classes_)}")
        print(f"Classes: {self.classes_}")
        
        return X, y
    
    def process_sensor_data(self, sensor_data):
        """
        Process 22-feature sensor data for API compatibility.
        
        Args:
            sensor_data: Can be:
                - Single dict with 22 sensor readings
                - List of dicts with 22 sensor readings  
                - DataFrame (returned as-is after validation)
        
        Returns:
            DataFrame with processed 22-feature sensor data
        """
        if isinstance(sensor_data, pd.DataFrame):
            df = sensor_data.fillna(0)
        elif isinstance(sensor_data, dict):
            df = pd.DataFrame([sensor_data])
        elif isinstance(sensor_data, list):
            df = pd.DataFrame(sensor_data)
        else:
            raise ValueError("sensor_data must be dict, list of dicts, or DataFrame")
        
        # Ensure all 22 expected sensor columns exist
        for sensor in self.ALL_SENSORS:
            if sensor not in df.columns:
                df[sensor] = 0.0
        
        # Select only the 22 expected sensors in correct order
        df = df[self.ALL_SENSORS].fillna(0.0)
        
        # Ensure numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        return df
    
    def _preprocess_features(self, X, fit=False):
        """
        Apply preprocessing with separate treatment for resistance and environmental sensors.
        
        Key principle:
        - Resistance sensors (R1-R17): scaled and variance filtered
        - Environmental sensors (T, H, CO2, H2S, CH2O): kept as raw values
        """
        X_processed = X.copy().fillna(0)
        
        # Separate resistance and environmental sensors
        resistance_data = X_processed[self.RESISTANCE_SENSORS]
        environmental_data = X_processed[self.ENVIRONMENTAL_SENSORS]
        
        # Process resistance sensors only
        if fit:
            # Variance threshold for resistance sensors only
            self.variance_threshold = VarianceThreshold(threshold=1e-8)
            resistance_filtered = self.variance_threshold.fit_transform(resistance_data)
            
            # Get active resistance features
            active_resistance = np.array(self.RESISTANCE_SENSORS)[self.variance_threshold.get_support()]
            
            # Scale resistance sensors
            resistance_scaled = self.resistance_scaler.fit_transform(resistance_filtered)
            
            # Track active features (resistance + all environmental)
            self.active_features = list(active_resistance) + self.ENVIRONMENTAL_SENSORS
            
            print(f"Active features: {len(active_resistance)} resistance + {len(self.ENVIRONMENTAL_SENSORS)} environmental = {len(self.active_features)} total")
            
        else:
            # Transform using fitted preprocessors
            if self.variance_threshold is not None:
                resistance_filtered = self.variance_threshold.transform(resistance_data)
                active_resistance = np.array(self.RESISTANCE_SENSORS)[self.variance_threshold.get_support()]
            else:
                resistance_filtered = resistance_data.values
                active_resistance = self.RESISTANCE_SENSORS
            
            resistance_scaled = self.resistance_scaler.transform(resistance_filtered)
        
        # Create processed dataframe
        resistance_df = pd.DataFrame(
            resistance_scaled, 
            index=X_processed.index,
            columns=active_resistance if fit or self.variance_threshold is not None else self.RESISTANCE_SENSORS
        )
        
        # Combine scaled resistance + raw environmental sensors
        X_final = pd.concat([resistance_df, environmental_data], axis=1)
        
        return X_final
    
    def _select_features(self, X, y, k=None):
        """Select features conservatively, preserving environmental sensors."""
        if k is None or k >= X.shape[1]:
            self.selected_features = X.columns.tolist()
            return X
        
        print(f"Selecting top {k} features from {X.shape[1]} available features...")
        print("Note: Environmental sensors will be prioritized in selection")
        
        # Always include environmental sensors
        environmental_features = [col for col in X.columns if col in self.ENVIRONMENTAL_SENSORS]
        resistance_features = [col for col in X.columns if col not in self.ENVIRONMENTAL_SENSORS]
        
        # Select from resistance sensors if we need more features
        k_resistance = max(0, k - len(environmental_features))
        
        if k_resistance > 0 and len(resistance_features) > 0:
            selector = SelectKBest(f_classif, k=min(k_resistance, len(resistance_features)))
            X_resistance = X[resistance_features]
            selector.fit(X_resistance, y)
            
            selected_mask = selector.get_support()
            selected_resistance = np.array(resistance_features)[selected_mask].tolist()
        else:
            selected_resistance = []
        
        # Combine environmental + selected resistance features
        self.selected_features = environmental_features + selected_resistance
        
        print(f"Selected features: {len(environmental_features)} environmental + {len(selected_resistance)} resistance = {len(self.selected_features)} total")
        print(f"Selected features: {self.selected_features}")
        
        return X[self.selected_features]
    
    def train(self, X, y, test_size=0.2, use_augmentation=True, 
              n_augmentations=5, noise_std=0.0015, k_features=None):
        """Train the 22-feature smell classifier."""
        print(f"\n=== Training {self.model_type.upper()} Classifier (22 Features) ===")
        
        # Store data for analysis
        self.last_training_data = X.copy()
        self.last_training_labels = y.copy()
        
        # Ensure string labels
        y = y.astype(str)
        self.classes_ = np.array([str(c) for c in np.unique(y)])
        
        # Data augmentation (only on resistance sensors)
        if use_augmentation:
            X_train, y_train = self.augment_data(X, y, n_augmentations, noise_std)
        else:
            X_train, y_train = X, y
        
        # Split data
        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train, y_train, test_size=test_size, 
            random_state=self.random_state, stratify=y_train
        )
        
        # Preprocess features (separate treatment for resistance vs environmental)
        print("Preprocessing features (resistance sensors scaled, environmental sensors raw)...")
        X_train_processed = self._preprocess_features(X_train_split, fit=True)
        
        # Feature selection (prioritizing environmental sensors)
        X_train_final = self._select_features(X_train_processed, y_train_split, k_features)
        
        # Calculate class weights for SGD
        if self.model_type == 'sgd':
            try:
                weights = class_weight.compute_class_weight(
                    'balanced', classes=self.classes_, y=y_train_split
                )
                self.class_weights_ = dict(zip(self.classes_, weights))
                print(f"Class weights: {self.class_weights_}")
            except Exception as e:
                print(f"Warning: Could not compute class weights: {e}")
                self.class_weights_ = None
        
        # Train model
        print(f"Training with {X_train_final.shape[0]} samples, {X_train_final.shape[1]} features...")
        start_time = time.time()
        
        if self.model_type == 'sgd':
            self.model.partial_fit(X_train_final, y_train_split, classes=self.classes_)
        else:
            self.model.fit(X_train_final, y_train_split)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set
        X_test_processed = self._preprocess_features(X_test, fit=False)
        X_test_final = X_test_processed[self.selected_features]
        
        y_pred = self.model.predict(X_test_final)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n=== Evaluation Results ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Generate and save confusion matrix
        self._save_confusion_matrix(y_test, y_pred, "initial_training")
        
        # Update training history
        self.training_history['accuracy'].append(accuracy)
        self.training_history['timestamps'].append(time.time())
        self.training_history['sample_counts'].append(len(X_train_final))
        
        self.is_fitted = True
        return accuracy
    
    def online_update(self, X_new, y_new, use_augmentation=False, 
                     n_augmentations=1, noise_std=0.0015):
        """Update model with new 22-feature data (online learning)."""
        if not self.is_fitted:
            print("Model not trained. Use train() first.")
            return False
        
        if not self.online_learning:
            print(f"Model type {self.model_type} doesn't support online learning.")
            return False
        
        # Ensure string labels
        if isinstance(y_new, list):
            y_new = pd.Series(y_new, name='smell')
        y_new = y_new.astype(str)
        
        # Check for new classes
        new_classes = set(y_new.unique()) - set(self.classes_)
        if new_classes:
            print(f"New classes detected: {new_classes}")
            self.classes_ = np.array(sorted(list(set(self.classes_) | set(y_new.unique()))))
        
        # Optional augmentation (only resistance sensors)
        if use_augmentation:
            X_update, y_update = self.augment_data(X_new, y_new, n_augmentations, noise_std)
        else:
            X_update, y_update = X_new, y_new
        
        # Preprocess new data
        X_processed = self._preprocess_features(X_update, fit=False)
        X_final = X_processed[self.selected_features]
        
        # Update model
        try:
            self.model.partial_fit(X_final, y_update, classes=self.classes_)
            print(f"Model updated with {len(X_update)} samples")
            
            # Update history
            last_accuracy = self.training_history['accuracy'][-1] if self.training_history['accuracy'] else 0
            self.training_history['accuracy'].append(last_accuracy)
            self.training_history['timestamps'].append(time.time())
            cumulative = self.training_history['sample_counts'][-1] if self.training_history['sample_counts'] else 0
            self.training_history['sample_counts'].append(cumulative + len(X_final))
            
            return True
        except Exception as e:
            print(f"Error during online update: {e}")
            return False
    
    def predict(self, X):
        """Make predictions on 22-feature data."""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_processed = self._preprocess_features(X, fit=False)
        X_final = X_processed[self.selected_features]
        
        return self.model.predict(X_final)
    
    def predict_proba(self, X):
        """Get prediction probabilities for 22-feature data."""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_processed = self._preprocess_features(X, fit=False)
        X_final = X_processed[self.selected_features]
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_final)
        elif hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(X_final)
            if len(self.classes_) == 2:
                probs = 1 / (1 + np.exp(-decision))
                return np.column_stack([1 - probs, probs])
            else:
                exp_vals = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            return np.ones((len(X_final), len(self.classes_))) / len(self.classes_)
    
    def get_model_info(self):
        """Get comprehensive model information for 22-feature setup."""
        info = {
            "model_loaded": True,
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "supports_online_learning": self.online_learning,
            "random_state": self.random_state,
            "feature_info": {
                "total_features": 22,
                "resistance_sensors": len(self.RESISTANCE_SENSORS),
                "environmental_sensors": len(self.ENVIRONMENTAL_SENSORS),
                "original_features": len(self.original_features) if self.original_features else 0,
                "active_features": len(self.active_features) if self.active_features else 0,
                "selected_features": len(self.selected_features) if self.selected_features else 0,
                "feature_names": self.selected_features if self.selected_features else []
            }
        }
        
        if self.is_fitted:
            info["classes"] = self.classes_.tolist()
            info["n_classes"] = len(self.classes_)
            
            if self.training_history['accuracy']:
                info["current_accuracy"] = self.training_history['accuracy'][-1]
                info["training_samples"] = self.training_history['sample_counts'][-1] if self.training_history['sample_counts'] else 0
                info["training_iterations"] = len(self.training_history['accuracy'])
            
            if self.class_weights_:
                info["class_weights"] = self.class_weights_
        
        return info
    
    def test_console_input(self):
        """Test classifier with 22-feature console input."""
        if not self.is_fitted:
            print("Model not trained yet. Cannot test with console input.")
            return
        
        print("\n=== Console Testing Mode (22 Features) ===")
        print("Enter sensor values as comma-separated values:")
        print("Format: R1,R2,...,R17,T,H,CO2,H2S,CH2O (22 values total)")
        print("Example: 0.5,0.3,0.8,0.0,0.9,0.7,0.6,0.4,0.8,0.5,0.7,0.3,0.9,0.6,0.8,0.4,0.7,21.5,45.2,400,0.1,5.0")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("Enter 22 sensor values: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Exiting console testing mode")
                    break
                
                if not user_input:
                    continue
                
                # Parse input
                values = [float(x.strip()) for x in user_input.split(',')]
                
                if len(values) != 22:
                    print(f"Error: Expected 22 values, got {len(values)}. Please try again.")
                    print("Format: R1-R17 (17 values) + T,H,CO2,H2S,CH2O (5 values) = 22 total")
                    continue
                
                # Create DataFrame
                sensor_dict = {}
                # R1-R17
                for i in range(17):
                    sensor_dict[f'R{i+1}'] = values[i]
                # Environmental sensors
                env_sensors = ['T', 'H', 'CO2', 'H2S', 'CH2O']
                for i, sensor in enumerate(env_sensors):
                    sensor_dict[sensor] = values[17 + i]
                
                test_df = pd.DataFrame([sensor_dict])
                
                # Make prediction
                prediction = self.predict(test_df)[0]
                probabilities = self.predict_proba(test_df)[0]
                
                # Display results
                print(f"\n--- Prediction Results ---")
                print(f"Predicted smell: {prediction}")
                print(f"Confidence: {max(probabilities):.3f}")
                print(f"\nAll probabilities:")
                
                prob_dict = dict(zip(self.classes_, probabilities))
                sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                
                for smell, prob in sorted_probs:
                    print(f"  {smell}: {prob:.3f}")
                
                # Show input breakdown
                print(f"\nInput breakdown:")
                print(f"  Resistance sensors (R1-R17): {values[:17]}")
                print(f"  Environmental sensors (T,H,CO2,H2S,CH2O): {values[17:]}")
                print()
                
            except ValueError as e:
                print(f"Error: Invalid input format. {e}")
                print("Please enter 22 comma-separated numbers")
            except KeyboardInterrupt:
                print("\nExiting console testing mode")
                break
            except Exception as e:
                print(f"Error during prediction: {e}")
    
    def analyze_environmental_sensors(self, data=None, output_dir="data"):
        """
        Analyze relationships between environmental sensors and odor detection.
        This is a new analysis specifically for the 22-feature setup.
        """
        if data is None:
            if self.last_training_data is not None:
                X = self.last_training_data
                y = self.last_training_labels
            else:
                raise ValueError("No data available for environmental sensor analysis")
        else:
            X, y = data if isinstance(data, tuple) else (data, self.last_training_labels)
        
        print("\n=== Environmental Sensor Analysis ===")
        os.makedirs(output_dir, exist_ok=True)
        
        # Environmental sensor statistics by class
        env_stats = {}
        for sensor in self.ENVIRONMENTAL_SENSORS:
            if sensor in X.columns:
                stats_by_class = {}
                for class_name in np.unique(y):
                    class_data = X.loc[y == class_name, sensor]
                    stats_by_class[class_name] = {
                        'mean': class_data.mean(),
                        'std': class_data.std(),
                        'min': class_data.min(),
                        'max': class_data.max(),
                        'median': class_data.median()
                    }
                env_stats[sensor] = stats_by_class
        
        # Print summary
        print("Environmental sensor ranges by odor class:")
        for sensor in self.ENVIRONMENTAL_SENSORS:
            if sensor in env_stats:
                print(f"\n{sensor}:")
                for class_name, stats in env_stats[sensor].items():
                    print(f"  {class_name}: {stats['mean']:.2f}±{stats['std']:.2f} (range: {stats['min']:.2f}-{stats['max']:.2f})")
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Environmental Sensors Analysis by Odor Class", fontsize=16)
        axes = axes.flatten()
        
        for i, sensor in enumerate(self.ENVIRONMENTAL_SENSORS):
            if i < len(axes) and sensor in X.columns:
                ax = axes[i]
                
                # Box plot for each class
                plot_data = []
                plot_labels = []
                for class_name in sorted(np.unique(y)):
                    class_data = X.loc[y == class_name, sensor].values
                    plot_data.append(class_data)
                    plot_labels.append(class_name)
                
                ax.boxplot(plot_data, labels=plot_labels)
                ax.set_title(f'{sensor} by Odor Class')
                ax.set_ylabel(f'{sensor} Value')
                ax.tick_params(axis='x', rotation=45)
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        if len(self.ENVIRONMENTAL_SENSORS) < len(axes):
            axes[-1].axis('off')
        
        plt.tight_layout()
        env_plot_path = os.path.join(output_dir, "environmental_sensors_analysis.png")
        fig.savefig(env_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\nEnvironmental sensor analysis plot saved to {env_plot_path}")
        
        # Correlation analysis between environmental sensors and resistance sensors
        fig2, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Calculate correlations between environmental and resistance sensors
        resistance_cols = [col for col in X.columns if col in self.RESISTANCE_SENSORS]
        env_cols = [col for col in X.columns if col in self.ENVIRONMENTAL_SENSORS]
        
        if len(resistance_cols) > 0 and len(env_cols) > 0:
            # Get top 10 most active resistance sensors
            top_resistance = (X[resistance_cols] != 0).sum().nlargest(10).index.tolist()
            
            # Calculate cross-correlation matrix
            cross_corr_data = pd.concat([X[env_cols], X[top_resistance]], axis=1)
            cross_corr = cross_corr_data.corr()
            
            # Extract environmental vs resistance correlations
            env_resistance_corr = cross_corr.loc[env_cols, top_resistance]
            
            # Create heatmap
            sns.heatmap(env_resistance_corr, annot=True, cmap='RdBu_r', center=0,
                       fmt='.3f', ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Environmental vs Resistance Sensors Correlation')
            ax.set_xlabel('Top Resistance Sensors')
            ax.set_ylabel('Environmental Sensors')
            
            plt.tight_layout()
            corr_plot_path = os.path.join(output_dir, "environmental_resistance_correlation.png")
            fig2.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            print(f"Environmental-resistance correlation plot saved to {corr_plot_path}")
        
        return env_stats
    
    def visualize_data(self, data=None, file_path=None):
        """Enhanced data visualization for 22 features including environmental analysis."""
        if data is None and file_path is None:
            if self.last_training_data is not None:
                X = self.last_training_data
                y = self.last_training_labels
                print("Using last training data for visualization")
            else:
                raise ValueError("No data available for visualization")
        elif data is None:
            try:
                X, y = self.load_data(file_path)
            except Exception as e:
                print(f"Error loading data for visualization: {e}")
                return None
        else:
            if isinstance(data, tuple) and len(data) == 2:
                X, y = data
            else:
                raise ValueError("Data must be a tuple of (X, y)")
        
        print(f"Visualizing 22-feature data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        os.makedirs("data", exist_ok=True)
        
        # Get sensor columns
        resistance_cols = [col for col in X.columns if col in self.RESISTANCE_SENSORS]
        env_cols = [col for col in X.columns if col in self.ENVIRONMENTAL_SENSORS]
        
        try:
            # Figure 1: Enhanced Data Overview (22 features)
            fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig1.suptitle("E-nose Data Overview (22 Features)", fontsize=16)
            
            # 1.1 Average resistance readings by class
            ax1 = axes[0, 0]
            try:
                if resistance_cols:
                    avg_resistance = pd.concat([X[resistance_cols], y], axis=1).groupby(y.name)[resistance_cols].mean()
                    # Show top 10 most active resistance sensors
                    top_resistance = (X[resistance_cols] != 0).sum().nlargest(10).index
                    avg_resistance[top_resistance].T.plot(kind='bar', ax=ax1, legend=True)
                    ax1.set_title('Top 10 Resistance Sensors by Class')
                    ax1.set_xlabel('Sensor')
                    ax1.set_ylabel('Average Reading')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            except Exception as e:
                ax1.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax1.transAxes)
            
            # 1.2 Environmental sensors by class
            ax2 = axes[0, 1]
            try:
                if env_cols:
                    avg_env = pd.concat([X[env_cols], y], axis=1).groupby(y.name)[env_cols].mean()
                    avg_env.T.plot(kind='bar', ax=ax2, legend=True)
                    ax2.set_title('Environmental Sensors by Class')
                    ax2.set_xlabel('Environmental Sensor')
                    ax2.set_ylabel('Average Value')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            except Exception as e:
                ax2.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax2.transAxes)
            
            # 1.3 Sensor activity overview
            ax3 = axes[0, 2]
            try:
                resistance_activity = (X[resistance_cols] != 0).sum() if resistance_cols else pd.Series()
                env_activity = X[env_cols].count() if env_cols else pd.Series()
                
                activity_data = pd.concat([resistance_activity.head(10), env_activity])
                activity_data.plot(kind='bar', ax=ax3)
                ax3.set_title('Sensor Activity (Non-zero/Valid Count)')
                ax3.set_xlabel('Sensor')
                ax3.set_ylabel('Active Count')
                ax3.tick_params(axis='x', rotation=45)
            except Exception as e:
                ax3.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax3.transAxes)
            
            # 1.4 Class distribution
            ax4 = axes[1, 0]
            try:
                class_counts = y.value_counts()
                class_counts.plot(kind='bar', ax=ax4)
                ax4.set_title('Class Distribution')
                ax4.set_xlabel('Class')
                ax4.set_ylabel('Count')
                ax4.tick_params(axis='x', rotation=45)
            except Exception as e:
                ax4.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax4.transAxes)
            
            # 1.5 Environmental sensor ranges
            ax5 = axes[1, 1]
            try:
                if env_cols:
                    env_ranges = X[env_cols].max() - X[env_cols].min()
                    env_ranges.plot(kind='bar', ax=ax5)
                    ax5.set_title('Environmental Sensor Ranges')
                    ax5.set_xlabel('Environmental Sensor')
                    ax5.set_ylabel('Range (Max - Min)')
                    ax5.tick_params(axis='x', rotation=45)
            except Exception as e:
                ax5.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax5.transAxes)
            
            # 1.6 Feature type comparison
            ax6 = axes[1, 2]
            try:
                feature_summary = {
                    'Resistance Sensors': len(resistance_cols),
                    'Environmental Sensors': len(env_cols),
                    'Active Resistance': (X[resistance_cols] != 0).any().sum() if resistance_cols else 0,
                    'Valid Environmental': X[env_cols].notna().any().sum() if env_cols else 0
                }
                
                pd.Series(feature_summary).plot(kind='bar', ax=ax6)
                ax6.set_title('Feature Type Summary')
                ax6.set_ylabel('Count')
                ax6.tick_params(axis='x', rotation=45)
            except Exception as e:
                ax6.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax6.transAxes)
            
            plt.tight_layout()
            overview_path = os.path.join("data", "data_overview_22features.png")
            fig1.savefig(overview_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            print(f"22-feature data overview saved to {overview_path}")
            
        except Exception as e:
            print(f"Error generating data overview: {e}")
        
        # Generate environmental sensor analysis
        try:
            self.analyze_environmental_sensors(data=(X, y), output_dir="data")
        except Exception as e:
            print(f"Error generating environmental analysis: {e}")
        
        print("22-feature data visualization completed!")
        return True
    
    def _save_confusion_matrix(self, y_true, y_pred, suffix=""):
        """Save confusion matrix plot to data folder."""
        os.makedirs("data", exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred, labels=self.classes_)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes_, yticklabels=self.classes_)
        plt.title(f'Confusion Matrix{" - " + suffix if suffix else ""} (22 Features)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        filename = f"confusion_matrix_{suffix}_22features.png" if suffix else "confusion_matrix_22features.png"
        filepath = os.path.join("data", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {filepath}")
    
    def save_model(self, filepath=None):
        """Save the trained 22-feature model with automatic timestamping."""
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
        
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"smell_classifier_{self.model_type}_22features_{timestamp}.joblib"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        joblib.dump(self, filepath)
        print(f"22-feature model saved to {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved 22-feature model."""
        model = joblib.load(filepath)
        print(f"22-feature model loaded from {filepath}")
        
        # Verify it's a 22-feature model
        if hasattr(model, 'original_features') and model.original_features:
            if len(model.original_features) != 22:
                print(f"Warning: Loaded model has {len(model.original_features)} features, expected 22")
        
        return model


def main():
    """Main function for command-line usage with 22-feature support."""
    parser = argparse.ArgumentParser(description='Enhanced Smell Classifier (22 Features: R1-R17 + T,H,CO2,H2S,CH2O)')
    parser.add_argument('--data', required=True, help='Path to training data CSV with 22 features')
    parser.add_argument('--model_type', choices=['sgd', 'rf', 'svc'], default='sgd')
    parser.add_argument('--test', action='store_true', help='Run comprehensive test')
    parser.add_argument('--console', action='store_true', help='Test with 22-feature console input')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--analyze', action='store_true', help='Analyze data quality')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation (default: ON)')
    parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--n_aug', type=int, default=5, help='Number of augmentations (optimal: 5)')
    parser.add_argument('--noise', type=float, default=0.0015, help='Augmentation noise level')
    parser.add_argument('--k_features', type=int, help='Number of features to select (default: all 22)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--save', help='Path to save trained model')
    parser.add_argument('--env_analysis', action='store_true', help='Generate environmental sensor analysis')
    
    args = parser.parse_args()
    
    # Default to augmentation ON with optimal n_aug=5
    use_augmentation = not args.no_augment
    if args.augment:
        use_augmentation = True
    
    # Initialize 22-feature classifier
    classifier = SmellClassifier(model_type=args.model_type)
    
    if args.test:
        # Comprehensive test mode
        print("=== Comprehensive Test Mode (22 Features) ===")
        
        # Load and train
        X, y = classifier.load_data(args.data)
        accuracy = classifier.train(
            X, y,
            test_size=args.test_size,
            use_augmentation=use_augmentation,
            n_augmentations=args.n_aug,
            noise_std=args.noise,
            k_features=args.k_features
        )
        
        print(f"Training completed with {accuracy:.4f} accuracy")
        
        # Generate all visualizations
        classifier.visualize_data()
        classifier.analyze_data_quality()
        classifier.get_feature_importance()
        classifier.plot_training_history()
        classifier.analyze_environmental_sensors()
        
        # Console testing
        if input("\nTest with 22-feature console input? (y/n): ").lower() == 'y':
            classifier.test_console_input()
        
        # Save model
        if args.save:
            classifier.save_model(args.save)
        
    else:
        # Individual actions
        X, y = classifier.load_data(args.data)
        
        if not args.console:
            # Train model
            accuracy = classifier.train(
                X, y,
                test_size=args.test_size,
                use_augmentation=use_augmentation,
                n_augmentations=args.n_aug,
                noise_std=args.noise,
                k_features=args.k_features
            )
            
            print(f"Training completed with {accuracy:.4f} accuracy")
            
            # Feature importance
            classifier.get_feature_importance()
        
        if args.visualize:
            classifier.visualize_data(data=(X, y))
        
        if args.analyze:
            classifier.analyze_data_quality(data=(X, y))
        
        if args.env_analysis:
            classifier.analyze_environmental_sensors(data=(X, y))
        
        if args.console:
            if not classifier.is_fitted:
                print("Training model first...")
                classifier.train(X, y, use_augmentation=use_augmentation, 
                               n_augmentations=args.n_aug, noise_std=args.noise)
            classifier.test_console_input()
        
        # Save model
        if args.save:
            classifier.save_model(args.save)
    
    print(f"\nAll outputs saved to 'data' folder")
    
    # Performance benchmark
    if hasattr(classifier, 'training_history') and classifier.training_history['accuracy']:
        final_accuracy = classifier.training_history['accuracy'][-1]
        if final_accuracy >= 0.80:
            print("✅ PERFORMANCE TARGET MET: Accuracy >= 80%")
        else:
            print(f"⚠️ PERFORMANCE BELOW TARGET: {final_accuracy:.4f} < 0.80")
    
    print("Note: This model supports 22 features (R1-R17 + T, H, CO2, H2S, CH2O)")


if __name__ == "__main__":
    main()