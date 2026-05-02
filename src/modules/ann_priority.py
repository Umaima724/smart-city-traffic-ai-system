"""
ann_priority.py
Module 3: ANN Priority Prediction Module for the Smart City Traffic AI System.

This module estimates urgency or priority for requests that require priority-aware
handling. In the traffic domain, this is especially important for emergency response.
The module is NOT designed to decide whether a request is legally allowed.
Instead, it predicts how urgent a situation appears based on structured operational
features.

The module contains TWO neural network implementations:

1. BINARY CLASSIFIER (Perceptron):
   - Single neuron with weighted sum and sigmoid activation
   - Inputs: VehicleType, Severity, TimeSensitivity, TrafficDensity, Distance, PriorityClaim
   - Output: Urgent? {0,1}
   - Used for quick binary urgency classification

2. MULTI-CLASS MLP (Multi-Layer Perceptron):
   - 2 Hidden Layers with configurable neurons
   - Inputs: Same 6 features
   - Outputs: Low, Normal, High, Critical (4 classes)
   - Used for detailed priority level classification

Both networks use manual backpropagation implemented with NumPy for educational
demonstration. The module includes training functionality using the provided
training data and prediction functionality for real-time requests.

Author: [Your Name]
Group: [Your Group]
Course: AL-2002 Artificial Intelligence Lab
"""


import json
import numpy as np
import os

from src.config import (
    ANN_INPUT_FEATURES,
    ANN_BINARY_OUTPUT,
    ANN_MULTICLASS_OUTPUT,
    ANN_TRAINING_DATA_PATH
)
from src.utils.exceptions import (
    InvalidRequestError,
    DataLoadError,
    InvalidValueError
)


# =============================================================================
# BINARY CLASSIFIER: Single Neuron with Sigmoid Activation
# =============================================================================

class BinaryPriorityClassifier:
    """
    Binary Priority Classifier using a single neuron (Perceptron).
    
    This classifier predicts whether a traffic request is URGENT (1) or
    NOT URGENT (0) based on 6 input features. It uses weighted sum with
    bias and sigmoid activation function.
    
    Architecture:
        [6 Inputs] -> [Weighted Sum + Bias] -> [Sigmoid] -> [Binary Output {0,1}]
    
    Features:
        - VehicleType (0-3)
        - Severity (0-3)
        - TimeSensitivity (0-1)
        - TrafficDensity (0-10)
        - Distance (km)
        - PriorityClaim (0-3)
    
    Attributes:
        weights (np.ndarray): 6 input weights
        bias (float): Bias term
        learning_rate (float): Training learning rate
    """
    
    def __init__(self, learning_rate=0.1):
        """
        Initialize the Binary Priority Classifier.
        
        Args:
            learning_rate (float): Learning rate for gradient descent.
                                   Default: 0.1
        """
        self.weights = np.random.randn(ANN_INPUT_FEATURES) * 0.1
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.training_history = []
    
    # =====================================================================
    # FUNCTION 1: sigmoid
    # =====================================================================
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Maps any real-valued number to the range (0, 1).
        Formula: sigma(z) = 1 / (1 + e^(-z))
        
        Args:
            z (float or np.ndarray): Input value(s)
        
        Returns:
            float or np.ndarray: Sigmoid output in range (0, 1)
        
        Example:
            >>> classifier = BinaryPriorityClassifier()
            >>> classifier.sigmoid(0)
            0.5
            >>> classifier.sigmoid(2)
            0.880797...
        """
        # Clip to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    # =====================================================================
    # FUNCTION 2: sigmoid_derivative
    # =====================================================================
    def sigmoid_derivative(self, z):
        """
        Derivative of sigmoid function for backpropagation.
        
        Formula: sigma'(z) = sigma(z) * (1 - sigma(z))
        
        Args:
            z (float or np.ndarray): Input value(s)
        
        Returns:
            float or np.ndarray: Derivative value(s)
        """
        s = self.sigmoid(z)
        return s * (1.0 - s)
    
    # =====================================================================
    # FUNCTION 3: forward_pass
    # =====================================================================
    def forward_pass(self, X):
        """
        Perform forward propagation through the neuron.
        
        Computes: z = w1*x1 + w2*x2 + ... + w6*x6 + b
                  output = sigmoid(z)
        
        Args:
            X (np.ndarray): Input feature vector of shape (6,) or (batch, 6)
        
        Returns:
            dict: Contains 'z' (weighted sum) and 'output' (sigmoid result)
        
        Example:
            >>> classifier = BinaryPriorityClassifier()
            >>> X = np.array([1, 3, 1, 8, 4.5, 3])
            >>> result = classifier.forward_pass(X)
            >>> print(result['output'])
            0.5... (random initial weights)
        """
        # Ensure X is 2D for batch processing
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Weighted sum: z = X @ weights + bias
        z = np.dot(X, self.weights) + self.bias
        
        # Sigmoid activation
        output = self.sigmoid(z)
        
        return {
            'z': z,
            'output': output
        }
    
    # =====================================================================
    # FUNCTION 4: predict
    # =====================================================================
    def predict(self, X):
        """
        Predict urgency for given input features.
        
        Args:
            X (np.ndarray or list): Input feature vector [6 elements]
        
        Returns:
            dict: Prediction result with:
                - 'probability': Sigmoid output (0-1)
                - 'prediction': Binary class {0, 1}
                - 'label': Human-readable label
        
        Example:
            >>> classifier = BinaryPriorityClassifier()
            >>> result = classifier.predict([1, 3, 1, 8, 4.5, 3])
            >>> print(result['label'])
            'Urgent' or 'Not Urgent'
        """
        X = np.array(X, dtype=float)
        
        if X.shape[0] != ANN_INPUT_FEATURES:
            raise InvalidValueError(
                'feature_vector',
                f"shape {X.shape}",
                f"vector with {ANN_INPUT_FEATURES} features"
            )
        
        # Forward pass
        result = self.forward_pass(X)
        probability = float(result['output'][0])
        
        # Binary threshold at 0.5
        prediction = 1 if probability >= 0.5 else 0
        
        label = "Urgent" if prediction == 1 else "Not Urgent"
        
        return {
            'probability': probability,
            'prediction': prediction,
            'label': label
        }
    
    # =====================================================================
    # FUNCTION 5: train
    # =====================================================================
    def train(self, X_train, y_train, epochs=100, verbose=False):
        """
        Train the binary classifier using gradient descent.
        
        Uses Mean Squared Error (MSE) loss and updates weights via
        backpropagation.
        
        Loss: L = (1/2) * (y_pred - y_true)^2
        dL/dw = (y_pred - y_true) * sigmoid'(z) * x
        dL/db = (y_pred - y_true) * sigmoid'(z)
        
        Args:
            X_train (np.ndarray): Training features, shape (n_samples, 6)
            y_train (np.ndarray): Training labels, shape (n_samples,)
            epochs (int): Number of training iterations
            verbose (bool): Whether to print progress
        
        Returns:
            dict: Training history with loss per epoch
        
        Example:
            >>> X = np.array([[1,3,1,8,4.5,3], [0,0,0,2,5,0]])
            >>> y = np.array([1, 0])
            >>> history = classifier.train(X, y, epochs=50)
        """
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)
        
        n_samples = X_train.shape[0]
        self.training_history = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for i in range(n_samples):
                x = X_train[i]
                y_true = y_train[i]
                
                # Forward pass
                result = self.forward_pass(x)
                y_pred = result['output'][0]
                z = result['z'][0]
                
                # Compute loss (MSE)
                loss = 0.5 * (y_pred - y_true) ** 2
                total_loss += loss
                
                # Backpropagation
                error = y_pred - y_true
                dz = error * self.sigmoid_derivative(z)
                
                # Update weights and bias
                self.weights -= self.learning_rate * dz * x
                self.bias -= self.learning_rate * dz
            
            avg_loss = total_loss / n_samples
            self.training_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return {
            'epochs': epochs,
            'final_loss': self.training_history[-1],
            'loss_history': self.training_history
        }


# =============================================================================
# MULTI-CLASS MLP: 2 Hidden Layers
# =============================================================================

class MultiClassPriorityMLP:
    """
    Multi-Class Priority MLP with 2 Hidden Layers.
    
    This neural network classifies requests into 4 priority levels:
    Low (0), Normal (1), High (2), Critical (3).
    
    Architecture:
        Input Layer:  6 neurons (features)
        Hidden Layer 1: configurable neurons (default: 8)
        Hidden Layer 2: configurable neurons (default: 6)
        Output Layer: 4 neurons (classes) with softmax
    
    Activation: Sigmoid for hidden layers, Softmax for output
    
    Attributes:
        input_size (int): Number of input features (6)
        hidden1_size (int): Neurons in first hidden layer
        hidden2_size (int): Neurons in second hidden layer
        output_size (int): Number of output classes (4)
        learning_rate (float): Training learning rate
    """
    
    def __init__(self, hidden1_size=8, hidden2_size=6, learning_rate=0.1):
        """
        Initialize the Multi-Class Priority MLP.
        
        Args:
            hidden1_size (int): Neurons in first hidden layer. Default: 8
            hidden2_size (int): Neurons in second hidden layer. Default: 6
            learning_rate (float): Learning rate. Default: 0.1
        """
        self.input_size = ANN_INPUT_FEATURES
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = ANN_MULTICLASS_OUTPUT
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(self.input_size, self.hidden1_size) * 0.1
        self.b1 = np.zeros(self.hidden1_size)
        
        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * 0.1
        self.b2 = np.zeros(self.hidden2_size)
        
        self.W3 = np.random.randn(self.hidden2_size, self.output_size) * 0.1
        self.b3 = np.zeros(self.output_size)
        
        self.training_history = []
    
    # =====================================================================
    # FUNCTION 6: sigmoid
    # =====================================================================
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Args:
            z (np.ndarray): Input array
        
        Returns:
            np.ndarray: Sigmoid output
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    # =====================================================================
    # FUNCTION 7: sigmoid_derivative
    # =====================================================================
    def sigmoid_derivative(self, a):
        """
        Derivative of sigmoid given activation value.
        
        Args:
            a (np.ndarray): Sigmoid activation output
        
        Returns:
            np.ndarray: Derivative
        """
        return a * (1.0 - a)
    
    # =====================================================================
    # FUNCTION 8: softmax
    # =====================================================================
    def softmax(self, z):
        """
        Softmax activation for output layer.
        
        Converts raw scores to probability distribution.
        Formula: softmax(z_i) = e^(z_i) / sum(e^(z_j))
        
        Args:
            z (np.ndarray): Raw output scores
        
        Returns:
            np.ndarray: Probability distribution (sums to 1)
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    
    # =====================================================================
    # FUNCTION 9: forward_pass
    # =====================================================================
    def forward_pass(self, X):
        """
        Perform forward propagation through the entire MLP.
        
        Layer 1: z1 = X @ W1 + b1, a1 = sigmoid(z1)
        Layer 2: z2 = a1 @ W2 + b2, a2 = sigmoid(z2)
        Output:  z3 = a2 @ W3 + b3, a3 = softmax(z3)
        
        Args:
            X (np.ndarray): Input features, shape (6,) or (batch, 6)
        
        Returns:
            dict: All intermediate values for backpropagation
        
        Example:
            >>> mlp = MultiClassPriorityMLP()
            >>> X = np.array([1, 3, 1, 8, 4.5, 3])
            >>> result = mlp.forward_pass(X)
            >>> print(result['output'])
            [0.25, 0.25, 0.25, 0.25] (initial random weights)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        # Output Layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        
        return {
            'z1': self.z1, 'a1': self.a1,
            'z2': self.z2, 'a2': self.a2,
            'z3': self.z3, 'a3': self.a3,
            'output': self.a3
        }
    
    # =====================================================================
    # FUNCTION 10: predict
    # =====================================================================
    def predict(self, X):
        """
        Predict priority level for given input features.
        
        Args:
            X (np.ndarray or list): Input feature vector [6 elements]
        
        Returns:
            dict: Prediction result with:
                - 'probabilities': Probability for each class
                - 'predicted_class': Integer class (0-3)
                - 'priority_level': Human-readable priority
                - 'confidence': Probability of predicted class
        
        Example:
            >>> mlp = MultiClassPriorityMLP()
            >>> result = mlp.predict([1, 3, 1, 8, 4.5, 3])
            >>> print(result['priority_level'])
            'Critical' or other level
        """
        X = np.array(X, dtype=float)
        
        if X.shape[0] != ANN_INPUT_FEATURES:
            raise InvalidValueError(
                'feature_vector',
                f"shape {X.shape}",
                f"vector with {ANN_INPUT_FEATURES} features"
            )
        
        # Forward pass
        result = self.forward_pass(X)
        probabilities = result['output'][0]
        
        # Get predicted class
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        # Map to priority levels
        priority_map = {
            0: "Low",
            1: "Normal",
            2: "High",
            3: "Critical"
        }
        
        return {
            'probabilities': probabilities.tolist(),
            'predicted_class': predicted_class,
            'priority_level': priority_map[predicted_class],
            'confidence': confidence
        }
    
    # =====================================================================
    # FUNCTION 11: train
    # =====================================================================
    def train(self, X_train, y_train, epochs=200, verbose=False):
        """
        Train the MLP using backpropagation.
        
        Uses Cross-Entropy loss for multi-class classification.
        
        Args:
            X_train (np.ndarray): Training features, shape (n_samples, 6)
            y_train (np.ndarray): Training labels (integers 0-3), 
                                  shape (n_samples,)
            epochs (int): Number of training iterations
            verbose (bool): Whether to print progress
        
        Returns:
            dict: Training history with loss per epoch
        
        Example:
            >>> X = np.array([[1,3,1,8,4.5,3], [0,0,0,2,5,0]])
            >>> y = np.array([3, 0])
            >>> history = mlp.train(X, y, epochs=100)
        """
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=int)
        
        n_samples = X_train.shape[0]
        self.training_history = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for i in range(n_samples):
                x = X_train[i].reshape(1, -1)
                y_true = y_train[i]
                
                # One-hot encode label
                y_onehot = np.zeros(self.output_size)
                y_onehot[y_true] = 1.0
                
                # Forward pass
                result = self.forward_pass(x)
                y_pred = result['output'][0]
                
                # Cross-entropy loss
                loss = -np.sum(y_onehot * np.log(y_pred + 1e-8))
                total_loss += loss
                
                # Backpropagation
                # Output layer error
                dz3 = y_pred - y_onehot
                
                # Layer 2 error
                da2 = np.dot(dz3.reshape(1, -1), self.W3.T)
                dz2 = da2 * self.sigmoid_derivative(self.a2)
                
                # Layer 1 error
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self.sigmoid_derivative(self.a1)
                
                # Update weights (gradient descent)
                self.W3 -= self.learning_rate * np.dot(self.a2.T, dz3.reshape(1, -1))
                self.b3 -= self.learning_rate * dz3
                
                self.W2 -= self.learning_rate * np.dot(self.a1.T, dz2)
                self.b2 -= self.learning_rate * dz2[0]
                
                self.W1 -= self.learning_rate * np.dot(x.T, dz1)
                self.b1 -= self.learning_rate * dz1[0]
            
            avg_loss = total_loss / n_samples
            self.training_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return {
            'epochs': epochs,
            'final_loss': self.training_history[-1],
            'loss_history': self.training_history
        }


# =============================================================================
# ANN PRIORITY MODULE: Main Interface
# =============================================================================

class ANNPriorityModule:
    """
    Main ANN Priority Prediction Module interface.
    
    This class provides a unified interface for both the Binary Classifier
    and Multi-Class MLP. It handles data loading, model training, and
    prediction for traffic requests.
    
    Attributes:
        binary_classifier (BinaryPriorityClassifier): Binary urgency classifier
        multiclass_mlp (MultiClassPriorityMLP): Multi-class priority MLP
        training_data (dict): Loaded training data
        is_trained (bool): Whether models have been trained
    """
    
    # Priority level mappings
    PRIORITY_LABELS = {
        0: "Low",
        1: "Normal",
        2: "High",
        3: "Critical"
    }
    
    def __init__(self, binary_lr=0.1, mlp_lr=0.1, 
                 hidden1=8, hidden2=6):
        """
        Initialize the ANN Priority Module.
        
        Args:
            binary_lr (float): Learning rate for binary classifier
            mlp_lr (float): Learning rate for MLP
            hidden1 (int): Hidden layer 1 size for MLP
            hidden2 (int): Hidden layer 2 size for MLP
        """
        self.binary_classifier = BinaryPriorityClassifier(learning_rate=binary_lr)
        self.multiclass_mlp = MultiClassPriorityMLP(
            hidden1_size=hidden1,
            hidden2_size=hidden2,
            learning_rate=mlp_lr
        )
        self.training_data = None
        self.is_trained = False
    
    # =====================================================================
    # FUNCTION 12: load_training_data
    # =====================================================================
    def load_training_data(self, file_path=None):
        """
        Load training data from JSON file.
        
        Args:
            file_path (str): Path to training data JSON.
                           Defaults to config.ANN_TRAINING_DATA_PATH
        
        Returns:
            dict: Loaded training data
        
        Raises:
            DataLoadError: If file cannot be loaded
        """
        if file_path is None:
            file_path = ANN_TRAINING_DATA_PATH
        
        if not os.path.exists(file_path):
            raise DataLoadError(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.training_data = json.load(f)
            return self.training_data
        except Exception as e:
            raise DataLoadError(f"{file_path} ({str(e)})")
    
    # =====================================================================
    # FUNCTION 13: prepare_training_data
    # =====================================================================
    def prepare_training_data(self, data_type='binary'):
        """
        Prepare training data in NumPy format.
        
        Args:
            data_type (str): 'binary' or 'multiclass'
        
        Returns:
            tuple: (X_train, y_train) as NumPy arrays
        
        Raises:
            InvalidRequestError: If training data not loaded
        """
        if self.training_data is None:
            raise InvalidRequestError(
                'training_data',
                "Training data not loaded. Call load_training_data() first."
            )
        
        if data_type == 'binary':
            data = self.training_data.get('binary_training_data', [])
        else:
            data = self.training_data.get('multiclass_training_data', [])
        
        if not data:
            raise InvalidRequestError(
                'training_data',
                f"No {data_type} training data found in file"
            )
        
        X = np.array([sample['features'] for sample in data])
        y = np.array([sample['label'] for sample in data])
        
        return X, y
    
    # =====================================================================
    # FUNCTION 14: train_models
    # =====================================================================
    def train_models(self, binary_epochs=100, mlp_epochs=200, verbose=False):
        """
        Train both neural network models.
        
        Args:
            binary_epochs (int): Epochs for binary classifier
            mlp_epochs (int): Epochs for multi-class MLP
            verbose (bool): Print training progress
        
        Returns:
            dict: Training results for both models
        
        Example:
            >>> ann_module = ANNPriorityModule()
            >>> ann_module.load_training_data()
            >>> results = ann_module.train_models(verbose=True)
        """
        # Load training data if not already loaded
        if self.training_data is None:
            self.load_training_data()
        
        results = {}
        
        # Train Binary Classifier
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING BINARY CLASSIFIER (Perceptron)")
            print("=" * 60)
        
        X_binary, y_binary = self.prepare_training_data('binary')
        
        if verbose:
            print(f"Training samples: {len(X_binary)}")
            print(f"Features: {ANN_INPUT_FEATURES}")
            print(f"Output: Binary (Urgent/Not Urgent)")
            print(f"Epochs: {binary_epochs}")
            print("-" * 60)
        
        binary_result = self.binary_classifier.train(
            X_binary, y_binary, epochs=binary_epochs, verbose=verbose
        )
        results['binary'] = binary_result
        
        if verbose:
            print(f"Final Loss: {binary_result['final_loss']:.6f}")
        
        # Train Multi-Class MLP
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING MULTI-CLASS MLP (2 Hidden Layers)")
            print("=" * 60)
            print(f"Architecture: {ANN_INPUT_FEATURES} -> 8 -> 6 -> {ANN_MULTICLASS_OUTPUT}")
            print(f"Training samples: {len(X_binary)}")
            print(f"Classes: Low, Normal, High, Critical")
            print(f"Epochs: {mlp_epochs}")
            print("-" * 60)
        
        X_multi, y_multi = self.prepare_training_data('multiclass')
        
        mlp_result = self.multiclass_mlp.train(
            X_multi, y_multi, epochs=mlp_epochs, verbose=verbose
        )
        results['multiclass'] = mlp_result
        
        if verbose:
            print(f"Final Loss: {mlp_result['final_loss']:.6f}")
        
        self.is_trained = True
        
        return results
    
    # =====================================================================
    # FUNCTION 15: predict_priority_binary
    # =====================================================================
    def predict_priority_binary(self, feature_vector):
        """
        Predict binary urgency using the single-neuron classifier.
        
        Args:
            feature_vector (list or np.ndarray): 6-element feature vector
        
        Returns:
            dict: Binary prediction result
        
        Example:
            >>> result = ann_module.predict_priority_binary([1,3,1,8,4.5,3])
            >>> print(result['label'])
            'Urgent'
        """
        if not self.is_trained:
            raise InvalidRequestError(
                'model_state',
                "Models not trained. Call train_models() first."
            )
        
        return self.binary_classifier.predict(feature_vector)
    
    # =====================================================================
    # FUNCTION 16: predict_priority_multiclass
    # =====================================================================
    def predict_priority_multiclass(self, feature_vector):
        """
        Predict multi-class priority using the 2-layer MLP.
        
        Args:
            feature_vector (list or np.ndarray): 6-element feature vector
        
        Returns:
            dict: Multi-class prediction result
        
        Example:
            >>> result = ann_module.predict_priority_multiclass([1,3,1,8,4.5,3])
            >>> print(result['priority_level'])
            'Critical'
        """
        if not self.is_trained:
            raise InvalidRequestError(
                'model_state',
                "Models not trained. Call train_models() first."
            )
        
        return self.multiclass_mlp.predict(feature_vector)
    
    # =====================================================================
    # FUNCTION 17: predict_priority (Unified Interface)
    # =====================================================================
    def predict_priority(self, feature_vector, mode='multiclass'):
        """
        Unified priority prediction interface.
        
        This is the main function used by the system to get priority
        predictions. It supports both binary and multiclass modes.
        
        Args:
            feature_vector (list or np.ndarray): 6-element feature vector
                [vehicle_type, severity, time_sensitivity, 
                 traffic_density, distance, priority_claim]
            mode (str): 'binary' or 'multiclass'
        
        Returns:
            dict: Priority prediction result
        
        Raises:
            InvalidValueError: If mode is invalid
        
        Example:
            >>> features = [1, 3, 1, 8, 4.5, 3]  # Ambulance, High severity, etc.
            >>> result = ann_module.predict_priority(features, 'multiclass')
            >>> print(result)
            {
                'mode': 'multiclass',
                'priority_level': 'Critical',
                'confidence': 0.85,
                ...
            }
        """
        if mode not in ['binary', 'multiclass']:
            raise InvalidValueError(
                'mode',
                mode,
                "'binary' or 'multiclass'"
            )
        
        if mode == 'binary':
            result = self.predict_priority_binary(feature_vector)
            return {
                'mode': 'binary',
                'is_urgent': result['prediction'] == 1,
                'probability': result['probability'],
                'label': result['label']
            }
        else:
            result = self.predict_priority_multiclass(feature_vector)
            return {
                'mode': 'multiclass',
                'priority_level': result['priority_level'],
                'priority_class': result['predicted_class'],
                'confidence': result['confidence'],
                'all_probabilities': result['probabilities']
            }
    
    # =====================================================================
    # FUNCTION 18: evaluate_models
    # =====================================================================
    def evaluate_models(self):
        """
        Evaluate trained models on training data.
        
        Computes accuracy for both binary and multiclass models.
        
        Returns:
            dict: Evaluation metrics
        
        Raises:
            InvalidRequestError: If models not trained
        """
        if not self.is_trained:
            raise InvalidRequestError(
                'model_state',
                "Models not trained. Call train_models() first."
            )
        
        results = {}
        
        # Evaluate Binary Classifier
        X_binary, y_binary = self.prepare_training_data('binary')
        correct = 0
        for i in range(len(X_binary)):
            pred = self.binary_classifier.predict(X_binary[i])
            if pred['prediction'] == int(y_binary[i]):
                correct += 1
        
        results['binary_accuracy'] = correct / len(X_binary)
        
        # Evaluate Multi-Class MLP
        X_multi, y_multi = self.prepare_training_data('multiclass')
        correct = 0
        for i in range(len(X_multi)):
            pred = self.multiclass_mlp.predict(X_multi[i])
            if pred['predicted_class'] == int(y_multi[i]):
                correct += 1
        
        results['multiclass_accuracy'] = correct / len(X_multi)
        
        return results
    
    # =====================================================================
    # FUNCTION 19: get_model_architecture
    # =====================================================================
    def get_model_architecture(self):
        """
        Get architecture details of both neural networks.
        
        Returns:
            dict: Architecture information for display
        """
        return {
            'binary_classifier': {
                'type': 'Single Neuron (Perceptron)',
                'input_features': ANN_INPUT_FEATURES,
                'activation': 'Sigmoid',
                'output': 'Binary {0, 1}',
                'weights_shape': self.binary_classifier.weights.shape,
                'learning_rate': self.binary_classifier.learning_rate
            },
            'multiclass_mlp': {
                'type': 'Multi-Layer Perceptron',
                'input_features': ANN_INPUT_FEATURES,
                'hidden_layer_1': self.multiclass_mlp.hidden1_size,
                'hidden_layer_2': self.multiclass_mlp.hidden2_size,
                'output_classes': ANN_MULTICLASS_OUTPUT,
                'activations': 'Sigmoid (hidden), Softmax (output)',
                'learning_rate': self.multiclass_mlp.learning_rate
            }
        }
    
    # =====================================================================
    # FUNCTION 20: display_architecture
    # =====================================================================
    def display_architecture(self):
        """
        Display formatted architecture information.
        
        Returns:
            str: Formatted architecture description
        """
        arch = self.get_model_architecture()
        
        lines = [
            "=" * 60,
            "ANN PRIORITY MODULE - ARCHITECTURE",
            "=" * 60,
            "",
            "BINARY CLASSIFIER (Single Neuron):",
            "-" * 60,
            f"  Type:           {arch['binary_classifier']['type']}",
            f"  Input Features: {arch['binary_classifier']['input_features']}",
            f"  Activation:     {arch['binary_classifier']['activation']}",
            f"  Output:         {arch['binary_classifier']['output']}",
            f"  Learning Rate:  {arch['binary_classifier']['learning_rate']}",
            "",
            "Architecture Diagram:",
            "  [VehicleType] ──┐",
            "  [Severity] ─────┤",
            "  [TimeSens] ─────┤──> [Σ + Activation] ──> [Urgent? {0,1}]",
            "  [TrafficDen] ───┤      (Sigmoid)",
            "  [Distance] ─────┤",
            "  [Priority] ─────┤",
            "  [Bias] ─────────┘",
            "",
            "MULTI-CLASS MLP (2 Hidden Layers):",
            "-" * 60,
            f"  Type:           {arch['multiclass_mlp']['type']}",
            f"  Input:          {arch['multiclass_mlp']['input_features']} features",
            f"  Hidden Layer 1: {arch['multiclass_mlp']['hidden_layer_1']} neurons",
            f"  Hidden Layer 2: {arch['multiclass_mlp']['hidden_layer_2']} neurons",
            f"  Output:         {arch['multiclass_mlp']['output_classes']} classes",
            f"  Activations:    {arch['multiclass_mlp']['activations']}",
            f"  Learning Rate:  {arch['multiclass_mlp']['learning_rate']}",
            "",
            "Architecture Diagram:",
            "  [VehicleType] ──┐",
            "  [Severity] ─────┤",
            "  [TimeSens] ─────┤──> [Hidden Layer 1] ──> [Hidden Layer 2] ──> [Output]",
            "  [TrafficDen] ───┤      (8 neurons)          (6 neurons)         (4 classes)",
            "  [Distance] ─────┤                             ↓                    ↓",
            "  [Priority] ─────┘                         [Low] [Normal] [High] [Critical]",
            "=" * 60
        ]
        
        return '\n'.join(lines)


# =============================================================================
# Standalone testing functionality
# =============================================================================
if __name__ == "__main__":
    """
    Standalone test for the ANN Priority Prediction Module.
    Run this file directly to test module functionality.
    """
    import sys
    import os
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    print("=" * 70)
    print("ANN PRIORITY PREDICTION MODULE - STANDALONE TEST")
    print("=" * 70)
    
    # Create module
    ann_module = ANNPriorityModule(binary_lr=0.1, mlp_lr=0.1)
    
    # Display Architecture
    print(ann_module.display_architecture())
    
    # Load and display training data info
    print("\n" + "=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)
    
    try:
        ann_module.load_training_data()
        binary_data = ann_module.training_data.get('binary_training_data', [])
        multi_data = ann_module.training_data.get('multiclass_training_data', [])
        print(f"Binary training samples: {len(binary_data)}")
        print(f"Multiclass training samples: {len(multi_data)}")
        print("✓ Training data loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load training data: {e}")
        sys.exit(1)
    
    # Train models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)
    
    try:
        results = ann_module.train_models(
            binary_epochs=100,
            mlp_epochs=200,
            verbose=True
        )
        print("\n✓ Models trained successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate models
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    try:
        eval_results = ann_module.evaluate_models()
        print(f"Binary Classifier Accuracy: {eval_results['binary_accuracy']:.2%}")
        print(f"Multi-Class MLP Accuracy:   {eval_results['multiclass_accuracy']:.2%}")
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    # Test Case 1: Emergency Request (Ambulance, High Severity)
    print("\n" + "=" * 70)
    print("TEST CASE 1: Emergency Request (Ambulance, High Severity)")
    print("=" * 70)
    features_1 = [1, 3, 1, 9, 6.0, 3]  # Ambulance, High, TimeSens, Density, Dist, Priority
    
    try:
        binary_result = ann_module.predict_priority(features_1, 'binary')
        multi_result = ann_module.predict_priority(features_1, 'multiclass')
        
        print(f"Input Features: {features_1}")
        print(f"  [Vehicle=Ambulance(1), Severity=High(3), TimeSens=High(1)]")
        print(f"  [TrafficDensity=9, Distance=6.0km, PriorityClaim=Critical(3)]")
        print()
        print(f"Binary Prediction:")
        print(f"  Label:      {binary_result['label']}")
        print(f"  Probability: {binary_result['probability']:.4f}")
        print()
        print(f"Multi-Class Prediction:")
        print(f"  Priority Level: {multi_result['priority_level']}")
        print(f"  Confidence:     {multi_result['confidence']:.4f}")
        print(f"  All Probs:      {[f'{p:.4f}' for p in multi_result['all_probabilities']]}")
        print("\n✓ Test Case 1 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 1 FAILED: {e}")
    
    # Test Case 2: Civilian Request (Low Severity)
    print("\n" + "=" * 70)
    print("TEST CASE 2: Civilian Request (Low Severity)")
    print("=" * 70)
    features_2 = [0, 1, 0, 3, 4.0, 1]  # Civilian, Low, Normal, Density, Dist, Priority
    
    try:
        binary_result = ann_module.predict_priority(features_2, 'binary')
        multi_result = ann_module.predict_priority(features_2, 'multiclass')
        
        print(f"Input Features: {features_2}")
        print(f"  [Vehicle=Civilian(0), Severity=Low(1), TimeSens=Normal(0)]")
        print(f"  [TrafficDensity=3, Distance=4.0km, PriorityClaim=Normal(1)]")
        print()
        print(f"Binary Prediction:")
        print(f"  Label:      {binary_result['label']}")
        print(f"  Probability: {binary_result['probability']:.4f}")
        print()
        print(f"Multi-Class Prediction:")
        print(f"  Priority Level: {multi_result['priority_level']}")
        print(f"  Confidence:     {multi_result['confidence']:.4f}")
        print(f"  All Probs:      {[f'{p:.4f}' for p in multi_result['all_probabilities']]}")
        print("\n✓ Test Case 2 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 2 FAILED: {e}")
    
    # Test Case 3: Police Request (Medium Severity)
    print("\n" + "=" * 70)
    print("TEST CASE 3: Police Request (Medium Severity)")
    print("=" * 70)
    features_3 = [3, 2, 1, 6, 3.5, 2]  # Police, Medium, High, Density, Dist, Priority
    
    try:
        binary_result = ann_module.predict_priority(features_3, 'binary')
        multi_result = ann_module.predict_priority(features_3, 'multiclass')
        
        print(f"Input Features: {features_3}")
        print(f"  [Vehicle=Police(3), Severity=Medium(2), TimeSens=High(1)]")
        print(f"  [TrafficDensity=6, Distance=3.5km, PriorityClaim=High(2)]")
        print()
        print(f"Binary Prediction:")
        print(f"  Label:      {binary_result['label']}")
        print(f"  Probability: {binary_result['probability']:.4f}")
        print()
        print(f"Multi-Class Prediction:")
        print(f"  Priority Level: {multi_result['priority_level']}")
        print(f"  Confidence:     {multi_result['confidence']:.4f}")
        print(f"  All Probs:      {[f'{p:.4f}' for p in multi_result['all_probabilities']]}")
        print("\n✓ Test Case 3 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 3 FAILED: {e}")
    
    # Test Case 4: Fire Truck (Critical Scenario)
    print("\n" + "=" * 70)
    print("TEST CASE 4: Fire Truck (Critical Scenario)")
    print("=" * 70)
    features_4 = [2, 3, 1, 10, 8.0, 3]  # Fire, High, High, Max Density, Long Dist, Critical
    
    try:
        binary_result = ann_module.predict_priority(features_4, 'binary')
        multi_result = ann_module.predict_priority(features_4, 'multiclass')
        
        print(f"Input Features: {features_4}")
        print(f"  [Vehicle=Fire(2), Severity=High(3), TimeSens=High(1)]")
        print(f"  [TrafficDensity=10, Distance=8.0km, PriorityClaim=Critical(3)]")
        print()
        print(f"Binary Prediction:")
        print(f"  Label:      {binary_result['label']}")
        print(f"  Probability: {binary_result['probability']:.4f}")
        print()
        print(f"Multi-Class Prediction:")
        print(f"  Priority Level: {multi_result['priority_level']}")
        print(f"  Confidence:     {multi_result['confidence']:.4f}")
        print(f"  All Probs:      {[f'{p:.4f}' for p in multi_result['all_probabilities']]}")
        print("\n✓ Test Case 4 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 4 FAILED: {e}")
    
    # Test Case 5: Invalid Feature Vector
    print("\n" + "=" * 70)
    print("TEST CASE 5: Invalid Feature Vector (Wrong Size)")
    print("=" * 70)
    bad_features = [1, 3, 1]  # Only 3 features instead of 6
    
    try:
        result = ann_module.predict_priority(bad_features, 'multiclass')
        print("✗ Test Case 5 FAILED: Should have raised exception")
    except InvalidValueError as e:
        print(f"✓ Test Case 5 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 5 FAILED: Unexpected error: {e}")
    
    # Test Case 6: Untrained Model Error
    print("\n" + "=" * 70)
    print("TEST CASE 6: Untrained Model Error")
    print("=" * 70)
    untrained_module = ANNPriorityModule()
    
    try:
        result = untrained_module.predict_priority([1, 3, 1, 8, 4.5, 3], 'binary')
        print("✗ Test Case 6 FAILED: Should have raised exception")
    except InvalidRequestError as e:
        print(f"✓ Test Case 6 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 6 FAILED: Unexpected error: {e}")
    
    # Test Case 7: Invalid Mode
    print("\n" + "=" * 70)
    print("TEST CASE 7: Invalid Prediction Mode")
    print("=" * 70)
    
    try:
        result = ann_module.predict_priority([1, 3, 1, 8, 4.5, 3], 'invalid_mode')
        print("✗ Test Case 7 FAILED: Should have raised exception")
    except InvalidValueError as e:
        print(f"✓ Test Case 7 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 7 FAILED: Unexpected error: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)