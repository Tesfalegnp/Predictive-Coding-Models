import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

def preprocess_data(X, y, n_samples=None):
    if n_samples is not None:
        X = X[:n_samples]
        y = y[:n_samples]
    
    X_flat = X.reshape(X.shape[0], -1).astype('float32')
    X_norm = MinMaxScaler().fit_transform(X_flat)
    
    y_onehot = np.zeros((len(y), 10))
    y_onehot[np.arange(len(y)), y] = 1
    
    return X_norm, y_onehot

X_train_prep, y_train_prep = preprocess_data(X_train, y_train, n_samples=10000)
X_test_prep, y_test_prep = preprocess_data(X_test, y_test)

class PredictiveCodingModel:
    def __init__(self, layer_sizes=[784, 256, 10], learning_rate=0.001, n_iterations=5):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
        # Improved weight initialization
        self.weights = []
        for i in range(len(layer_sizes)-1):
            # He initialization (better for ReLU/sigmoid)
            std = np.sqrt(2. / layer_sizes[i])
            self.weights.append(np.random.normal(0, std, (layer_sizes[i+1], layer_sizes[i])))
        
        # Smaller lateral connections
        self.lateral_weights = []
        for size in layer_sizes[1:]:
            self.lateral_weights.append(0.01 * np.random.randn(size, size))
    
    def sigmoid(self, x):
        # Numerically stable sigmoid
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))
    
    def sigmoid_derivative(self, x):
        return np.clip(x * (1 - x), 1e-8, 1)  # Clip to avoid numerical instability
    
    def forward_pass(self, x):
        layer_predictions = []
        current_activity = x
        
        for i in range(len(self.weights)):
            prediction = self.sigmoid(np.dot(self.weights[i], current_activity))
            layer_predictions.append(prediction)
            current_activity = prediction
        
        return layer_predictions
    
    def update_states_and_weights(self, x, y):
        states = [x.copy()]
        predictions = self.forward_pass(x)
        states.extend(predictions)
        states[-1] = y.copy()
        
        errors = [np.zeros_like(s) for s in states]
        
        for _ in range(self.n_iterations):
            # Top-down updates
            for l in range(len(self.weights), 0, -1):
                if l == len(self.weights):
                    errors[l] = states[l] - predictions[l-1]
                else:
                    error_from_above = np.dot(self.weights[l].T, errors[l+1])
                    if l > 0:
                        lateral_error = np.dot(self.lateral_weights[l-1], errors[l])
                        errors[l] = error_from_above + 0.1 * lateral_error - (states[l] - predictions[l-1])
                    else:
                        errors[l] = error_from_above - (states[l] - predictions[l-1])
                
                # Update with momentum and clipping
                state_update = 0.1 * self.learning_rate * np.clip(errors[l], -10, 10)
                if l > 0:
                    state_update *= self.sigmoid_derivative(states[l])
                states[l] += state_update
            
            # Recompute predictions
            for l in range(len(self.weights)):
                predictions[l] = self.sigmoid(np.dot(self.weights[l], states[l]))
        
        # Update weights with gradient clipping
        for l in range(len(self.weights)):
            weight_update = np.outer(
                np.clip(errors[l+1], -5, 5) * self.sigmoid_derivative(states[l+1]), 
                states[l]
            )
            self.weights[l] += self.learning_rate * np.clip(weight_update, -0.1, 0.1)
            
            if l > 0:
                lateral_update = np.outer(
                    np.clip(errors[l], -5, 5), 
                    np.clip(errors[l], -5, 5)
                )
                self.lateral_weights[l-1] += 0.01 * self.learning_rate * lateral_update
        
        return predictions, errors
    
    def train(self, X, y, epochs=5, batch_size=32):
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            epoch_loss = 0
            correct = 0
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                batch_loss = 0
                batch_correct = 0
                
                for j in range(batch_X.shape[0]):
                    predictions, errors = self.update_states_and_weights(batch_X[j], batch_y[j])
                    
                    # Add L2 regularization to prevent NaN
                    reg_loss = 0
                    for w in self.weights:
                        reg_loss += 0.001 * np.sum(w**2)
                    
                    loss = 0.5 * np.sum(errors[-1]**2) + reg_loss
                    batch_loss += loss
                    
                    predicted_class = np.argmax(predictions[-1])
                    true_class = np.argmax(batch_y[j])
                    if predicted_class == true_class:
                        batch_correct += 1
                
                epoch_loss += batch_loss / batch_size
                correct += batch_correct
            
            accuracy = correct / n_samples
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_samples:.4f}, Accuracy: {accuracy:.4f}")
    
    def predict(self, X):
        predictions = []
        for x in X:
            layer_predictions = self.forward_pass(x)
            predictions.append(layer_predictions[-1])
        return np.array(predictions)

# Initialize with smaller learning rate
pc_model = PredictiveCodingModel(layer_sizes=[784, 256, 10], 
                                learning_rate=0.001,  # Reduced from 0.01
                                n_iterations=5)

# Train with more epochs
pc_model.train(X_train_prep, y_train_prep, epochs=20, batch_size=32)

# Evaluate
y_pred_probs = pc_model.predict(X_test_prep)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_prep, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")