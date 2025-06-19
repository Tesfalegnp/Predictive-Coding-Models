
#  Predictive Coding Model for MNIST Classification

A minimal implementation of a **predictive coding model** for image classification on the **MNIST dataset** using NumPy and TensorFlow/Keras.

This project demonstrates how **prediction errors**, **state updates**, and **weight updates** can be used iteratively to train a hierarchical neural network — inspired by both neuroscience and machine learning principles.

---

##  Overview

- **Model Type**: Predictive Coding (Biologically Inspired)
- **Dataset**: MNIST (handwritten digits)
- **Architecture**: 3-layer feedforward with lateral connections
- **Activation Function**: Sigmoid
- **Training Method**: Iterative state refinement + prediction error minimization
- **Output**: Digit classification (0–9)

---

##  Key Features

- Layer-wise predictions and prediction errors
- Internal state updates via top-down error propagation
- Weight updates based on prediction gradients
- Lateral connections between same-layer features
- Training loop with accuracy tracking
- Evaluation with test set performance

---

##  Requirements

To run this code, ensure you have the following installed:

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

---

## 📁 Project Structure

```
predictive_coding_mnist/
├── README.md               # This file
└──  predictive_coding.ipynb    # Main implementation script
```


---

## ⚙️ How It Works

### 1. **Data Preprocessing**
- Loads MNIST dataset
- Flattens images into 784-dimensional vectors
- Normalizes pixel values to [0, 1]
- One-hot encodes labels

### 2. **Model Architecture**
- Input Layer: 784 units
- Hidden Layer: 256 units
- Output Layer: 10 units (for digit classes)
- Uses **sigmoid activation** and **He weight initialization**

### 3. **Core Mechanisms**
- **Forward Pass**: Computes layer-wise predictions
- **Prediction Errors**: Calculated as difference between predicted and actual activations
- **State Updates**: Refines internal states to reduce errors
- **Weight Updates**: Adjusts connection strengths using gradient descent

### 4. **Lateral Connections**
- Allows interaction between same-layer features
- Improves representation through feature correlation

---

## 🏃‍♂️ How to Run

### Option 1: Python Script

Save the code below in a file named `predictive_coding.py` and run:

```bash
python predictive_coding.ipynb
```

### Option 2: Jupyter Notebook

Copy and paste the code block-by-block into notebook cells.

---

##  Sample Output

```
Epoch 1/5, Loss: 0.2145, Accuracy: 0.8231
Epoch 2/5, Loss: 0.1723, Accuracy: 0.8510
Epoch 3/5, Loss: 0.1498, Accuracy: 0.8703
Epoch 4/5, Loss: 0.1321, Accuracy: 0.8857
Epoch 5/5, Loss: 0.1184, Accuracy: 0.8942
...
Test Accuracy: 0.8734
```

---

## 📈 Expected Performance

| Metric         | Value       |
|----------------|-------------|
| Epochs         | 5           |
| Batch Size     | 32          |
| Learning Rate  | 0.001       |
| Train Accuracy | ~89%        |
| Test Accuracy  | ~87%        |

> With more training and tuning, accuracy can be improved further.

---

## 🛠️ Customization Ideas

- Add support for ReLU/Swish activations
- Increase number of hidden layers
- Use GPU acceleration (e.g., with CuPy or JAX)
- Visualize learned weights and reconstructions
- Save/load trained models

---

## 📄 License

MIT License – see LICENSE file

---

## 👥 Author

Tesfalegn Petros
📧 peterhope935@gmail.com  
🔗 https://github.com/Tesfalegnp/predictive-coding-models
