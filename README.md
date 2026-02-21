[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/09aG2bkW)
# Lab: Binary Classification from Scratch

**COMP 395 – Deep Learning**

In this lab, you will implement binary classification using gradient descent — the same algorithm that powers deep learning. You'll classify breast cancer tumors as malignant or benign using real medical data.

## Learning Objectives

By completing this lab, you will:

1. Implement the sigmoid activation function
2. Implement a forward pass (prediction)
3. Implement mean squared error loss
4. Derive and implement gradients using the chain rule
5. Train a model using gradient descent
6. Compare your model against an sklearn classifier

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Your Task

Open `binary_classification.py` and replace the `TODO` placeholders with your implementations. There are **10 TODOs** to complete across 4 functions:

| Function | Blanks | Description |
|----------|--------|-------------|
| `sigmoid(z)` | 1 | Sigmoid activation function |
| `forward(x, w, b)` | 2 | Compute prediction for one sample |
| `compute_loss(y, y_hat)` | 1 | Mean squared error loss |
| `compute_gradients(x, y, y_hat)` | 5 | Gradients for weights and bias |

The training loop has **4 additional TODOs** that call your functions.

## Testing Your Code

Run the tests to check your implementation:

```
python -m pytest test_binary_classification.py -v
```

Or run individual test functions:

```
python -m pytest test_binary_classification.py::test_sigmoid -v
```

## Running the Full Training

Once all tests pass, run the complete training:

```
python binary_classification.py
```

You should see:
- Loss decreasing over epochs
- Training accuracy > 95%
- Test accuracy > 93%

## Files

| File | Description |
|------|-------------|
| `binary_classification.py` | **Your code goes here** — fill in the blanks |
| `comparison.py` | **Your code goes here** — model comparison (Part 2) |
| `test_binary_classification.py` | Automated tests (do not modify) |
| `requirements.txt` | Python dependencies |

## The Math

**Forward pass** (one sample):
```
z = w · x + b
ŷ = σ(z) = 1 / (1 + e^(-z))
```

**Loss** (one sample):
```
L = ½(ŷ - y)²
```

**Gradients** (one sample):
```
error = ŷ - y
sigmoid_deriv = ŷ(1 - ŷ)
δ = error × sigmoid_deriv

∂L/∂w = δ × x
∂L/∂b = δ
```

**Update rule**:
```
w = w - α × ∂L/∂w
b = b - α × ∂L/∂b
```

## Hints

- `torch.dot(a, b)` computes the dot product of two vectors
- `torch.exp(x)` computes e^x
- The sigmoid derivative can be computed from the output: `y_hat * (1 - y_hat)`
- Make sure your loss includes the `0.5` factor

## Part 2: Model Comparison

After completing your from-scratch implementation, choose a **different classification model** from [scikit-learn](https://scikit-learn.org/stable/supervised_learning.html) (e.g., Decision Tree, Random Forest, SVM, k-Nearest Neighbors, etc. — but **not** Logistic Regression, since that is essentially what you just built).

In a file called `comparison.py`:

1. **Describe your chosen model** — Write a short paragraph (3–5 sentences) as a comment at the top of the file explaining how the model works and why you chose it.
2. **Implement it** — Train your chosen sklearn model on the same breast cancer dataset (use `load_data()` from `binary_classification.py`).
3. **Compare** — Print the test accuracy of both your from-scratch model and the sklearn model, and write a brief comment (2–3 sentences) discussing which performed better and why that might be.

## Grading

| Component | Points |
|-----------|--------|
| `sigmoid` passes all tests | 12 |
| `forward` passes all tests | 16 |
| `compute_loss` passes all tests | 12 |
| `compute_gradients` passes all tests | 24 |
| Training loop runs and achieves >90% accuracy | 16 |
| Model comparison (description, implementation, and comparison) | 20 |
| **Total** | **100** |

## Submission

Push your completed `binary_classification.py` and `comparison.py` to this repository. GitHub Classroom will automatically run the tests and report your score.

---


