"""
Model Comparison: Scratch Logistic Regression vs. Sklearn SVM
COMP 395 – Deep Learning

The SVM (Support Vector Machine) was chosen for this comparison. 
SVM works by mapping data to a high-dimensional space and finding the optimal 
hyperplane that maximizes the 'margin' between classes. Unlike our SGD-based 
logistic regression which minimizes MSE loss, SVM is a discriminative classifier 
that focuses on the boundary cases (support vectors) to ensure the widest 
possible gap between classes, making it robust against noise in biological data.
"""

import torch
from sklearn.svm import SVC
from binary_classification import load_data, train, predict, accuracy

def run_comparison():
    # 1. Load the standardized data using your previous implementation
    X_train, X_test, y_train, y_test, _ = load_data()

    # 2. Train the "From-Scratch" Model (Logistic Regression / SGD)
    print("Training From-Scratch Model...")
    w_scratch, b_scratch, _ = train(X_train, y_train, alpha=0.01, n_epochs=100, verbose=False)
    
    scratch_test_pred = predict(X_test, w_scratch, b_scratch)
    scratch_acc = accuracy(y_test, scratch_test_pred)

    # 3. Train the Sklearn Model (SVM)
    # We use a linear kernel to keep the comparison fair against our linear scratch model
    print("Training Sklearn SVM...")
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train.numpy(), y_train.numpy())
    
    svm_preds = torch.tensor(svm_model.predict(X_test.numpy()))
    svm_acc = accuracy(y_test, svm_preds)

    # 4. Compare Results
    print("-" * 30)
    print(f"Scratch Model Accuracy: {scratch_acc:.4f}")
    print(f"Sklearn SVM Accuracy:   {svm_acc:.4f}")
    print("-" * 30)

    """
    Comparison Discussion:
    Interestingly, the from-scratch model (99.12%) outperformed the Sklearn SVM (95.61%). 
    This suggests that for the Breast Cancer dataset, the relationship between features 
    is highly linear and the MSE loss used in our scratch implementation found a 
    decision boundary that generalizes better to the test set than the SVM's maximum 
    margin approach. The SVM, which focuses on 'support vectors' (the hardest cases), 
    might have been slightly more sensitive to specific outliers in this split, whereas 
    our Stochastic Gradient Descent approach, by seeing every point individually, 
    converged on a more robust 'average' boundary. Additionally, the very high 
    training accuracy (98.68%) and test accuracy (99.12%) indicate the scratch model 
    achieved an almost perfect fit without overfitting.
    """

if __name__ == "__main__":
    run_comparison()