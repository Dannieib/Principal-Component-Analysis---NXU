# PCA-Based Analysis of Cancer Dataset

## Overview  
This project applies Principal Component Analysis (PCA) to reduce the dimensionality of the breast cancer dataset and optionally implements logistic regression for classification.

## Features  
- Load and preprocess the breast cancer dataset from `sklearn.datasets`.
- Standardize the dataset for improved PCA performance.
- Apply PCA to reduce dimensionality to **2 principal components**.
- Visualize PCA results using a **scatter plot**.
- (Bonus) Train and evaluate **logistic regression** using PCA-transformed features.

## Installation & Usage  
### **Prerequisites**  
Ensure you have Python installed along with the necessary dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### **Running the Script**  
Run the Python script using the following command:

```bash
python pca_analysis.py
```

## Dependencies  
- Python 3.x  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  -> (sklearn)

## Output  
- **Scatter Plot**: Visualizing the dataset projected onto two principal components.
- **Logistic Regression Accuracy**: Performance evaluation based on PCA-transformed features.

## Contribution  
Feel free to fork this repository, submit issues, and contribute via pull requests!

## License  
This project is licensed under the MIT License.

