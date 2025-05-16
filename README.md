# ek
K-means clustering with weights

To install:	```pip install ek```

## Overview
The `ek` package provides an implementation of the K-means clustering algorithm that incorporates sample weights. This is particularly useful in scenarios where certain data points are of more significance than others and should have a greater influence on the formation of clusters.

## Main Features
- **Weighted K-means Clustering**: Allows clustering with weighted data points, which can be crucial for datasets where some instances are more important than others.
- **Compatibility with Scikit-learn**: The implementation is designed to be compatible with Scikit-learn's clustering framework, making it easy to integrate with existing codebases that use Scikit-learn for machine learning tasks.
- **Support for Sparse Data**: Efficiently handles sparse matrices, which is beneficial for high-dimensional data.
- **Custom Initialization Methods**: Supports various methods for initializing cluster centers, including a weighted version of the k-means++ initialization.

## Installation
To install the package, use the following pip command:
```bash
pip install ek
```

## Usage

### Basic Example
Here is a simple example of how to use the `ek` package to perform weighted K-means clustering:

```python
import numpy as np
from ek import KMeansWeighted

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Weights for each data point
weights = np.array([1, 2, 1, 1, 1, 2])

# Number of clusters
n_clusters = 2

# Create a KMeansWeighted instance
kmeans = KMeansWeighted(n_clusters=n_clusters)

# Fit the model
kmeans.fit(X, weights)

# Get cluster labels
labels = kmeans.labels_

# Print the labels
print(labels)
```

### Advanced Usage
For more advanced usage, you can specify additional parameters such as `init` for the initialization method, `max_iter` for the maximum number of iterations, and `tol` for the convergence tolerance.

```python
kmeans = KMeansWeighted(n_clusters=3, init='random', max_iter=100, tol=1e-4)
kmeans.fit(X, weights)
```

## Documentation

### Classes and Functions

#### `KMeansWeighted`
A class for K-means clustering with weights.

- **Parameters**:
  - `n_clusters`: Number of clusters.
  - `init`: Method for initialization (`'k-means++_with_weights'`, `'random'` or an ndarray).
  - `max_iter`: Maximum number of iterations.
  - `tol`: Tolerance for convergence.
  - `precompute_distances`: Whether to precompute distances (`'auto'`, `True`, `False`).
  - `verbose`: Verbosity mode.
  - `random_state`: Seed or numpy.RandomState instance.
  - `copy_x`: If True, input data is copied.
  - `n_jobs`: Number of parallel jobs to run.

- **Methods**:
  - `fit(X, weights)`: Compute K-means clustering.
  - `fit_predict(X, weights)`: Compute clustering and predict cluster indices.
  - `fit_transform(X, weights)`: Compute clustering and transform X to cluster-distance space.
  - `transform(X)`: Transform X to cluster-distance space.
  - `predict(X)`: Predict the closest cluster each sample in X belongs to.
  - `score(X)`: Opposite of the value of X on the K-means objective.

This package is designed to be easy to use while providing the flexibility needed for more complex clustering tasks. The implementation is optimized for performance and can handle large datasets efficiently.