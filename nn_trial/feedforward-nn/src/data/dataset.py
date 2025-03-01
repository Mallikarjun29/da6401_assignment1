import numpy as np

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def split_data(X, y, test_size=0.2):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    test_set_size = int(num_samples * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return X_train, y_train, X_test, y_test

def normalize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)