def normalize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def calculate_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def calculate_accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    return np.mean(predictions == y_true)