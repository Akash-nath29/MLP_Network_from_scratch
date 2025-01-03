from neuralnet import MLPNetwork
from dataloader import load_data
from functions import one_hot_encoding
import json
import matplotlib.pyplot as plt
import numpy as np

# Test Purpose
def plot_predictions(X_test, y_test, y_pred, num_samples=10):
    plt.figure(figsize=(10, 10))
    
    for i in range(num_samples):
        idx = np.random.randint(0, X_test.shape[0])
        
        image = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        predicted_label = y_pred[idx]
        
        plt.subplot(5, 5, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {true_label}, Pred: {predicted_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

X_train, y_train, X_test, y_test = load_data('data/mnist_train.csv', 'data/mnist_test.csv')

y_train_encoded = one_hot_encoding(y_train, 10)
y_test_encoded = one_hot_encoding(y_test, 10)

input_size = 784 
hidden_size = 64 
output_size = 10 
learning_rate = 0.1

nn = MLPNetwork(input_size, hidden_size, output_size, learning_rate)
nn.train(X_train, y_train_encoded, epochs=5000)

nn.plot_loss()

y_test_pred = nn.predict(X_test)

plot_predictions(X_test, y_test, y_test_pred, num_samples=10)

with open('model.json', 'w') as f:
    json.dump(nn.model_to_json(), f)