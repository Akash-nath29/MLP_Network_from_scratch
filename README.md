# Handwritten Digit Classifier

This project implements a handwritten digit classifier using a Multi-Layer Perceptron (MLP) neural network. The model is trained on the MNIST dataset and can predict digits from 0 to 9.

## Project Structure

```
MLP_Network_from_scratch/
│
├── dataloader/
│   └── dataloader.py
├── functions/
│   └── functions.py
├── neuralnet/
│   └── model.py
├── data/
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── train.py
├── model.ipynb
└── .gitignore
```

## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/Akash-nath29/MLP_Network_from_scratch.git
    cd MLP_Network_from_scratch
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the MNIST dataset and place the CSV files in the `data/` directory.

## Usage

### Training the Model

To train the model, run the `train.py` script:
```sh
python train.py
```
This will train the MLP neural network on the MNIST dataset and save the trained model to `model.json`.

### Evaluating the Model

You can evaluate the model and visualize the predictions using the Jupyter notebook `model.ipynb`. Open the notebook and run the cells to see the results.

### Plotting Predictions

The `plot_predictions` function in `train.py` can be used to visualize the model's predictions on the test set.

## Functions and Classes

### Data Loader

- `load_data(train_csv, test_csv)`: Loads and normalizes the training and testing data from CSV files.

### Functions

- `one_hot_encoding(y, num_classes)`: Converts labels to one-hot encoded vectors.
- `sigmoid(z)`: Sigmoid activation function.
- `sigmoid_derivative(z)`: Derivative of the sigmoid function.
- `softmax(z)`: Softmax activation function.
- `cross_entropy_loss(y_true, y_pred)`: Computes the cross-entropy loss.
- `accuracy(y_true, y_pred)`: Computes the accuracy of predictions.

### Neural Network Model

- `MLPNetwork`: Class implementing the MLP neural network with methods for forward pass, backward pass, training, prediction, and loss plotting.

## License

This project is licensed under the MIT License.