import pandas as pd

def load_data(train_csv, test_csv):
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)
    
    X_train = train_data.drop(columns=['label']).values / 255.0
    y_train = train_data['label'].values
    
    X_test = test_data.drop(columns=['label']).values / 255.0
    y_test = test_data['label'].values
    
    return X_train, y_train, X_test, y_test