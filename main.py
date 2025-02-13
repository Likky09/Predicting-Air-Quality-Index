from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd

# Function to prepare data
def prepare_data(train):
    train1 = train.drop(['AQI Value'], axis=1)
    target = train['AQI Value']
    return train1, target

# Function to train and evaluate RandomForest model
def train_random_forest(train1, target):
    rf_model = RandomForestRegressor()
    rf_model.fit(train1, target)
    score = cross_val_score(rf_model, train1, target, cv=5)
    print(f"RandomForest CV Score: {score.mean()*100:.2f}%")
    return rf_model

# Function to train and evaluate AdaBoost model
def train_ada_boost(train1, target):
    ada_model = AdaBoostRegressor()
    ada_model.fit(train1, target)
    score = cross_val_score(ada_model, train1, target, cv=5)
    print(f"AdaBoost CV Score: {score.mean()*100:.2f}%")
    return ada_model

# Function to make predictions
def make_predictions(model, test_data):
    prediction_result = model.predict(test_data)
    print(f"Prediction Result: {prediction_result}")

def main():
    # Assuming train is already defined
    train = pd.read_csv('path_to_train_file.csv')  # Example: Replace with actual path
    train1, target = prepare_data(train)
    
    # Train and evaluate models
    rf_model = train_random_forest(train1, target)
    ada_model = train_ada_boost(train1, target)
    
    # Make predictions with an example test data
    test_data = [[1, 10, 5, 11, 10, 5]]  # Example: Replace with actual test data
    make_predictions(rf_model, test_data)
    make_predictions(ada_model, test_data)

if __name__ == "__main__":
    main()
