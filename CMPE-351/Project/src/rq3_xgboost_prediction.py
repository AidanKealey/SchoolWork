import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score
import time  # Import the time module

# Load data from CSV file
file_path = 'data/review_sentiments_stars.csv'  # Update with the path to your actual data file
df = pd.read_csv(file_path)

# Splitting the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define features and target
features = ['Price', 'Service', 'Ambiance', 'Food']
target = 'Stars'

# Prepare XGBoost DMatrices, handling NaN values natively
dtrain = xgb.DMatrix(train_df[features], label=train_df[target])
dtest = xgb.DMatrix(test_df[features], label=test_df[target])

# Specify model training parameters
params = {
    'max_depth': 3,  # Maximum depth of a tree
    'eta': 0.1,  # Learning rate
    'objective': 'reg:squarederror',  # Loss function for regression
    'eval_metric': 'rmse'
}
num_rounds = 100  # Number of boosting rounds

# Start the training timer
start_time = time.time()

# Training the model
bst = xgb.train(params, dtrain, num_rounds)

# End the training timer
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.4f} seconds")  # Output the training time

# Predicting the test set
test_predictions = np.rint(bst.predict(dtest)).astype(int)

# Evaluation
actuals = np.rint(test_df[target].values).astype(int)
cm = confusion_matrix(actuals, test_predictions)
accuracy = accuracy_score(actuals, test_predictions)
print("Accuracy:", accuracy)
print("Non-normalized Confusion Matrix:")
print(cm)

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Feature Importance Extraction
# Retrieve raw importances and normalize them
feature_importance_raw = bst.get_score(importance_type='weight')
# Sum all feature importances to normalize
total_importance = sum(feature_importance_raw.values())
# Normalize and extract in the order of the features array
feature_importances = [feature_importance_raw.get(f, 0.) / total_importance for f in features]
print(f"Feature Importances: {feature_importances}")

