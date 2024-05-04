import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load data from CSV file
file_path = 'data/review_sentiments_stars.csv'  # Update with the path to your actual data file
df = pd.read_csv(file_path)

# Handling missing values using median imputation (a common choice for classification)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df_imputed = imputer.fit_transform(df.drop('Stars', axis=1))
df_imputed = pd.DataFrame(df_imputed, columns=df.columns[1:])  # without the 'Stars' column

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_imputed, df['Stars'], test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees in the forest

# Start timer
start_time = time.time()

# Train the model
random_forest_model.fit(X_train, y_train)

# End timer
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.4f} seconds")

# Predict on the test set
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')  # Printing the accuracy as a decimal
print('Non-normalized Confusion Matrix:')
print(conf_matrix)

# Visualize the non-normalized confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Non-normalized Confusion Matrix')
plt.show()
plt.savefig()

# Normalize the confusion matrix by the number of instances in each actual class
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Visualize the normalized confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Normalized Confusion Matrix')
plt.show()

# Optionally, print the feature importances
feature_importances = random_forest_model.feature_importances_
print(f'Feature Importances: {feature_importances}')
