<h1 align=center>Advanced-Wine-Quality-Classification</h1>

## Introduction
This project is a continuation of a previous project on wine quality classification. In the previous project, we explored basic logistic regression and linear regression models. Now, we will take a deeper dive into wine quality classification using advanced methods such as feature selection, hyperparameter tuning, regularization, and cross-validation to optimize model performance and increase accuracy.

## Step 1: Download the Dataset, Data Preprocessing and Split the Data
Download the "Wine Quality" dataset from the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv).
- Load the dataset.
- Transform the 'quality' column into a binary classification task where '1' represents high quality (score >= 7) and '0' represents low quality.
- Split the dataset into training and testing sets (80% training, 20% testing) using scikit-learn's train_test_split function.

```python
# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
data = pd.read_csv(url, sep=";")

# Data preprocessing
data['quality'] = (data['quality'] >= 7).astype(int)

# Split the data into training and testing sets
X = data.drop('quality', axis=1).values
y = data['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Step 2: Modeling - Linear Regression
Create a linear regression model using TensorFlow to predict wine quality scores.

```python
# Linear Regression Model
model_linear = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(11,)),
    tf.keras.layers.Dense(1)
])

model_linear.compile(optimizer='adam', loss='mean_squared_error')
model_linear.fit(X_train, y_train, epochs=100, verbose=0)
```

## Step 3: Modeling - Logistic Regression
Create a logistic regression model using TensorFlow to classify wines as high quality or not.

```python
# Logistic Regression Model
model_logistic = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(11,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_logistic.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_logistic.fit(X_train, y_train, epochs=100, verbose=0)
```

## Step 4: Hyperparameter Tuning
- Define hyperparameters and the search space.
- Create a pipeline with feature scaling and logistic regression.
- Use GridSearchCV to find the best hyperparameters for logistic regression.

```python
# Define hyperparameters and search space
param_grid = {
    'model__penalty': ['l1', 'l2'],
    'model__C': [0.001, 0.01, 0.1, 1, 10],
}

# Create a pipeline with feature scaling and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_logistic_model = grid_search.best_estimator_
```

## Step 5: Feature Selection
- Select the top k features based on importance scores.
- Train logistic regression with selected features.

```python
# Select the top k features based on importance scores.
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train logistic regression with selected features
logistic_model_selected = LogisticRegression(penalty='l2', C=1.0)
logistic_model_selected.fit(X_train_selected, y_train)
```

## Step 6: Model Evaluation
- Evaluate the logistic regression model with hyperparameter tuning on the test dataset.
- Evaluate the logistic regression model with feature selection on the test dataset.
- Evaluate the linear regression model on the test dataset.

```python
# Model Evaluation
logistic_loss, logistic_accuracy = best_logistic_model.named_steps['model'].score(X_test, y_test), \
                                   best_logistic_model.named_steps['model'].score(X_test, y_test)
logistic_predictions = (best_logistic_model.predict(X_test) >= 0.5).astype(int)

logistic_loss_selected, logistic_accuracy_selected = logistic_model_selected.score(X_test_selected, y_test), \
                                                     logistic_model_selected.score(X_test_selected, y_test)
logistic_predictions_selected = (logistic_model_selected.predict(X_test_selected) >= 0.5).astype(int)

linear_loss = model_linear.evaluate(X_test, y_test), model_linear.evaluate(X_test, y_test)
linear_predictions = (model_linear.predict(X_test) >= 0.5

).astype(int)
```

## Step 7: Visualization - Confusion Matrices
- Visualize the quality distribution.
- Visualize the confusion matrix of the linear regression model.
- Visualize the confusion matrix of the logistic regression model with hyperparameter tuning.
- Visualize the confusion matrix of the logistic regression model with feature selection.

```python
# Data Visualization
plt.figure(figsize=(14, 10))

# Quality Distribution
plt.subplot(3, 2, 1)
plt.title("Quality Distribution")
sns.countplot(data['quality'])
plt.xlabel("Quality")
plt.ylabel("Frequency")

# Confusion Matrix - Linear Regression
plt.subplot(3, 2, 3)
plt.title("Confusion Matrix - Linear Regression")
conf_matrix_linear = confusion_matrix(y_test, linear_predictions)
sns.heatmap(conf_matrix_linear.T, annot=True, fmt='d', cmap='Blues')  # Transpose for vertical view
plt.xlabel("True")
plt.ylabel("Predicted")

# Confusion Matrix - Logistic Regression (Hyperparameter Tuning)
plt.subplot(3, 2, 4)
plt.title("Confusion Matrix - Logistic Regression (Hyperparameter Tuning)")
conf_matrix_logistic = confusion_matrix(y_test, logistic_predictions)
sns.heatmap(conf_matrix_logistic.T, annot=True, fmt='d', cmap='Blues')  # Transpose for vertical view
plt.xlabel("True")
plt.ylabel("Predicted")

# Confusion Matrix - Logistic Regression (Feature Selection)
plt.subplot(3, 2, 5)
plt.title("Confusion Matrix - Logistic Regression (Feature Selection)")
conf_matrix_selected = confusion_matrix(y_test, logistic_predictions_selected)
sns.heatmap(conf_matrix_selected.T, annot=True, fmt='d', cmap='Blues')  # Transpose for vertical view
plt.xlabel("True")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()
```

## Step 8: Classification Reports
- Generate classification reports for the linear regression model.
- Generate classification reports for the logistic regression model with hyperparameter tuning.
- Generate classification reports for the logistic regression model with feature selection.

```python
# Classification Reports
print("\nClassification Report - Linear Regression:")
print(classification_report(y_test, linear_predictions))

print("\nClassification Report - Logistic Regression (Hyperparameter Tuning):")
print(classification_report(y_test, logistic_predictions))

print("\nClassification Report - Logistic Regression (Feature Selection):")
print(classification_report(y_test, logistic_predictions_selected))
```

## Step 9: Conclusion
- Summarize the project, highlighting the optimization techniques used.
- Provide accurate results for each model.
- Discuss the impact of these techniques on model performance.
- Emphasize the importance of selecting the right features and hyperparameters in classification tasks.
