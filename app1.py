# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
#data = pd.read_csv('/mnt/data/B7_adult.csv')
data = pd.read_csv(r'C:\Users\Scofield\Desktop\B7_adult.csv' , delimiter=';')
# Display the first few rows of the dataset
st.write("## Dataset")
st.write(data.head())

# Preprocessing the dataset
st.write("## Preprocessing")

# Drop rows with missing values
data.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split the data into features and target
X = data.drop('income', axis=1)
y = data['income']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
st.write("## Model Evaluation")
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Feature importance
st.write("### Feature Importance")
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
feature_importance.nlargest(10).plot(kind='barh', ax=ax)
st.pyplot(fig)

# Histograms of the features
st.write("## Feature Distributions")
selected_columns = st.multiselect('Select columns to visualize', X.columns.tolist(), default=X.columns.tolist()[:3])
for column in selected_columns:
    fig, ax = plt.subplots()
    sns.histplot(data[column], kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    st.pyplot(fig)

# Deploy the model
st.write("## Deployment")

def predict(input_data):
    # Transform input data
    input_data = pd.DataFrame([input_data])
    for column in label_encoders:
        input_data[column] = label_encoders[column].transform(input_data[column])
    input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    return label_encoders['income'].inverse_transform(prediction)[0]

# Input form
st.write("### Predict Income")
input_data = {}
for column in X.columns:
    input_data[column] = st.text_input(f'Enter {column}')
    
if st.button("Predict"):
    prediction = predict(input_data)
    st.write(f'The predicted income is: {prediction}')
