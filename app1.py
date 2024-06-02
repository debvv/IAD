# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Устанавливаем фоновое изображение
page_bg_img = '''
<style>
.stApp {
  background-image: url("background.png");
  background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)



# Load dataset
st.title("Clasificarea persoanelor dupa venit")
st.write("Devs: Eugeniu Casian&Virgiliu Plesca/SD-231M .")


# Function to load and preprocess the dataset
def load_data(file):
    data = pd.read_csv(file, delimiter=';')
    data.dropna(inplace=True)
    
    # Encode categorical variables except the target variable 'income'
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column != 'income':
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])
        else:
            # Encode the target variable separately
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])
    
    return data, label_encoders

# Streamlit UI
st.write("## Upload your dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data, label_encoders = load_data(uploaded_file)
    
    # Display the first few rows of the dataset
    st.write("## Dataset")
    st.write(data.head())

    # Preprocessing the dataset
    st.write("## Preprocessing")

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
        # Check if all necessary columns are in the input data
        for column in X.columns:
            if column not in input_data or input_data[column] == '':
                st.error(f"Missing input for {column}")
                return None
        
        # Transform input data
        input_df = pd.DataFrame([input_data])
        for column in label_encoders:
            if column in input_df.columns:
                try:
                    input_df[column] = label_encoders[column].transform(input_df[column])
                except ValueError as e:
                    st.error(f"Invalid value for {column}: {input_df[column][0]}")
                    return None
        input_df = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_df)
        return label_encoders['income'].inverse_transform(prediction)[0]

    # Input form
    st.write("### Predict Income")
    input_data = {}
    for column in X.columns:
        input_data[column] = st.text_input(f'Enter {column}')
        
    if st.button("Predict"):
        prediction = predict(input_data)
        if prediction is not None:
            st.write(f'The predicted income is: {prediction}')
