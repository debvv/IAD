import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных
data = pd.read_csv(r'C:\Users\Scofield\Desktop\B7_adult.csv' , delimiter=';')

# Преобразование категориальных переменных в числовые
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Выделение целевой переменной и признаков
X = data.drop('income', axis=1)
y = data['income']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Интерфейс Streamlit
st.title("Классификация людей по доходам")
st.write("Точность модели: {:.2f}%".format(accuracy * 100))
st.write("Отчет о классификации:")
st.text(report)

# Форма для предсказаний
st.write("Сделайте предсказание:")
input_data = {}
for column in X.columns:
    input_data[column] = st.number_input(f'{column}', value=0)

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)
st.write("Прогнозируемый доход:", "Больше 50К" if prediction[0] == 1 else "Меньше 50К")
