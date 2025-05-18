# iris_app.py
# iris_app.py
import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib 
# Load data
iris = load_iris()
# Load and train

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

# Streamlit UI
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter the dimensions of the flower:")

sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
species = iris.target_names[prediction]

st.subheader(f"Predicted Species: **{species.capitalize()}**")
