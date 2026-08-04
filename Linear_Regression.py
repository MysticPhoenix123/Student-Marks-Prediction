import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Student Marks Prediction", layout="wide")

st.title("Student Marks Prediction")
st.header("By Ilina K Srinivasulu")
st.subheader("Dataset")
Name = st.text_input("Enter name: ")
Age = st.number_input("Enter age: ", step = 1)
st.write(Name, Age)

df = pd.read_csv("Linear_Reg - Linear_Reg.csv")
st.dataframe(df)

X = df[['Hours','Sleep','Attendance']]
y = df['Marks']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

prediction = model.predict(X_test)
st.write(y_test)
st.write(prediction)

score = r2_score(y_test,prediction)

st.write("R² Score: ", round(score,2))

st.header("Predict Student Marks")

hours = st.number_input("Study Hours",1,15)

sleep = st.number_input("Sleep Hours",4,10)

attendance = st.number_input("Attendance (%)",50,100)

if st.button("Predict"):

    result = model.predict([[hours,sleep,attendance]])

    st.success(f"Predicted Marks : {result[0]:.2f}")