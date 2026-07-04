import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.title("🚗 Car Price Prediction Dashboard")

df = pd.read_csv("car data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

encoders = {}
for col in ["Fuel_Type","Selling_type","Transmission","Car_Name"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestRegressor(random_state=42)
model.fit(X_train,y_train)

pred=model.predict(X_test)

st.metric("Model R² Score", f"{r2_score(y_test,pred):.3f}")

st.subheader("Prediction")

car=st.selectbox("Car",encoders["Car_Name"].classes_)
year=st.number_input("Year",2000,2035,2020)
present=st.number_input("Present Price",0.0,100.0,5.0)
kms=st.number_input("Driven Kms",0,500000,10000)
fuel=st.selectbox("Fuel Type",encoders["Fuel_Type"].classes_)
selling=st.selectbox("Seller Type",encoders["Selling_type"].classes_)
trans=st.selectbox("Transmission",encoders["Transmission"].classes_)
owner=st.selectbox("Owner",[0,1,2,3])

if st.button("Predict"):
    row=pd.DataFrame([{
        "Car_Name":encoders["Car_Name"].transform([car])[0],
        "Year":year,
        "Present_Price":present,
        "Driven_kms":kms,
        "Fuel_Type":encoders["Fuel_Type"].transform([fuel])[0],
        "Selling_type":encoders["Selling_type"].transform([selling])[0],
        "Transmission":encoders["Transmission"].transform([trans])[0],
        "Owner":owner
    }])
    price=model.predict(row)[0]
    st.success(f"Estimated Selling Price: ₹ {price:.2f} Lakhs")

st.subheader("Actual vs Predicted")
fig,ax=plt.subplots()
ax.scatter(y_test,pred)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig)

st.caption("Created by: Shiva Prasad Ghale")
