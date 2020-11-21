import pandas as pd 
import streamlit as st
import joblib
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier



@st.cache
def load_data(dataset):
    df=pd.read_csv(dataset)
    return df

gender_label={"Male":0,"Female":1}
driving_license_label={"Yes":1,"No":0}
previous_insured_label={"Yes":1,"No":0}
vehicle_age_label={"< 1 Year":0,"1-2 Year":1,"> 2 Years":2}
vehicle_damage_label={"Yes":1,"No":0}
class_label={"No":0,"Yes":1}

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val==key:
            return value

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val==value:
            return key

def load_prediction_model(model_file):
    loaded_model=joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def main():
    st.title("Health Insurance ML App")
    st.subheader("Built with Streamlit by Ferhat Metin")

    menu=["EDA","Prediction"]

    choices=st.sidebar.selectbox("Select Activities",menu)
    if choices=="EDA":
        st.subheader("EDA")

        data=load_data("healthinsurance.csv")
        st.dataframe(data.head(10))

        if st.checkbox("Show Shape"):
            st.write(data.shape)

        if st.checkbox("Show Summary"):
            st.write(data[["Age","Annual_Premium","Policy_Sales_Channel","Vintage"]].describe().T)
            st.write(data.describe(include="object").T)
            st.write(data.Response.value_counts().to_frame())
    if choices=="Prediction":
        st.subheader("Prediction")

        gender=st.selectbox("Select Gender",tuple(gender_label.keys()))
        age=st.slider("Select Age",20,120)
        driving_license=st.selectbox("Select Driver Licence Condition",tuple(driving_license_label.keys()))
        previous_insured=st.selectbox("Select Previously Insured Condition",tuple(previous_insured_label.keys()))
        vehicle_age=st.selectbox("Select Vehicle Age",tuple(vehicle_age_label.keys()))
        vehicle_damage=st.selectbox("Select Vehicle Damage Condition",tuple(vehicle_damage_label.keys()))
        annual_premium=st.number_input("Select Annual Premium",0.0,600000.0)
        policy_sales_channel=st.number_input("Select Policy Sales Channel",0.0,200.0)
        vintage=st.number_input("Select Vintage",0.0,300.0)


        v_gender=get_value(gender,gender_label)
        v_driving_license=get_value(driving_license,driving_license_label)
        v_previous_insured=get_value(previous_insured,previous_insured_label)
        v_vehicle_age=get_value(vehicle_age,vehicle_age_label)
        v_vehicle_damage=get_value(vehicle_damage,vehicle_damage_label)


        sample_data=[v_gender,age,v_driving_license,v_previous_insured, v_vehicle_age,
                        v_vehicle_damage,annual_premium,policy_sales_channel,vintage]

        if st.button("Evaluate"):
            predictor=load_prediction_model("model.pkl")
            prediction=predictor.predict(np.array(sample_data).reshape(1,-1))

            final_result=get_key(prediction,class_label)
            if final_result=="Yes":
                st.success(final_result)
            else:
                st.error(final_result)
            
    


if __name__=='__main__':
    main()

