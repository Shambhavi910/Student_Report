# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 21:41:39 2025

@author: SHAMBHAVI ROY
"""

import numpy as np
import pickle
import  streamlit as st

#loading the trained model
loaded_model = pickle.load(open("C:/Users/SHAMBHAVI ROY/OneDrive/Desktop/Student's Report/trained_model.sav",'rb'))

loaded_rf = loaded_model['rf_regressor']
loaded_clf = loaded_model['classifier']
loaded_sc = loaded_model['scaler_clf']
loaded_sr = loaded_model['scaler_rf']

#creating a function for prediction

def report_prediction(input_data):
    
    input_np = np.array(input_data).reshape(1, -1)



    # Score prediction
    input_rf_scaled = loaded_sr.transform(input_np)
    
    predicted_score_rf = loaded_rf.predict(input_rf_scaled)[0]
    
    # Risk prediction
    input_clf_scaled = loaded_sc.transform(input_np)
    
    risk = loaded_clf.predict(input_clf_scaled)[0]
    
    
    print(f"Predicted Score: {predicted_score_rf:.2f}")
    
    if risk == 0:
        print("🔧Early Warning: Student needs improvement.")
    else:
        print("✅ Student is on the safe side.")
        
        
        
def main():
    
    #giving a title
    st.title('Report Prediction Web App')
    
    #getting the input_data from the user
    
    weekly_self_study_hours = st.text_input('Weekly self-study hours (0–40)')
    attendance_percentage = st.text_input('Attendance percentage (50–100)')
    class_participation = st.text_input('Participation in Class(0-10)')
    
    #code for score prediction
    Score = ''
    
    #code for risk prediction
    Prediction=''
    
    #creating a button for score prediction
    
    if st.button('Predicted Score'):
        Score = report_prediction([float(weekly_self_study_hours),
                                        float(attendance_percentage),
                                        float(class_participation),
                                        ])
        
        st.success(Score)
    #creating a button for risk prediction
    
    if st.button('Score Analysis'):
        Prediction = report_prediction([float(weekly_self_study_hours),
                                        float(attendance_percentage),
                                        float(class_participation),
                                        ])
        
        st.success(Prediction)
        
        
    if __name__ == '__main__':
        main()
