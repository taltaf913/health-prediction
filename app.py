import gradio
import joblib
import numpy as np
import xgboost

# Load your trained model
my_model = joblib.load(filename="xgboost-model.pkl")

# Function for prediction

yes_no_map = {'Yes':1, 'No':0}
gender_map = {'M':1, 'F':0}

def predict_death_event(age=55, anaemia='Yes', creatinine_phosphokinase=1280.25, diabetes='No',
                      ejection_fraction=38.0, high_blood_pressure='No', platelets=263358.03,
                      serum_creatinine=1.10, serum_sodium=136, sex='M', smoking='No', time=6):

    input = [age, yes_no_map[anaemia], creatinine_phosphokinase, yes_no_map[diabetes],
             ejection_fraction, yes_no_map[high_blood_pressure], platelets,
             serum_creatinine, serum_sodium, gender_map[sex], yes_no_map[smoking], time]

    input_to_model = np.array(input).reshape(1, -1)
    result = my_model.predict(input_to_model)
    #print(result)
if result[0]==1:
       return 'No'            # if DEATH_EVENT=1 means survive='No'
    else:
       return 'Yes'


# Input from user
in_age = gradio.Slider(minimum=40, maximum=100, value=55, step=1, label='Age (years)', show_label=True)
in_anaemia = gradio.Radio(["Yes", "No"], type="value", label="Decrease of red blood cells or hemoglobin", show_label=True)
in_creatinine = gradio.Slider(minimum=23.0, maximum=1281, value=1280.25, step=0.25, label='Level of the CPK enzyme in the blood (mcg/L)', show_label=True)
in_diabetes = gradio.Radio(["Yes", "No"], type="value", label="Has diabetes", show_label=True)
in_ejection = gradio.Slider(minimum=14, maximum=100, value=38, step=1, label='Percentage of blood leaving the heart at each contraction (%)', show_label=True)
in_bp = gradio.Radio(["Yes", "No"], type="value", label="Has hypertension/high blood pressure", show_label=True)
in_platelets = gradio.Slider(minimum=76000, maximum=440000, value=263358.03, step=1, label='Platelets in the blood (kiloplatelets/mL)', show_label=True)
in_serum_creatinine = gradio.Slider(minimum=0.5, maximum=2.15, value=1.1, step=.01, label='Level of serum creatinine in the blood (mg/dL)', show_label=True)
in_serum_sodium = gradio.Slider(minimum=125, maximum=148, value=136, step=1, label='Level of serum sodium in the blood (mEq/L)', show_label=True)
in_gender = gradio.Radio(["M", "F"], type="value", label="Gender", show_label=True)
in_smoking = gradio.Radio(["Yes", "No"], type="value", label="Smokes?", show_label=True)
in_time = gradio.Slider(minimum=4, maximum=285, value=6, step=1, label='Follow-up period (days)', show_label=True)

# Output response
out_response = gradio.components.Textbox(type="text", label='Survive')


# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = [in_age, in_anaemia, in_creatinine, in_diabetes, in_ejection,in_bp, in_platelets, in_serum_creatinine, in_serum_sodium, in_gender, in_smoking, in_time],
                         outputs = [out_response],
                         title = title,
                         description = description)

iface.launch(server_name="0.0.0.0", server_port = 8001)    # Ref: https://www.gradio.app/docs/interface
