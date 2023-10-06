import gradio as gr
import joblib
import pickle
import numpy as np
import pandas as pd
pickle_in = open('xgboost-model.pkl', 'rb')
classifier = pickle.load(pickle_in)

def predict_death_event(age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time):

    # Pre-processing user input
    if high_blood_pressure == "Yes":
        high_blood_pressure = 1
    else:
        high_blood_pressure = 0
    
    if sex == "M":
        Sex = 1
    else:
        Sex = 0

    if anaemia == "Yes":
        anaemia = 1
    else:
        anaemia = 0

    if diabetes == "Yes":
        diabetes = 1
    else:
        diabetes = 0

         # Making predictions
    predict_death_event = classifier.predict(
        [[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])

    if predict_death_event == 0:
        pred = 'will live'
    else:
        pred = 'Death'
    return pred

    def dataset_summary():
    summary = f"Dataset Summary:\n\n"
    summary += data.describe().to_string()  # Use describe() to generate dataset statistics

    return summary

in_prompt =[
                         gr.inputs.Slider(minimum=0, maximum=100, default=50, label="Patient's Age"),
                         gr.inputs.Radio(["Yes", "No"], label="High Blood Pressure (HBP)"),
                         gr.inputs.Slider(minimum=0, maximum=100, default=50, label="Duration of Condition (Time)"),
                         gr.inputs.Slider(minimum=0, maximum=100, default=50, label="Ejection Fraction (%)"),
                         gr.inputs.Slider(minimum=0, maximum=10, default=5, label="Serum Creatinine"),
                         gr.inputs.Slider(minimum=0, maximum=200, default=100, label="Serum Sodium")
          ]

# Output response
out_response =[
                         gr.outputs.Textbox(label="Prediction"),
                         gr.outputs.Textbox(label="Confidence Score (%)")
                     ]
# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gr.Interface(fn = predict_death_event,
                         inputs=[
        gr.components.Slider(minimum=0, maximum=100, label="Patient's Age"),
        gr.components.Radio(choices=["0", "1"], label="Anemia"),
        gr.components.Slider(minimum=0, maximum=100000, label="creatinine phosphokinase	"),
        gr.components.Radio(choices=["0", "1"], label="Diabetes"),
        gr.components.Slider(minimum=0, maximum=100,  label="Ejection Fraction (%)"),
        gr.components.Radio(["Yes", "No"], label="High Blood Pressure (HBP)"),
        gr.components.Slider(minimum=0, maximum=10000000, label="platelets"),
        gr.components.Slider(minimum=0, maximum=10, label="Serum Creatinine"),
        gr.components.Slider(minimum=0, maximum=200, label="Serum Sodium"),
        gr.components.Radio(["0", "1"], label="Sex"),
        gr.components.Radio(["0", "1"], label="Smoking"),
        gr.components.Slider(minimum=0, maximum=100, label="Duration of Condition (Time)")
    ],
    outputs=[
        gr.components.Textbox(type="text", label="Prediction DEATH"),
    ],
                         title = title,
                         description = description)



iface.launch(server_name = "0.0.0.0", server_port = 8001)  # Ref: https://www.gradio.app/docs/interface
