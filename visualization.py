import streamlit as st
import pandas as pd
import altair as alt
from pycaret.regression import *

# load model
@st.cache 
def predict_cache(test_data):
    rf_saved = load_model('rf_model')
    predictions = predict_model(rf_saved, data = test_data)
    return predictions['Label']

# load data
def load_data():
    data = pd.read_csv('./data/Train_Data.csv')
    return data

def preprocess_inputs(df):
    
    df = df.copy()
    sex_wrapper = {'male':0, 'female':1}
    df.sex = df.sex.replace(sex_wrapper)

    df.smoker.value_counts()
    smoker_wrapper = {'no':0, 'yes':1}
    df.smoker = df.smoker.replace(smoker_wrapper)
    
    df = pd.get_dummies(df, columns=['region'])
    
    return df

def addMissingColumns(df):
    if 'region_northeast' not in df:
        df['region_northeast'] = 0
    if 'region_northwest' not in df:
        df['region_northwest'] = 0
    if 'region_southeast' not in df:
        df['region_southeast'] = 0
    if 'region_southwest' not in df:
        df['region_southwest'] = 0

st.set_page_config(layout="wide")

st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.title('Medical Insurance cost Prediction')


data = load_data().copy()

prediction, visualization = st.tabs(['Prediction', 'Visualizations'])

with prediction:
    col1, col2 = st.columns(2)
    with col1:
        sex = col1.radio('Sex', ('female', 'male'), index=0, horizontal=True)
        smoker = col1.radio('Smoker', ('no', 'yes'), index=0, horizontal=True)
        region = col1.radio('Region', ('southeast', 'southwest', 'northeast', 'northwest'), index=0, horizontal=True)
        age = col1.slider('Age', 18, 64, 35, step=1)
        bmi = col1.slider('BMI', 15.0, 54.0, 21.0, step=0.1)
        children = col1.slider('Children', 0, 5, 0, step=1)

        button  = col1.button("Predict")

    with col2:
        if button:
            entry = pd.DataFrame({'age': [age], 
                                    'sex': [sex], 
                                    'bmi': [bmi], 
                                    'children' : [children], 
                                    'smoker': [smoker], 
                                    'region': [region]})

            entry = preprocess_inputs(entry)
            addMissingColumns(entry)
            pred = round(predict_cache(entry)[0], 4)

            hist = alt.Chart().mark_bar().encode(
                x=alt.X('charges:Q', bin=alt.Bin(extent=[0, 50000], step=1000), title='Insurance Charges'),
                y=alt.Y('count()', title='Count'),
                tooltip=[alt.Tooltip('count()', title='Count')]
            )

            median_line = alt.Chart().mark_rule().encode(
                x=alt.X('median(charges):Q', title='Insurance Charges'),
                size=alt.value(2),
                color=alt.value('yellow'),
                tooltip=[alt.Tooltip('median(charges):Q', title='Median Insurance Charge')]
            )

            pt = alt.Chart().mark_rule().encode(
                x=alt.X('pred:Q', title='Insurance Charges'),
                size=alt.value(2),
                color=alt.value('red'),
                tooltip=[alt.Tooltip('pred:Q', title='Your Insurance Charge')]
            )

            html_str = f"""
                    <h2 style='text-align: center; color: #FF4B4B;'>Insurance charges = ${pred} </h2>
                """

            col2.markdown(html_str, unsafe_allow_html=True)

            pot = alt.layer(
                hist, 
                median_line, 
                pt, 
                data=data
            ).transform_calculate(
                pred=str(pred)
            ).properties(
                title='Comparing your Insurance charge to other Users',
                width=600,
                height=500
            )

            col2.write(pot)
        else:
            col2.subheader('Press the predict button to view the predicted insurance cost and more!')
            col2.image('./images/healthInsurance.jpeg')

with visualization:
    col1, col2 = st.columns(2)
    
    with col1:
        density_graph = alt.Chart(data).transform_density(
                'charges', as_=['CHARGES', 'DENSITY']
            ).mark_area(
                color='blue',opacity=0.3
            ).encode(
                x="CHARGES:Q",y='DENSITY:Q'
            ).properties(
                title='Density Graph of Insurance Charges',
                width=600,
                height=400
            )
        col1.write(density_graph)
        data['age'] = data['age'].astype(int)
        hist1 = alt.Chart(data).mark_line().encode(
                x=alt.X('age:N'),
                y=alt.Y('mean(charges):Q', title='Average Insurance cost'),
                color='sex:N',
                tooltip=[alt.Tooltip('mean(charges):Q', title='Average Charges'), alt.Tooltip('age:Q', title='Age')]
            ).properties(
                title='Effect of Gender on Insurance Charges',
                width=600,
                height=400
            )
        col1.write(hist1)
        hist3 = alt.Chart(data).mark_bar().encode(
                x=alt.X('children:N'),
                y=alt.Y('mean(charges):Q', title='Average Insurance cost'),
                tooltip=[alt.Tooltip('mean(charges):Q', title='Average Charges')]
            ).properties(
                title='Children vs Charges',
                width=600,
                height=400
            )
        col1.write(hist3)

    with col2:
        scatter_plot = alt.Chart(data).mark_circle().encode(
                x=alt.X('bmi:Q', title='BMI'), 
                y=alt.Y('charges:Q', title='Insurance Cost'), 
                color='smoker:N'
            ).properties(
                title='BMI vs Charges',
                width=600,
                height=400
            )
        col2.write(scatter_plot)
        hist2 = alt.Chart(data).mark_line().encode(
                x=alt.X('age:N'),
                y=alt.Y('mean(charges):Q', title='Average Insurance cost'),
                color='smoker:N',
                tooltip=[alt.Tooltip('mean(charges):Q', title='Average Charges'), alt.Tooltip('age:Q', title='Age')]
            ).properties(
                title='Effect of Smoking on Insurance Charges',
                width=600,
                height=400
            )
        col2.write(hist2)
        hist4 = alt.Chart(data).mark_bar().encode(
                x=alt.X('region:O'),
                y=alt.Y('mean(charges):Q', title='Average Insurance cost'),
                tooltip=[alt.Tooltip('mean(charges):Q', title='Average Charges')]
            ).properties(
                title='Region vs Charges',
                width=600,
                height=400
            )
        col2.write(hist4)