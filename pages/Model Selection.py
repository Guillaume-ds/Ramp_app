import streamlit as st
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import time
#------------------------------------------------

st.markdown("""<h1 style='text-align: center; font-weight:bold;padding-bottom:15px; margin-bottom:10px;color:rgba(20,10,80,1); text-decoration: underline rgba(20,10,80,1) 3px;'>
                    RAMP MODEL SELECTION</h1>""", unsafe_allow_html=True)

st.write("""After a few days working ainly on the data, we decided to try new models. Indeed, the use of the regression model
was not as succesfull as expected!""")

#------------------------------------------------
st.markdown("""<hr style='text-align: center; font-weight:bold;padding-bottom:15px; margin-bottom:10px;border-color:rgba(20,10,80,0); ,1) 3px;'></hr>""", unsafe_allow_html=True)

st.write("The first algorithm we tried was the XGboost. We tried a lot of parameters and got decent results. It was our first great leap forward.")


with st.expander("Tune the parameters :"):
  col1, col2, col3 = st.columns(3)

  with col1:
    md = st.number_input(label="max_depth", min_value=None, max_value=None, value=9,)

  with col2:
    ns = st.number_input(label="max_depth", min_value=None, max_value=None, value=50,)

  with col3:
    vb = st.number_input(label="max_depth", min_value=None, max_value=None, value=3,)


code = f"""
def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    time_columns = ["month", "weekday", "hour"]
    place_columns = ["site_name"]
    weather_columns = ["u", "ff", "t"] 

    preprocessor = ColumnTransformer([
            #('holidays', OneHotEncoder(), holidays_columns),
            ('time', "passthrough", time_columns),
            ('place', OneHotEncoder(categories='auto',handle_unknown='ignore'),
                                     place_columns),
            ('weather', "passthrough", weather_columns)
        ], remainder="drop")

    regressor = GradientBoostingRegressor(max_depth={md}, n_estimators={ns}, verbose={vb})

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder, 
        preprocessor, 
        regressor)

    return pipe
"""

st.code(code)

st.write("This model helped us to get closer to the reality :")

path = Path("Static")
my_file = str(path)+'/XGB.png'
st.image(Image.open(my_file))

#------------------------------------------------
st.markdown("""<hr style='text-align: center; font-weight:bold;padding-bottom:15px; margin-bottom:10px;border-color:rgba(20,10,80,0); ,1) 3px;'></hr>""", unsafe_allow_html=True)

st.write("At the end of the project, we finally decided to work with CatBoost. This algorithm enabled us to get our best scores.")


code4 = """
#CatBoost
def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    holidays_columns = ["isholiday"]
    time_columns = ["month", "year","weekday", "hour"]
    place_columns = ["site_name"]
    weather_columns = ["u", "t", "ff"]
    phenomenon_columns = ["phenspe2"]
 
    preprocessor = ColumnTransformer([
            ('time', "passthrough", time_columns),
            ('place', OneHotEncoder(categories='auto',handle_unknown='ignore'), place_columns),
            ('weather', "passthrough", weather_columns),
        ], remainder="drop")
 
    regressor = CatBoostRegressor()
 
    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder, 
        preprocessor, 
        regressor)
 
    return pipe
"""

st.code(code4)