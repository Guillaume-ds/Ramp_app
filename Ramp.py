
import streamlit as st
import os 
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.markdown("""<h1 style='text-align: center; font-weight:bold;padding-bottom:15px; margin-bottom:10px;color:rgba(20,10,80,1); text-decoration: underline rgba(20,10,80,1) 3px;'>
                    RAMP PROJECT PRESENTATION</h1>""", unsafe_allow_html=True)

# Description
st.write("This web app is designed to present the data exploration done by Zofia and Guillaume during the ramp project.")

path = os.path.dirname(__file__)

my_file = path+'/Static/Bike.jpeg'
st.image(Image.open(my_file))
# Header
st.header("Structure of the presentation :")

st.markdown("""
            <ol>
            <li>Litterature and data analysis
                <ul>
                <li>Main factors</li>
                <li>Comparison to existing studies</li>
                <li>Correlation plots</li>
                </ul>
            </li>
            <li>Modelling
                <ul>
                <li>Regression</li>
                <li>XGB</li>
                <li>Catboost</li>
                </ul>
            </li>
            <li>Key takeways
                <ul>
                <li>What we've learned</li>
                <li>Some memes for the way</li>
                </ul>
            </li>
            </ol>
            """, unsafe_allow_html=True)

