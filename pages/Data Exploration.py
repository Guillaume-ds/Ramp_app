import streamlit as st
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

#------------------------------------------------

st.markdown("""<h1 style='text-align: center; font-weight:bold;padding-bottom:15px; margin-bottom:10px;color:rgba(20,10,80,1); text-decoration: underline rgba(20,10,80,1) 3px;'>
                    RAMP DATA EXPLORATION</h1>""", unsafe_allow_html=True)

st.write("At the beginning, we made sure that we understood the variables and their potential impact on the bike traffic.")
#------------------------------------------------
data = pd.read_parquet(Path("Static") / "train.parquet")
data["hour"] = data["date"].dt.hour
data["weekday"] = data["date"].dt.weekday
data["monthday"] = data["date"].dt.day
data["week"] = data["date"].dt.isocalendar().week 
data["month"] = data["date"].dt.month

st.table(data.dtypes)
#------------------------------------------------
st.markdown("""<hr style='text-align: center; font-weight:bold;padding-bottom:15px; margin-bottom:10px;border-color:rgba(20,10,80,0); ,1) 3px;'></hr>""", unsafe_allow_html=True)

st.write("Then we focused on time variables. We used a first function to clean and study the data:")

code = """
def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])
"""

st.code(code)

#------------------------------------------------

st.write("And we tried to do feature engineering on the data concerning the hours.")

code2 = """
df["modified_hour"] = df["hour"].apply(lambda x : 
                                  (-np.sin((2*x+1)/(np.pi))+4) if x>5 and x<20 
                                  else (-np.sin((2*x+1)/(np.pi))+1))
"""

st.code(code2)



addConst = st.slider(label="Added constant",min_value=0, max_value=15, value=4)

data["modified_hour"] = data["hour"].apply(lambda x : 
                                       (-np.sin((2*x+1)/(np.pi))+addConst) if x>5 and x<20
                                       else (-np.sin((2*x+1)/(np.pi))+1))


fig, ax = plt.subplots() 
ax = plt.scatter(data["hour"],data["modified_hour"],c="#1E0832",label="Modified value")
day = plt.axvspan(6, 19, alpha=0.3, color='#BD92E1')
plt.xlabel("Hour of the day")
plt.ylabel("Modified hour")
day.set_label('Day time')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
#plt.figure(figsize=(15, 7))
st.pyplot(fig)


#------------------------------------------------
st.markdown("""<hr style='text-align: center; font-weight:bold;padding-bottom:15px; margin-bottom:10px;border-color:rgba(20,10,80,0); ,1) 3px;'></hr>""", unsafe_allow_html=True)

st.write("We then did more data visualization in order to get a better understanding of the overall data set.")
step_name = "hour"
fig = go.Figure()
fig.update_layout(mapbox_style="open-street-map")
slider_range=data[step_name].unique()
slider_range.sort()

#adding trace
for step_range in slider_range:
    data_filtered=data[data[step_name] == step_range]


    data_point_sum = data_filtered.groupby(["site_name"])["bike_count"].mean().sort_values(
        ascending=False
    )
    counter_dict=data_filtered[["site_name", "latitude", "longitude"]].drop_duplicates("site_name")

    data_point_sum = pd.merge(data_point_sum, counter_dict, on ="site_name", how ='inner').drop_duplicates("site_name")

    fig.add_trace(go.Scattermapbox(
                        lat=data_point_sum["latitude"],
                        lon=data_point_sum["longitude"],
                        mode='markers',
                        text=data_point_sum["site_name"],
                        opacity=0.4,
                        marker_size=data_point_sum["bike_count"]/5, marker_color='#b50015',
                        hoverinfo='text'))

    
#adding slider    
steps = []
for i in slider_range:
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              #{"title": "Weekday: " + str(i)}
              ],
        label = '{} {}'.format(step_name,i))  # layout attribute
    step["args"][0]["visible"][i-1] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=i,
    #currentvalue={"prefix": "Frequency: "},
    #pad={"t": 0},
    steps=steps
)]


fig.update_geos(fitbounds="locations")
fig.update_mapboxes(
    center={"lat": 48.8566, "lon": 2.3522},
    zoom = 11)
fig.update_layout(
    sliders=sliders, 
    height=700,
    margin={"r":30,"t":20,"l":30,"b":10}
)

st.plotly_chart(fig)
#------------------------------------------------
st.markdown("""<hr style='text-align: center; font-weight:bold;padding-bottom:15px; margin-bottom:10px;border-color:rgba(20,10,80,0); ,1) 3px;'></hr>""", unsafe_allow_html=True)

st.write("""Once we got a better understanding of the data set
 we tried to add more feature engineering, especially with the external data. 
 We uploaded a part of the external data set, and we worked on those new pieces of informations : """)

code3 = """
def _merge_external_data(X):
    file_path = str(Path("Static")) +"/"+ "train.parquet"
    df_ext = pd.read_csv(file_path,encoding='latin-1')

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values("date"), df_ext.sort_values("date"), on="date")
    # Sort back to the original order
    X["ff_2"] = X["ff"] ** 2
    X = X.fillna(X.mean())
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X
"""

st.code(code3)
#------------------------------------------------
st.write("At that point we mainly tried to understand the correlations between the different factors:")

code4 = """
data_point_sum = data.groupby(["site_name"])["bike_count"].sum().sort_values(
    ascending=False
)
counter_dict=data[["site_name", "latitude", "longitude"]].drop_duplicates("site_name")

data_point_sum = pd.merge(data_point_sum, counter_dict, on ="site_name", how ='inner').drop_duplicates("site_name")

plt.figure(figsize=(16, 8))
# creating mask

mask = np.triu(np.ones_like(data_relevant_corr))
sns.set(font_scale=2, rc={'axes.facecolor':'#e0dcdc', 'figure.facecolor':'#e0dcdc'})
#sns.set(font_scale=2)
#e0dcdc
 
# plotting a triangle correlation heatmap
dataplot = sns.heatmap(data_relevant_corr, cmap="bwr", annot=True, mask=mask)
 
# displaying heatmap
#plt.axes(color='#e0dcdc')
plt.show()
"""

st.code(code4)
st.write("This table shows how important some of the variables are :")
path = Path("Static")
my_file = str(path)+'/corr.png'
st.image(Image.open(my_file))

st.markdown("""<hr style='text-align: center; font-weight:bold;padding-bottom:15px; margin-bottom:10px;border-color:rgba(20,10,80,0); ,1) 3px;'></hr>""", unsafe_allow_html=True)





st.write("Finaly we decided to work mainly with this dataframe:")

path = Path("Static")
my_file = str(path)+'/Final_data.png'
st.image(Image.open(my_file))