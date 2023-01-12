import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tools.ssm_tools import *

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    s = np.loadtxt(uploaded_file)
else:
    s = np.loadtxt("data/har_example.txt", delimiter=",")[:15000, :]


st.title("NOVA App - Time Series Segmentation and Annotation")


with st.sidebar:
    # put params
    window_size = st.slider('window size', 10, len(s) // 3, len(s) // 10)
    kernel_size = st.slider('kernel size', 2, int(len(s)/(window_size-int(window_size*0.95))), int(len(s)/(window_size-int(window_size*0.95)))//3)

S = compute_ssm(s, window_size, 0.95)

nov_ssm = compute_novelty(S, kernel_size)

fig = make_subplots(rows=3, cols=1, row_heights=[0.2, 0.6, 0.2], vertical_spacing=0.025, shared_xaxes=True)
print(s.ndim)
if(s.ndim>1):
    for s_i in s.T:
        fig.add_trace(
            go.Line(x=np.linspace(0, len(s), len(s)), y=s_i),
            row=1, col=1
        )
else:
    fig.add_trace(
        go.Line(x=np.linspace(0, len(s), len(s)), y=s),
        row=1, col=1
    )
fig.add_trace(
    go.Heatmap(z=S, x=np.linspace(0, len(s), len(S)), y = np.linspace(0, len(s), len(S)), coloraxis="coloraxis"),
    row=2, col=1
)

fig.add_trace(
    go.Line(x=np.linspace(0, len(s), len(nov_ssm)), y=nov_ssm),
    row=3, col=1
)
fig.update_coloraxes(showscale=False, colorscale="Jet")

fig.update_layout(autosize=False, width=500, height=750, xaxis1_visible=False, yaxis2_visible=False, xaxis2_visible=False)
st.plotly_chart(fig)