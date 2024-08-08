import gzip
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os

filename = "{0}/{1}".format(
    "/home/tgrogers-raid/a/pan251/accel-sim-framework/sim_run_11.7/sponza_4k/hotlab/RTX3070-SASS-concurrent-mps_sm8-utility-VISUAL",
    "gpgpusim_visualizer__Sun-Jun--2-00-45-19-2024.log.gz"
)

array = pd.DataFrame(
    columns=["globalcyclecount",'cycle_counter', "tex_line", "verte_lines", "compute", "invalid", "g_count", "c_count", 'dynamic_sm_count']
)

if (filename.endswith('.gz')):
        file = gzip.open(filename, 'rt')
else:
    file = open(filename, 'r')

cycle = 0

while file:
    line = file.readline()
    if not line : break
    nameNdata = line.split(":")
    if (len(nameNdata) != 2): 
        print("Syntax error at '%s'" % line) 
    namePart = nameNdata[0].strip()
    dataPart= nameNdata[1].strip()
    if namePart == "globalcyclecount":
        cycle = int(dataPart)
        # if (cycle == 500):
            # array = array[0:0]
        # array = array.append(pd.Series(0, index=array.columns), ignore_index=True)
        array = pd.DataFrame([pd.Series(0, index=array.columns)])
        array = pd.concat([array, new_row], ignore_index=True)
        array.iloc[-1]['globalcyclecount'] = cycle
        array.iloc[-1]['cycle_counter'] = 500 * (array.shape[0] - 1)
        
    elif namePart == "L2Breakdown":
        data = dataPart.split(' ')
        array.iloc[-1]['tex_line'] = int(data[0])
        array.iloc[-1]['verte_lines'] = int(data[1])
        array.iloc[-1]['compute'] = int(data[2])
        array.iloc[-1]['invalid'] = int(data[3])
    elif namePart == "AvgGRThreads":
        data = float(dataPart)
        array.iloc[-1]['g_count'] = data
    elif namePart == "AvgCPThreads":
        data = float(dataPart)
        array.iloc[-1]['c_count'] = data
    elif namePart == 'dynamic_sm_count':
        data = float(dataPart)
        array.iloc[-1]['dynamic_sm_count'] = data

fig = go.Figure()

fig.add_trace(go.Scatter(x=array['cycle_counter'], y=array['g_count'] / (1024+512), mode='lines', 
                         hoverinfo='x+y', stackgroup='one', name='Rendering Shader'))
fig.add_trace(go.Scatter(x=array['cycle_counter'], y=array['c_count'] / (1024+512), mode='lines', 
                         hoverinfo='x+y', stackgroup='one', name='Compute Kernel'))


fig.update_layout(
    xaxis_title='Global Cycle',
    yaxis_title='Occupancy',
    xaxis=dict(
        titlefont=dict(size=25, color="black", family="sans-serif"),
        tickfont=dict(size=15, color="black", family="sans-serif"),
        autorange=True,
    ),
    yaxis=dict(
        titlefont=dict(size=25, color="black", family="sans-serif"),
        tickfont=dict(size=15, color="black", family="sans-serif"),
        autorange=True,
    ),
    width=800,
    height=300,
    title_font_family="sans-serif",
    title_font_size=25,
    margin=dict(l=20, r=10, t=50, b=0),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0,
        font=dict(size=20, family="sans-serif")
        
    ),
    title=dict(
        font=dict(size=25, family="sans-serif"),
        text="PT + VIO",
        # text="Occupancy: {0}".format('73.13%'),
        x=0.98,
        y=0.95
    ),

)

fig.write_image("./{0}.pdf".format("concurrent"), format="pdf")
np.average(array['tex_line']/1024/32)