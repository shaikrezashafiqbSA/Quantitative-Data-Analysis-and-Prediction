import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import re

def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

def get_plotly(filename,
               df_dict,
               window,
               cols_to_plot=None,
               row_heights:list=[1,1,1,1,1,5]):
    print("Publishing ...")
    
    instruments = list(df_dict.keys())
    if cols_to_plot is not None:
        number_subplots = 6+len(cols_to_plot)
        row_heights = row_heights+[1]*len(cols_to_plot)
    else:
        number_subplots = 6
        row_heights = row_heights
    
    
    fig = make_subplots(rows=number_subplots,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing = 0.05,
                        row_heights=row_heights,
                        specs=[[{"type":"scatter"}]]*number_subplots,
                        #subplot_titles = ("1h tide", "4h tide", "1d tide", "1w tide", "Total tide", "1h ADX and DMG","1h klines","1h MFI")
                        )
    
    color_map = ['#ffffff',
                 '#b3c6ff',
                 '#668cff',
                 '#0040ff']
    # TF_color_map = {"LTF":'rgba(0,255,255,1)',
    #                 "MTF":'rgba(0,155,255,1)',
    #                 "HTF":'rgba(155,0,255,1)' }
    timeframes = None
    for instrument in instruments:
        print(f"plotting {instrument} ...")
        df=df_dict[instrument][window[0]:].copy()
        if timeframes is None:
            timeframes =sort_human(list(np.unique([tf.split("_")[0] for tf in df.filter(regex="(h_)|(d_)|(w_)").columns])))
            assert len(timeframes) <=4
        # TABLE 1
# =============================================================================
#         # SUBPLOT 1-5 (TIDE STATES)
# =============================================================================
        tides = df.filter(regex='tide$').replace(0,-1)
        row=1
        color_count=0
        for tf in timeframes:
            tide_TF_ax = go.Scattergl(x=tides.index,
                                      y=tides[f"{tf}_tide"],
                                      name=f"{tf}_tide", 
                                      mode='markers',
                                      marker=dict(color=color_map[color_count]))
            fig.append_trace(tide_TF_ax,row=row,col=1)
            row+=1
            color_count+=1
            
        total_tide = tides.sum(axis=1)
        tide_TF_ax = go.Scattergl(x=total_tide.index, y=total_tide,name=f"total_tide")
        fig.append_trace(tide_TF_ax,row=row,col=1)    
        fig['layout'][f'xaxis{row}']['title']="tide states"
        row+=1
        
        
# =============================================================================
#         # SUBPLOT 7 (KLINES + ebbs)
# =============================================================================
        klines_ax = go.Candlestick(x=df.index,
                                   open=df["1h_open"],
                                   high=df["1h_high"],
                                   low=df["1h_low"],
                                   close=df["1h_close"],
                                   name=f"{instrument} 1h",
                                   increasing_line_color='rgb(14,203,129)',
                                   decreasing_line_color='rgb(233,67,89)')
        fig.add_trace(klines_ax,row=row,col=1)
        
        color_count=0
        for tf in timeframes:
            # TIDE PLOTSS
            # print(f"----> tide states {tf}: {color_map[color_count]}")
            tide_TF_ax = go.Scattergl(x=df.index,
                                      y=df[f"{tf}_ebb"],
                                      name=f"{tf} mx",
                                      mode='lines',
                                      line=dict(color=color_map[color_count])
                                      )
            fig.append_trace(tide_TF_ax,row=row,col=1)
            color_count+=1
        row+=1
# =============================================================================
#         # SUBPLOT 6,7,etc (MISC INDICATORS)
# =============================================================================
        if cols_to_plot is not None:
            for col in cols_to_plot:
                color_count=0
                for tf in timeframes:
                    # TIDE PLOTSS
                    misc_ax = go.Scattergl(x=df.index,
                                           y=df[f"{tf}_{col}"],
                                           name=f"{tf} {col}",
                                           mode='lines',
                                           line=dict(color=color_map[color_count]))
                    fig.append_trace(misc_ax,row=row,col=1)
                    color_count+=1  
                fig['layout'][f'xaxis{row}']['title']=col
                row+=1
            

        
    Ld = len(fig.data)
    Lc = len(instruments)
    trace_per_selection = int(Ld/Lc)
    for k in range(0,Ld):
        fig.update_traces(visible=False,selector=k)
        
    def create_layout_button(k,instrument):
        visibility=[False]*int(Ld)
        for tr in range(trace_per_selection*k,(k+1)*trace_per_selection):
            visibility[tr]=True
        return dict(label=instrument,
                    method='restyle',
                    args=[{'visible':visibility,
                           'title':instrument,
                           'showlegend':True}])
    
    fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0,
                                                        buttons=[create_layout_button(k,instrument) for k,instrument in enumerate(instruments)],
                                                        font=dict(color='#dfd9d3',size=15)
                                                        )],
                      xaxis_rangeslider_visible=False,
                      hovermode="x",
                      font=dict(family="Courier New, monospace",size=15,color="white"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=False)
    
    # fig['layout']['yaxis']['title']='klines'

    fig.update_layout(paper_bgcolor="#21252C", plot_bgcolor="rgb(22,26,30)")
    # fig.update_xaxes(type="date", range=window)
    fig.update_xaxes(showgrid=True,gridwidth=1,gridcolor='rgb(61,58,58)')
    fig.update_yaxes(showgrid=True,gridwidth=1,gridcolor='rgb(61,58,58)')
    fig.update_xaxes(color="grey")
    fig.update_yaxes(color="grey")
    # fig.show()
    now = dt.now()
    publish_date = str(dt(now.year,now.month,now.day))[0:10]
    fig.write_html(filename+".html")
