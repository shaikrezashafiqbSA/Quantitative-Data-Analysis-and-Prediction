import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.get_time import get_dt_times_now

def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

def get_plotly_univariate(model_config, 
                          df_backtested, 
                          df_trades, 
                          df_summary, 
                          replay_dt = None,
                          html_file_path = "D:PycharmProjects/trading_signal_V3/backtests/trading_bot_plotly/", 
                          publish = True,
                          convert_tz = None,
                          cols_to_plot = [["cum_L_pnl", "cum_S_pnl"],
                                          ["ohlc_PLACEHOLDER_STR",'L_entry_price','L_exit_price','S_entry_price','S_exit_price',],
                                          ['zscore'],
                                          ],
                         subplot_titles = ["Long/short/total cumulative PNL $",f"USDSGD","signal"],
                          row_heights:list=[1,1,1,1,1,5],
                            width = 1200,
                            height = 800,
                          verbose = False,
                          manual_tail_window = None,):
    df_backtested0 = df_backtested.copy()
    df_trades0 = df_trades.copy()
    df_summary0 = df_summary.copy()
    if manual_tail_window is not None:
        df_backtested = df_backtested.tail(manual_tail_window)
        df_trades = df_trades.tail(manual_tail_window)
        df_summary = df_summary.tail(manual_tail_window)

    if os.path.exists(html_file_path):
        print("PLOTLY DIRECTORY EXISTS")
    else:
        print(f"PLOTLY DIRECTORY ({html_file_path}) DOES NOT EXIST")
        os.makedirs(html_file_path)
        print(f"CREATED {html_file_path} --> {os.path.exists(html_file_path)}")
        


    window = model_config["memory_len"]
    html_name = f"{model_config['instrument_to_trade']}_{model_config['model_name']}_{model_config['signal_to_trade']}_{model_config['timeframe_to_trade']}"
    html_file_path_name = f"{html_file_path}{html_name}"

    if verbose: print(f"Publishing ... {html_file_path_name}")
    df_dict = {model_config["instrument_to_trade"]: df_backtested}
    instruments = list(df_dict.keys())
    
    # =============================================================================
    # SET UP FIGURE
    # =============================================================================

    # 888888888888888888888888888
    # SPECS
    # 888888888888888888888888888
    specs = []
    for col in cols_to_plot:
        if col in ['table__df_trades', 'table__df_summary']:
            spec = {"type": "table", "secondary_y": False}
        else:
            spec = {"type": "xy", "secondary_y": True}
        specs.append([spec])

    if verbose: print(specs)
    fig = make_subplots(rows=len(cols_to_plot),
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing = 0.05,
                        row_heights=row_heights,
                        specs=specs,
                        subplot_titles=subplot_titles
                        )
    
    # =============================================================================
    # SET COLORS/FONT 
    # =============================================================================
    
    # TABLE HEADER
    header_font_sizes_bigger_cols = ["L_cost", "L_qty", "L_fees", "S_cost", "S_qty", "S_fees", "L_rpnl","S_rpnl","A_rpnl", "cum_A_pnl","S_positions","L_positions", "L_entry_price", "L_exit_price", "S_entry_price", "S_exit_price"]
    header_font_size_bigger = 9
    header_font_sizes_smaller_cols = ["id", "cost", "qty", "fees", "rpnl", "Trade datetime"]
    header_font_size_smaller = 15
    header_font_size_other = 12

    header_font_color_L = 'lightgreen'
    header_font_color_S = 'orange'
    header_font_color_A = 'lightblue'

    # TABLE CELL
    cell_fill_color_str = 'black'
    cell_fill_color_others = 'black'

    cell_font_color_lesser = 'red'
    cell_font_color_greater = 'green'
    cell_font_color_equal = 'white'
    cell_font_color_others = 'white'
    cols_to_highlight = ["L_rpnl", "S_rpnl", "A_rpnl", "cum_A_pnl"]
    
    cell_font_size_index = 11
    cell_font_size_int = 14
    cell_font_size_float = 12
    
    # Define the column widths
    # The first value is for the index column, and the rest are for the other columns
    cell_widths = [200] + [100]*(len(df_trades.columns)-1)


    # KLINES PLOT 
    color_S_trade = "orange"
    color_L_trade = "green"

    color_L_pnl = "green"
    color_S_pnl = "orange"
    color_pnl_opacity = 0.5
    color_pnl_size = 3
    # =============================================================================
    # BEGIN POPULATING FIGURE
    # =============================================================================
    for i, col in enumerate(cols_to_plot):
        if i ==0:
            continue
        fig.update_xaxes(title_text=subplot_titles[i], row=i+1, col=1)

    for i, col in enumerate(cols_to_plot):
        if i == 0:
            fig.update_yaxes(title_text=model_config["instrument_to_trade"], row=i+1, col=1)
        elif i ==1:
            fig.update_yaxes(title_text=model_config["signal_function"], row=i+1, col=1)
    
    timeframes = model_config["resample_to_list"]
    for instrument in instruments:
        if verbose: print(f"plotting {instrument} ...")
        df0=df_dict[instrument].tail(window)
        if timeframes is None:
            # timeframes =sort_human(list(np.unique([tf.split("_")[0] for tf in df.filter(regex="(h_)|(d_)|(w_)").columns])))
            assert len(timeframes) <=4
        
        # ===================================================================================================================
        # ===================================================================================================================
        #                                               INSTRUMENT i                                                        #
        # ===================================================================================================================
        # ===================================================================================================================
        df = df0.copy()
        row=1   
        # Convert index to datetime
        if convert_tz is not None:
            if verbose: print(f"CONVERTING TIMEZONE: {convert_tz['from']} ----> {convert_tz['to']}")
            if verbose: print(f"initially: {df.index[-1]}")
            df.index = df.index.tz_localize(convert_tz["from"])#.tz_convert('UTC')
            df.index = df.index.tz_convert(convert_tz["to"])
            if verbose: print(f"now: {df.index[-1]}")
            df.index = df.index.tz_localize(None)
            if verbose: print(f"now: {df.index[-1]}")

        for i, col in enumerate(cols_to_plot):
            if verbose: print(f"plotting: {col}")

            # ================================================================================
            #         # SUBPLOT 2: SUMMARY TABLE
            # ================================================================================ 
            if verbose: print(f"\nLOG i: {i} || col: {col} \nperformance_analytics.plots() ln167 - col == 'table__df_summary' \n==> {col} == {'table__df_summary'} \n==> {col == 'table__df_summary'}")
            
            if col == 'table__df_summary':
                # df_summary = df_summary0.copy().T

                # =============================================================================
                # CLEAN trades df
                # =============================================================================
                    # Round all float columns
                for col_i in df_summary.columns:
                    if df_summary[col_i].dtype == 'float64':
                        df_summary[col_i] = df_summary[col_i].round(model_config["precision"])
                    if col_i in ["L_cost", "L_qty", "L_fees", "S_cost", "S_qty", "S_fees", "L_rpnl","S_rpnl","A_rpnl", "cum_A_pnl"]:
                        df_summary[col_i] = df_summary[col_i].round(2)

                
                df_summary = df_summary.fillna('')

                cell_font_aligns = ['left'] * len(df_summary.columns)
                # Create a list of font sizes for the header
                header_font_sizes = []
                for col_i in df_summary.columns:
                    if col_i in header_font_sizes_smaller_cols:
                        header_font_sizes.append(header_font_size_smaller)
                    elif col_i in header_font_sizes_bigger_cols:
                        header_font_sizes.append(header_font_size_bigger) 
                    else:
                        header_font_sizes.append(header_font_size_other) 

                # Create a list of colors for the font color of each cell
                cell_font_colors = []
                for row_i in df_summary.values:
                    row_colors = []
                    for val, col_i in zip(row_i, df_summary.columns):
                        if col_i in cols_to_highlight:
                            
                            if type(val) != str:
                                if val > 0:
                                    row_colors.append(cell_font_color_greater)
                                elif val < 0:
                                    row_colors.append(cell_font_color_lesser)
                                elif val == 0:
                                    row_colors.append(cell_font_color_equal)
                                else:
                                    row_colors.append(cell_font_color_others)
                            else:
                                row_colors.append(cell_font_color_others)
                        else:
                            row_colors.append(cell_font_color_others)
                    cell_font_colors.append(row_colors)


                # Create a list of font sizes for each cell
                cell_font_sizes = []
                for col_i in df_summary.columns:
                    if df_summary[col_i].dtype == 'int64':
                        cell_font_sizes.append(cell_font_size_int)
                    elif df_summary[col_i].dtype == 'float64':
                        cell_font_sizes.append(cell_font_size_float)
                    else:
                        cell_font_sizes.append(cell_font_size_index)

                # Create a list of colors for the fill color of each cell
                cell_fill_colors = []
                for row_i in df_summary.values:
                    row_colors = []
                    for val in row_i:
                        if isinstance(val, str):
                            row_colors.append(cell_fill_color_str)
                        else:
                            row_colors.append(cell_fill_color_others)
                    cell_fill_colors.append(row_colors)

                # Create a list of font colors for the header
                header_font_colors = []
                for col_i in df_summary.columns:
                    if "L" in col_i:
                        header_font_colors.append(header_font_color_L)
                    elif "S" in col_i:
                        header_font_colors.append(header_font_color_S)
                    else:
                        header_font_colors.append(header_font_color_A)


                # Replace null values with an empty string

                fig.add_trace(go.Table(header=dict(values=list(df_summary.columns),
                                                    fill_color=cell_fill_colors,
                                                    align='left',
                                                    font=dict(size=header_font_sizes, color=header_font_colors)
                                                    ),
                                        cells=dict(values=[df_summary[col_i] for col_i in df_summary.columns],
                                                    fill_color=cell_fill_colors,
                                                    align=cell_font_aligns,
                                                    font=dict(size=cell_font_sizes, color=cell_font_colors)),
                                                    columnwidth=cell_widths  # Adjust this )
                                                    ),
                                secondary_y=False,
                                row=i+1,
                                col=1
                                )
                if verbose: print(f"performance_analytics.plots() ln276 ==> row: {row} col: {col} trade table done")
                

            # ================================================================================
            #         # SUBPLOT 1: TRADES TABLE
            # ================================================================================ 
            elif col == 'table__df_trades':
                df_trades = df_trades0.copy()

                # =============================================================================
                # CLEAN trades df
                # =============================================================================
                    # Round all float columns
                for col_i in df_trades.columns:
                    if df_trades[col_i].dtype == 'float64':
                        df_trades[col_i] = df_trades[col_i].round(model_config["precision"])
                    if col_i in ["L_cost", "L_qty", "L_fees", "S_cost", "S_qty", "S_fees", "L_rpnl","S_rpnl","A_rpnl", "cum_A_pnl"]:
                        df_trades[col_i] = df_trades[col_i].round(2)
                # sort df_trades by index descending
                df_trades = df_trades.sort_index(ascending=False)

                # Add the index as a column
                df_trades['index'] = df_trades.index
                # Move the new column to the front of the DataFrame
                df_trades = df_trades.set_index('index').reset_index().rename(columns={'index': 'Trade datetime'})
                df_trades = df_trades.fillna('')

                cell_font_aligns = ['left'] * len(df_trades.columns)
                # Create a list of font sizes for the header
                header_font_sizes = []
                for col_i in df_trades.columns:
                    if col_i in header_font_sizes_smaller_cols:
                        header_font_sizes.append(header_font_size_smaller)
                    elif col_i in header_font_sizes_bigger_cols:
                        header_font_sizes.append(header_font_size_bigger) 
                    else:
                        header_font_sizes.append(header_font_size_other) 

                # Create a list of colors for the font color of each cell
                cell_font_colors = []
                for row_i in df_trades.values:
                    row_colors = []
                    for val, col_i in zip(row_i, df_trades.columns):
                        if col_i in cols_to_highlight:
                            
                            if type(val) != str:
                                if val > 0:
                                    row_colors.append(cell_font_color_greater)
                                elif val < 0:
                                    row_colors.append(cell_font_color_lesser)
                                elif val == 0:
                                    row_colors.append(cell_font_color_equal)
                                else:
                                    row_colors.append(cell_font_color_others)
                            else:
                                row_colors.append(cell_font_color_others)
                        else:
                            row_colors.append(cell_font_color_others)
                    cell_font_colors.append(row_colors)

                # Create a list of font sizes for each cell
                cell_font_sizes = []
                for col_i in df_trades.columns:
                    if df_trades[col_i].dtype == 'int64':
                        cell_font_sizes.append(cell_font_size_int)
                    elif df_trades[col_i].dtype == 'float64':
                        cell_font_sizes.append(cell_font_size_float)
                    else:
                        cell_font_sizes.append(cell_font_size_index)

                # Create a list of colors for the fill color of each cell
                cell_fill_colors = []
                for row_i in df_trades.values:
                    row_colors = []
                    for val in row_i:
                        if isinstance(val, str):
                            row_colors.append(cell_fill_color_str)
                        else:
                            row_colors.append(cell_fill_color_others)
                    cell_fill_colors.append(row_colors)

                # Create a list of font colors for the header
                header_font_colors = []
                for col_i in df_trades.columns:
                    if "L" in col_i:
                        header_font_colors.append(header_font_color_L)
                    elif "S" in col_i:
                        header_font_colors.append(header_font_color_S)
                    else:
                        header_font_colors.append(header_font_color_A)


                # Replace null values with an empty string

                fig.add_trace(go.Table(header=dict(values=list(df_trades.columns),
                                                    fill_color=cell_fill_colors,
                                                    align='left',
                                                    font=dict(size=header_font_sizes, color=header_font_colors)
                                                    ),
                                        cells=dict(values=[df_trades[col_i] for col_i in df_trades.columns],
                                                    fill_color=cell_fill_colors,
                                                    align=cell_font_aligns,
                                                    font=dict(size=cell_font_sizes, color=cell_font_colors)),
                                                    columnwidth=cell_widths  # Adjust this )
                                                    ),
                                secondary_y=False,
                                row=i+1,
                                col=1, 
                                )
                if verbose: print(f"performance_analytics.plots() ln385 ==> row: {row} col: {col} summary table done")
                
    # =============================================================================
    #         # SUBPLOT 3: Klines and cumulative pnl on secondary_y
    # =============================================================================
            elif type(col)==list:
                if verbose: print(f"plotting: {col} --> LIST DETECTED")
                # Plot these cols in 1 plot
                # yaxis_i = 1
                
                for col_i in col:
                    if "cum_L_pnl" in col_i or "cum_S_pnl" in col_i:
                        # Create a new y-axis for cum_L_pnl and cum_S_pnl data
                        if "cum_L_pnl" in col_i:
                            color_pnl = color_L_pnl #"green"
                        elif "cum_S_pnl" in col_i:
                            color_pnl = color_S_pnl #"orange"
                        fig.add_trace(go.Scattergl(x=df.index, y=df[col_i], name=col_i,marker = dict(color=color_pnl,size=color_pnl_size), opacity=color_pnl_opacity),
                                      secondary_y=True,
                                      )

                    elif type(col_i) == dict:
                        # if yaxis_i == 1:
                        #     yaxis = "y"
                        #     yaxis_i+=1
                        # else:
                        #     yaxis = f"y{yaxis_i}"
                        #     yaxis_i+=1

                        # if verbose: print("yaxis: ", yaxis , " ", yaxis_i)
                        klines_ax = go.Candlestick(x=df.index,
                                                    open=df[col_i["open"]],
                                                    high=df[col_i["high"]],
                                                    low=df[col_i["low"]],
                                                    close=df[col_i["close"]],
                                                    name=col_i["instrument"],
                                                    increasing_line_color=col_i["up_color"],
                                                    decreasing_line_color=col_i["down_color"],
                                                    opacity=col_i["opacity"],
                                                )

                        fig.append_trace(klines_ax,row=row,col=1)
                    elif "L_entry_price" in col_i:
                        entries_ax=go.Scattergl(x=df.index, y=df[f"L_entry_price"],mode='markers',marker_symbol= 'arrow-up',name="long entry",marker = dict(color=color_L_trade,size=13),) #name="longEntry_price"
                        fig.append_trace(entries_ax,row=row,col=1)
                    elif "L_exit_price" in col_i:
                        exits_ax= go.Scattergl(x=df.index, y=df[f"L_exit_price"],mode='markers',marker_symbol= 'arrow-down-open',name="long exit",marker = dict(color=color_L_trade,size=13)) #name="longExit_price"
                        fig.append_trace(exits_ax,row=row,col=1)
                    elif "S_entry_price" in col_i:  
                        entries_ax=go.Scattergl(x=df.index, y=df[f"S_entry_price"],mode='markers',marker_symbol= 'arrow-down',name="short entry",marker = dict(color=color_S_trade,size=13)) #name="longEntry_price"
                        fig.append_trace(entries_ax,row=row,col=1)
                    elif "S_exit_price" in col_i:
                        exits_ax= go.Scattergl(x=df.index, y=df[f"S_exit_price"],mode='markers',marker_symbol= 'arrow-up-open',name="short exit",marker = dict(color=color_S_trade,size=13)) #name="longExit_price"
                        fig.append_trace(exits_ax,row=row,col=1)
                        
                        
                    elif "L_spread" in col_i:  
                        entries_ax=go.Scattergl(x=df.index, y=df[f"L_spread"],mode='markers',marker_symbol= 'arrow-up-open',name="long spread",marker = dict(color='red',size=13)) #name="longEntry_price"
                        fig.append_trace(entries_ax,row=row,col=1)
                        
                    elif "S_spread" in col_i:
                        exits_ax= go.Scattergl(x=df.index, y=df[f"S_spread"],mode='markers',marker_symbol= 'arrow-down',name="short spread",marker = dict(color='red',size=13)) #name="longExit_price"
                        fig.append_trace(exits_ax,row=row,col=1)
                        
                    elif "__scatter" in col_i:
                        if verbose: print(f"{col_i} -> scatter plot requested")
                        col_to_plot = col_i.split("__")[0]
                        ax = go.Scattergl(x=df.index, y=df[col_to_plot],name=col_to_plot, mode='markers', marker = dict(size=4))
                        fig.append_trace(ax,row=row,col=1)
                        
                    # elif (len(df[col_i].unique()) < 4):
                    #     if verbose: print(f"{col_i} -> cat data detected")
                    #     ax = go.Scattergl(x=df.index, y=df[col_i],name=col_i, mode='markers', marker = dict(size=4))
                    #     fig.append_trace(ax,row=row,col=1)
                        
                    else:
                        try:
                            if verbose: print(f"{col_i} -> line plot")
                            # Check if there are nans in the df columns, if so then forward fill?
                            if any(df[col_i].isna()):
                                if verbose: print(f"{col_i} ---> nan detected, forward fill")
                                df[col_i].fillna(method="ffill",inplace=True)
                            ax = go.Scattergl(x=df.index, y=df[col_i],name=col_i)
                            fig.append_trace(ax,row=row,col=1)
                            fig['layout'][f'xaxis{row}']['title']=subplot_titles[row-1]
                        except Exception as e:
                            print(f"ERROR: {e}")
                            print(f"ERROR: {col_i} could not be plotted")
                            continue
                row+=1
                if verbose: print(f"performance_analytics.plots() ln475 ==> row: {row} col: {col} klines pnl trade done")
            elif "__scatter" in col:
                if verbose: print(f"{col} -> scatter plot requested")
                col_to_plot = col.split("__")[0]
                ax = go.Scattergl(x=df.index, y=df[col_to_plot],name=col_to_plot, mode='markers', marker = dict(size=4))
                fig.append_trace(ax,row=row,col=1)  
                fig['layout'][f'xaxis{row}']['title']=subplot_titles[row-1]
                row+=1
                if verbose: print(f"performance_analytics.plots() ln483 ==> row: {row} col: {col} __scatter done")
                
            else:
                # Note: plotly_resampler only supports scattergl so other go objected will not be resampled
                if verbose: print(f"performance_analytics.plots() ln484 ==> row: {row} col: {col} ALL OTHER CASES TRIGGERED?! SHOULD BE FOR zscore")
                ax = go.Scattergl(x=df.index, y=df[col],name=col)

                fig.append_trace(ax,row=row,col=1)
                fig['layout'][f'xaxis{row}']['title']=subplot_titles[row-1]
                row+=1
                if verbose: print(f"performance_analytics.plots() 493 ==> row: {row} col: {col} msic line plot done")

 
# =============================================================================
# df_summary_table
# =============================================================================




# =============================================================================
#  Plot organization
# =============================================================================            

        
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
    fig.update_layout(
    yaxis2=dict(
        title="PNL",
        overlaying="y",
        side="right",
    )
)
    
    fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0,
                                                        buttons=[create_layout_button(k,instrument) for k,instrument in enumerate(instruments)],
                                                        font=dict(color='#dfd9d3',size=15)
                                                        )],
                      xaxis_rangeslider_visible=False,
                      hovermode="x",
                      font=dict(family="Courier New, monospace",size=15,color="white"))
    
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(paper_bgcolor="#21252C", plot_bgcolor="rgb(22,26,30)")
    # fig.update_xaxes(type="date", range=window)
    fig.update_xaxes(showgrid=True,gridwidth=1,gridcolor='rgb(61,58,58)')
    fig.update_yaxes(showgrid=True,gridwidth=1,gridcolor='rgb(61,58,58)')
    fig.update_xaxes(color="grey")
    fig.update_yaxes(color="grey")
    fig.update_yaxes(fixedrange=True, secondary_y=True)
    fig.update_layout(autosize=False,
                        width=width,
                        height=height,)
    
    # ensure fig title is model instrument model name timeframe and replay_dt if applicable
    if replay_dt is not None:
        report_dt = replay_dt
    else:
        dt_now_UTC, _ = get_dt_times_now()
        report_dt = dt_now_UTC

    fig.update_layout(annotations=[dict(x=0.5,
                                        y=1.0,
                                        xref='paper',
                                        yref='paper',
                                        text=f'<b>---- Trade Dashboard ----</b><br>Model: {html_name}<br>Report datetime: {report_dt}<br>Window: {df_backtested.index[0]} to {df_backtested.index[-1]}',
                                        showarrow=False,
                                        font=dict(size=16,)
                                        )
                                    ]
                    )
    fig.update_layout(margin=dict(t=100))  # Adjust this value
    fig.update_xaxes(tickformatstops = [dict(dtickrange=[None, 1000], value="%H:%M:%S.%L ms"),
                                        dict(dtickrange=[1000, 60000], value="%H:%M:%S s"),
                                        dict(dtickrange=[60000, 3600000], value="%H:%M m"),
                                        dict(dtickrange=[3600000, 86400000], value="%H:%M h"),
                                        dict(dtickrange=[86400000, 604800000], value="%e. %b d"),
                                        dict(dtickrange=[604800000, "M1"], value="%e. %b w"),
                                        dict(dtickrange=["M1", "M12"], value="%b '%y M"),
                                        dict(dtickrange=["M12", None], value="%Y Y")
                                        ]
                    )
    fig.update_layout(autosize=False,width=width,height=height)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(hovermode="x unified")
    # fig.show()
    print(f"PUBLISHED ~ '{html_file_path}/{html_name}.html'")
    if publish: fig.write_html(f"{html_file_path}/{html_name}.html")
    return fig




























def get_plotly(filename,
               df_dict,
               window,
               cols_to_plot=None,
               row_heights:list=[1,1,1,1,1,5]):
    print("Publishing ...")
    # window = model_config["memory_len"]

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
        df=df_dict[instrument].tail(window).copy()
        if timeframes is None:
            # timeframes =sort_human(list(np.unique([tf.split("_")[0] for tf in df.filter(regex="(h_)|(d_)|(w_)").columns])))
            assert len(timeframes) <=4

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
        fig.add_trace(klines_ax, row=row, col=1, secondary_y=True,)
        
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
# =============================================================================
#  Plot organization
# =============================================================================            

        
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
