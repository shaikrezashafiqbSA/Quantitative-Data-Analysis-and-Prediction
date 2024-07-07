from plotly.subplots import make_subplots
import plotly.graph_objects as go
# from plotly_resampler import FigureWidgetResampler

class build:
    def __init__(self,
                 row_heights: list=[1,2],
                 cols_to_plot = ["index","label"],
                 height: int = 500,
                 width: int = 500,
                 resampler=True,
                 publish=False,
                 output_path = "./telegram/",
                 output_name = "plotly_studies",
                 subplot_titles = None,
                 convert_tz = {"from":"UTC", "to":"Asia/Singapore"},
                 verbose=False,
                ):
        self.height = height
        self.width = width
        self.number_subplots = len(cols_to_plot)
        self.row_heights = row_heights
        self.cols_to_plot = cols_to_plot
        self.publish = publish
        self.output_path = output_path
        self.output_name = output_name
        self.subplot_titles = subplot_titles
        self.convert_tz = convert_tz
        self.verbose = verbose
        if resampler:
            pass
            # self.fig = FigureWidgetResampler(make_subplots(rows=self.number_subplots,
            #                                                cols=1, 
            #                                                shared_xaxes = True, 
            #                                                vertical_spacing = 0.05,
            #                                                row_heights = self.row_heights, 
            #                                                specs =[[{"type":"scatter"}]]*self.number_subplots,
            #                                                subplot_titles = self.subplot_titles
            #                                               ))
        else:
            self.fig = make_subplots(rows=self.number_subplots,
                                                           cols=1, 
                                                           shared_xaxes = True, 
                                                           vertical_spacing = 0.05,
                                                           row_heights = self.row_heights, 
                                                           specs =[[{"type":"scatter"}]]*self.number_subplots,
                                                           subplot_titles = self.subplot_titles,
                                                          )
        
        
    def plot(self,df0):
        df = df0.copy()
        if self.convert_tz is not None:
            if self.verbose: print(f"CONVERTING TIMEZONE: {self.convert_tz['from']} ----> {self.convert_tz['to']}")
            if self.verbose: print(f"initially: {df.index[-1]}")
            df.index = df.index.tz_localize(self.convert_tz["from"])#.tz_convert('UTC')
            df.index = df.index.tz_convert(self.convert_tz["to"])
            if self.verbose: print(f"now: {df.index[-1]}")
            df.index = df.index.tz_localize(None)
            if self.verbose: print(f"now: {df.index[-1]}")
            
        row=1    
        for col in self.cols_to_plot:
            if self.verbose: print(f"plotting: {col}")
            if type(col)==list:
                if self.verbose: print(f"plotting: {col} --> LIST DETECTED")
                # Plot these cols in 1 plot
                yaxis_i = 1
                
                for col_i in col:
                    if type(col_i) == dict:
                        if yaxis_i == 1:
                            yaxis = "y"
                            yaxis_i+=1
                        else:
                            yaxis = f"y{yaxis_i}"
                            yaxis_i+=1

                        if self.verbose: print("yaxis: ", yaxis , " ", yaxis_i)
                        klines_ax = go.Candlestick(x=df.index,
                                                   open=df[col_i["open"]],
                                                   high=df[col_i["high"]],
                                                   low=df[col_i["low"]],
                                                   close=df[col_i["close"]],
                                                   name=col_i["instrument"],
                                                   increasing_line_color=col_i["up_color"],
                                                   decreasing_line_color=col_i["down_color"],
                                                   opacity = col_i["opacity"],yaxis=yaxis)
                        self.fig.append_trace(klines_ax,row=row,col=1)
                    elif "L_entry_price" in col_i:
                        entries_ax=go.Scattergl(x=df.index, y=df[f"L_entry_price"],mode='markers',marker_symbol= 'arrow-up',name="long entry",marker = dict(color='blue',size=13)) #name="longEntry_price"
                        self.fig.append_trace(entries_ax,row=row,col=1)
                    elif "L_exit_price" in col_i:
                        exits_ax= go.Scattergl(x=df.index, y=df[f"L_exit_price"],mode='markers',marker_symbol= 'arrow-down-open',name="long exit",marker = dict(color='blue',size=13)) #name="longExit_price"
                        self.fig.append_trace(exits_ax,row=row,col=1)
                    elif "S_entry_price" in col_i:  
                        entries_ax=go.Scattergl(x=df.index, y=df[f"S_entry_price"],mode='markers',marker_symbol= 'arrow-down',name="short entry",marker = dict(color='black',size=13)) #name="longEntry_price"
                        self.fig.append_trace(entries_ax,row=row,col=1)
                    elif "S_exit_price" in col_i:
                        exits_ax= go.Scattergl(x=df.index, y=df[f"S_exit_price"],mode='markers',marker_symbol= 'arrow-up-open',name="short exit",marker = dict(color='black',size=13)) #name="longExit_price"
                        self.fig.append_trace(exits_ax,row=row,col=1)
                        
                        
                    elif "L_spread" in col_i:  
                        entries_ax=go.Scattergl(x=df.index, y=df[f"L_spread"],mode='markers',marker_symbol= 'arrow-up-open',name="long spread",marker = dict(color='red',size=13)) #name="longEntry_price"
                        self.fig.append_trace(entries_ax,row=row,col=1)
                        
                    elif "S_spread" in col_i:
                        exits_ax= go.Scattergl(x=df.index, y=df[f"S_spread"],mode='markers',marker_symbol= 'arrow-down',name="short spread",marker = dict(color='red',size=13)) #name="longExit_price"
                        self.fig.append_trace(exits_ax,row=row,col=1)
                        
                    elif "__scatter" in col_i:
                        if self.verbose: print(f"{col_i} -> scatter plot requested")
                        col_to_plot = col_i.split("__")[0]
                        ax = go.Scattergl(x=df.index, y=df[col_to_plot],name=col_to_plot, mode='markers', marker = dict(size=4))
                        self.fig.append_trace(ax,row=row,col=1)
                        
                    # elif (len(df[col_i].unique()) < 4):
                    #     if self.verbose: print(f"{col_i} -> cat data detected")
                    #     ax = go.Scattergl(x=df.index, y=df[col_i],name=col_i, mode='markers', marker = dict(size=4))
                    #     self.fig.append_trace(ax,row=row,col=1)
                        
                    else:
                        if self.verbose: print(f"{col_i} -> line plot")
                        # Check if there are nans in the df columns, if so then forward fill?
                        if any(df[col_i].isna()):
                            if self.verbose: print(f"{col_i} ---> nan detected, forward fill")
                            df[col_i].fillna(method="ffill",inplace=True)
                        ax = go.Scattergl(x=df.index, y=df[col_i],name=col_i)
                        self.fig.append_trace(ax,row=row,col=1)
                row+=1
                
            elif "__scatter" in col:
                if self.verbose: print(f"{col} -> scatter plot requested")
                col_to_plot = col.split("__")[0]
                ax = go.Scattergl(x=df.index, y=df[col_to_plot],name=col_to_plot, mode='markers', marker = dict(size=4))
                self.fig.append_trace(ax,row=row,col=1)  
                row+=1
                
            # elif (len(df[col].unique()) < 4):
            #     if self.verbose: print(f"{col} -> cat data detected")
            #     ax = go.Scattergl(x=df.index, y=df[col],name=col, mode='markers', marker = dict(size=4))
            #     self.fig.append_trace(ax,row=row,col=1)
            #     row+=1
            else:
                # Note: plotly_resampler only supports scattergl so other go objected will not be resampled
                ax = go.Scattergl(x=df.index, y=df[col],name=col)
                self.fig.append_trace(ax,row=row,col=1)
                row+=1

        self.fig.update_layout(autosize=False,width=self.width,height=self.height)
        self.fig.update_layout(xaxis_rangeslider_visible=False)
        self.fig.update_xaxes(rangeslider_visible=False)
        self.fig.update_layout(hovermode="x unified")
        # self.fig.update_traces(connectgaps=False)
        if self.publish: self.fig.write_html(f"{self.output_path}{self.output_name}.html")
        return self.fig
    

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
