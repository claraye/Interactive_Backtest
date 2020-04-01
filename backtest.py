# -*- coding: utf-8 -*-
#reference: backtesting module developed by kernc
"""
Created on Sat Nov 28 10:59:45 2019

@author: qy
"""

import pandas as pd
import numpy as np

import os
import warnings
warnings.filterwarnings("ignore")

from functools import partial
import pandas_datareader.data as web
import yfinance as yf
from pandas.tseries.offsets import BDay, Minute
from pytz import timezone

from bokeh.plotting import figure as _figure
from bokeh.models import (
    #CustomJS,
    ColumnDataSource,
    NumeralTickFormatter,
    #Span,
    HoverTool,
    Range1d,
    #DateFormatter,
    #DatetimeTickFormatter,
    #FuncTickFormatter,
    WheelZoomTool
)
from bokeh.io import output_notebook, output_file, show
from bokeh.io.state import curstate
from bokeh.layouts import gridplot

IS_JUPYTER_NOTEBOOK = 'JPY_PARENT_PID' in os.environ
if IS_JUPYTER_NOTEBOOK:
    warnings.warn('Jupyter Notebook detected. '
                  'Setting Bokeh output to notebook. '
                  'This may not work in Jupyter clients without JavaScript '
                  'support (e.g. PyCharm, Spyder IDE). '
                  'Reset with `bokeh.io.reset_output()`.')
    output_notebook()
    
    
def _bokeh_reset(filename=None):
    curstate().reset()
    # Test if we are in Jupyter notebook
    if IS_JUPYTER_NOTEBOOK:
        curstate().output_notebook()
    elif filename:
        if not filename.endswith('.html'):
            filename += '.html'

        output_file(filename, title=filename)
        print('OUTPUT FILE.')
        

def strategy_plot(performance, benchmark='^GSPC', show_legend=True, filename='TestPlot', freq='day'):
    _bokeh_reset(filename)
    plots = []
    #values = performance['values_data'].iloc[1:]
    values = performance['values_data']
    drawdowns = performance['drawdowns_data']
    if benchmark:
        benchmark_values = performance['benchmark_values_data']
    is_datetime_index = values.index.is_all_dates
    index = values.index
    
    # prepration for maximum drawdown duration line
    mdd_idx = drawdowns.idxmin()
    try:
        dd_start_idx = values[:mdd_idx].idxmax()
    except Exception:
        # ValueError: attempt to get argmax of an empty sequence
        dd_start_idx = dd_end_idx = index[0]
        dd_duration = 0
    else:
        dd_end_idx = (values[mdd_idx:] > values[dd_start_idx]).idxmax()
        if dd_end_idx == mdd_idx:    # adjust the ending index
            dd_end_idx = index[-1]
        dd_duration = dd_end_idx - dd_start_idx
    
    new_bokeh_figure = partial(
        _figure,
        x_axis_type='datetime' if is_datetime_index else 'linear',
        plot_height=600,
        tools="xpan,xwheel_zoom,box_zoom,undo,redo,reset,crosshair,save",
        active_drag='xpan',
        active_scroll='xwheel_zoom')   
    
    # Set padding for the start/end of the x-axis in plots
    pad = (index[-1] - index[0]) / 50
    
    #print(index[0] - pad)
    def new_indicator_figure(**kwargs):
        kwargs.setdefault('plot_height', 90)
        fig = new_bokeh_figure(x_range=Range1d(index[0], index[-1], bounds=(index[0] - pad, index[-1] + pad) if index.size > 1 else None),
                               active_drag='xpan', active_scroll='xwheel_zoom', **kwargs)
        fig.xaxis.visible = True
        fig.yaxis.minor_tick_line_color = None
        return fig
    
    def set_tooltips(fig, tooltips=(), vline=True, renderers=(), show_arrow=True):
        tooltips = list(tooltips)
        renderers = list(renderers)
    
        if is_datetime_index:
            if freq=='day':
                formatters = dict(index='datetime')
                tooltips = [("Date", "@index{%F}")] + tooltips
            elif freq=='intraday':
                formatters = dict(index='datetime')
                tooltips = [("Datetime", "@index{%H:%M:%S}")] + tooltips
        else:
            formatters = {}
            tooltips = [("#", "@index")] + tooltips
        fig.add_tools(HoverTool(
            point_policy='follow_mouse',
            renderers=renderers, formatters=formatters, show_arrow=show_arrow,
            tooltips=tooltips, mode='vline' if vline else 'mouse'))
    
    fig = new_indicator_figure(y_axis_label="portfolio", **(dict(plot_height=300)))
    
    # High-watermark drawdown dents
    fig.patch('index', 'portfolio_dd',
              source=ColumnDataSource(dict(
                  index=np.r_[index, index[::-1]],
                  portfolio_dd=np.r_[values, values.cummax()[::-1]])),
              fill_color='#ffffea', line_color='#ffcb66')

    # Portfolio line
    r = fig.line('index', 'portfolio', 
                 source=ColumnDataSource(dict(
                         index=index, portfolio=values.values)), 
                 line_width=1.5, line_alpha=1)
    tooltip_format = '@portfolio{$ 0,0}'
    tick_format = '$ 0.000 a'
    set_tooltips(fig, [('portfolio', tooltip_format)], renderers=[r])
    fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)
    
    if benchmark:
        # Benchmark line
        fig.line('index', 'benchmark', 
                 source=ColumnDataSource(dict(
                         index=index, benchmark=benchmark_values.values)), 
                 line_width=1.5, line_alpha=1, color='black')
    
    # Peaks and Finals
    max_value_idx = values.idxmax()
    fig.scatter(max_value_idx, values[max_value_idx],
                legend='Peak (${:,.2f})'.format(values[max_value_idx]),
                color='green', size=8)
    fig.scatter(index[-1], values.iloc[-1],
                legend='Final (${:,.2f})'.format(values.iloc[-1]),
                color='blue', size=8)
    mdd_idx = drawdowns.idxmin()
    fig.scatter(mdd_idx, values[mdd_idx],
                legend='Max Drawdown ({:.2f}%)'.format(performance.Max_Drawdown*100),
                color='red', size=8)

    fig.line([dd_start_idx, dd_end_idx], values[dd_start_idx],
             line_color='red', line_width=2,
             legend='Max DD Duration ({})'.format(dd_duration).replace(' 00:00:00', '').replace('0 days ', ''))
    
    plots.append(fig)
    
    for f in plots:
        if f.legend:
            f.legend.location = 'top_left' if show_legend else None
            f.legend.border_line_width = 1
            f.legend.border_line_color = '#333333'
            f.legend.padding = 5
            f.legend.spacing = 0
            f.legend.margin = 0
            f.legend.label_text_font_size = '8pt'
        f.min_border_left = 0
        f.min_border_top = 3
        f.min_border_bottom = 6
        f.min_border_right = 10
        f.outline_line_color = '#666666'

        wheelzoom_tool = next(wz for wz in f.tools if isinstance(wz, WheelZoomTool))
        wheelzoom_tool.maintain_focus = False

    fig = gridplot(
        plots,
        ncols=1,
        toolbar_location='right',
        toolbar_options=dict(logo=None),
        merge_tools=True
    )
    show(fig)
    return fig


# Calculate the performance for a medium-frequency strategy
def performance_analysis_MF(pnls_or_values, initial_cash=1, benchmark='^GSPC', risk_free='^IRX', mar=0.0, input_type='value'):
    if input_type == 'value':
        values_df = pd.Series(pnls_or_values)
    elif input_type == 'pnl':
        values_df = pd.Series(pnls_or_values).cumsum() + initial_cash
        
    values_df.index = pd.to_datetime(values_df.index)   
    start_date = values_df.index[0]
    end_date = values_df.index[-1]
    
    # add the initial portfolio values
    values_df = pd.concat([pd.Series([initial_cash], index=[start_date+BDay(-1)]), 
                           values_df])
    
    # calc the daily returns
    returns_df = (values_df - values_df.shift(1)) / values_df.shift(1)
    returns_df = returns_df.dropna()
    
    # calc the annualized return
    cum_return = values_df.iloc[1:] / initial_cash - 1
    annual_returns_df = (cum_return + 1)**(252/np.array(range(1, len(returns_df)+1))) - 1
    
    # calc the annualized volatility
    annual_vol = returns_df.std() * np.sqrt(252)
    
    # calc the Sharpe ratio / sortino ratio
    if risk_free:
        # get the risk-free prices
        RF_quotes = web.DataReader(risk_free, 'yahoo', start_date+BDay(-1), end_date)['Close']
        # get the expected risk-free rate
        risk_free = np.mean(1/(1-RF_quotes*0.01)-1)
    else:
        risk_free = 0.0
        
    daily_risk_free = risk_free / 252
    daily_mar = mar / 252
    sharpe_ratio = (returns_df - daily_risk_free).mean() / (returns_df - daily_risk_free).std() *252**0.5
    sortino_ratio = (returns_df.mean() - daily_mar) / (returns_df[returns_df < daily_mar]).std() *252**0.5
    #sharpe_ratio = (returns_df.mean()*252 - risk_free) / ((returns_df - daily_risk_free).std()*252**0.5)
    #sortino_ratio = (returns_df.mean()*252 - mar) / ((returns_df[returns_df < daily_mar]).std()*252**0.5)
    
    # calc the maximum drawdown
    cum_max_value = (1+cum_return).cummax()
    drawdowns = ((1+cum_return) - cum_max_value) / cum_max_value
    max_drawdown = np.min(drawdowns)
    avg_drawdown = drawdowns.mean()
    
    if benchmark:
        # get the benchmark prices
        benchmark_prices = web.DataReader(benchmark, 'yahoo', start_date+BDay(-1), end_date)['Close']
        print(benchmark_prices.shape)
        # calc the benchmark daily returns
        benchmark_returns = (benchmark_prices - benchmark_prices.shift(1)) / benchmark_prices.shift(1)
        benchmark_returns = benchmark_returns.dropna()
        # calc the benchmark annualized return
        benchmark_cum_return = np.exp(np.log1p(benchmark_returns).cumsum()) - 1
        benchmark_annual_returns = (benchmark_cum_return + 1)**(252/np.array(range(1, len(benchmark_returns)+1))) - 1
        # calc the benchmark values based on the same initial_cash of portfolio
        benchmark_values = pd.concat([pd.Series([initial_cash], index=[start_date+BDay(-1)]), 
                                      initial_cash * (1+benchmark_cum_return)])
        # calc the benchmark annualized volatility
        benchmark_annual_vol = benchmark_returns.std() * np.sqrt(252)
        # calc the maximum drawdown
        benchmark_cum_max_value = (1+benchmark_cum_return).cummax()
        benchmark_drawdowns = ((1+benchmark_cum_return) - benchmark_cum_max_value) / benchmark_cum_max_value
        benchmark_max_drawdown = np.min(benchmark_drawdowns)
        benchmark_avg_drawdown = benchmark_drawdowns.mean()

        # compare with the benchmark
        relative_return = annual_returns_df.iloc[-1] - benchmark_annual_returns.iloc[-1]
        relative_vol = annual_vol - benchmark_annual_vol
        relative_max_drawdown = max_drawdown - benchmark_max_drawdown
        relative_avg_drawdown = avg_drawdown - benchmark_avg_drawdown
        excess_return_std = (returns_df - benchmark_returns).std() * np.sqrt(252)
        info_ratio = relative_return / excess_return_std
    
    # organize the output
    performance = pd.Series()
    performance.loc['Begin'] = start_date
    performance.loc['End'] = end_date
    performance.loc['Duration'] = performance.End - performance.Begin
    performance.loc['Initial_Value'] = initial_cash
    performance.loc['Highest_Value'] = np.max(values_df)
    performance.loc['Lowest_Value'] = np.min(values_df)
    performance.loc['Final_Value'] = values_df.iloc[-1]
    performance.loc['Total_Return'] = performance['Final_Value'] / performance['Initial_Value'] - 1
    performance.loc['Total_Return_(Annualized)'] = annual_returns_df.iloc[-1]
    performance.loc['Volatility_(Annualized)'] = annual_vol
    performance.loc['Max_Drawdown'] = max_drawdown
    performance.loc['Avg_Drawdown'] = avg_drawdown
    performance.loc['Sharpe_Ratio'] = sharpe_ratio
    performance.loc['Sortino_Ratio'] = sortino_ratio
    if benchmark:
        performance.loc['Relative_Return'] = relative_return
        performance.loc['Relative_Vol'] = relative_vol
        performance.loc['Relative_Max_DD'] = relative_max_drawdown
        performance.loc['Relative_Avg_DD'] = relative_avg_drawdown
        performance.loc['Information_Ratio'] = info_ratio
    
    print(performance)
    performance.loc['values_data'] = values_df
    performance.loc['returns_data'] = returns_df
    performance.loc['annual_returns_data'] = annual_returns_df
    performance.loc['drawdowns_data'] = drawdowns
    if benchmark:
        performance.loc['benchmark_values_data'] = benchmark_values
    
    strategy_plot(performance)
    
    return performance


def performance_analysis_HF(pnls_or_values, initial_cash=1, benchmark='^GSPC', risk_free='^IRX', mar=0.0, input_type='value'):
    if input_type == 'value':
        values_df = pd.Series(pnls_or_values)
    elif input_type == 'pnl':
        values_df = pd.Series(pnls_or_values).cumsum() + initial_cash
    
    values_df.index = pd.to_datetime(values_df.index)
    start_time = values_df.index[0]
    end_time = values_df.index[-1]
    
    # downsample the series into 3 min bins
    values_df_3T = values_df.resample('3T', label='right').last()
    snapshot_per_day = (16-9.5) * 60/3
    
    # add the initial portfolio values
    values_df = pd.concat([pd.Series([initial_cash], index=[start_time+Minute(-3)]), 
                           values_df_3T])
    
    # calc the 3-minute returns
    returns_df = (values_df - values_df.shift(1)) / values_df.shift(1)
    returns_df = returns_df.dropna()
    
    # calc the daily return
    cum_return = values_df.iloc[1:] / initial_cash - 1
    annual_returns_df = (cum_return + 1)**(252*snapshot_per_day/np.array(range(1, len(returns_df)+1))) - 1
    
    # calc the daily volatility
    annual_vol = returns_df.std() * np.sqrt(252*snapshot_per_day)
    
    # calc the Sharpe ratio / sortino ratio
    if risk_free:
        # get the risk-free prices
        RF_quotes = web.DataReader(risk_free, 'yahoo', start_time+Minute(-3), end_time)['Close']
        # get the expected risk-free rate
        risk_free = np.mean(1/(1-RF_quotes*0.01)-1)
    else:
        risk_free = 0.0
    
    # calc the Sharpe ratio / sortino ratio
    risk_free_per_snapshot = risk_free / 252 / snapshot_per_day
    mar_per_snapshot = mar / 252 / snapshot_per_day
    sharpe_ratio = (returns_df.mean() - risk_free_per_snapshot) / (returns_df - risk_free_per_snapshot).std() \
                    * (252*snapshot_per_day)**0.5
    sortino_ratio = (returns_df.mean() - mar_per_snapshot) / (returns_df[returns_df < mar_per_snapshot]).std() \
                    * (252*snapshot_per_day)**0.5
    
    # calc the maximum drawdown
    cum_max_value = (1+cum_return).cummax()
    drawdowns = ((1+cum_return) - cum_max_value) / cum_max_value
    max_drawdown = np.min(drawdowns)
    avg_drawdown = drawdowns.mean()

    if benchmark:
        start_time = values_df.index[0].replace(tzinfo=timezone('America/New_York'))
        end_time = values_df.index[-1].replace(tzinfo=timezone('America/New_York'))
        # get the benchmark prices
        benchmark_prices = yf.download(benchmark, start=start_time.strftime('%Y-%m-%e'), 
                                       end=(end_time+BDay(1)).strftime('%Y-%m-%e'), 
                                       interval="1m")['Close']
        benchmark_prices_3T = benchmark_prices.resample('3T', label='right').last()[values_df_3T.index]
        benchmark_prices = pd.concat([pd.Series([benchmark_prices.iloc[-1]], index=[start_time+Minute(-3)]), 
                                      benchmark_prices_3T])
        # calc the benchmark daily returns
        benchmark_returns = (benchmark_prices - benchmark_prices.shift(1)) / benchmark_prices.shift(1)
        benchmark_returns = benchmark_returns.dropna()
        # calc the benchmark daily return
        benchmark_cum_return = np.exp(np.log1p(benchmark_returns).cumsum()) - 1
        benchmark_annual_returns = (benchmark_cum_return + 1)**(252*snapshot_per_day/np.array(range(1, len(benchmark_cum_return)+1))) - 1
        # calc the benchmark values based on the same initial_cash of portfolio
        benchmark_values = pd.concat([pd.Series([initial_cash], index=[start_time+Minute(-3)]), 
                                      initial_cash * (1+benchmark_cum_return)])
        benchmark_values.index = [local_time.replace(tzinfo=timezone('UTC')) for local_time in benchmark_values.index]
        # calc the benchmark daily volatility
        benchmark_annual_vol = benchmark_returns.std() * np.sqrt(252*snapshot_per_day)
        # calc the maximum drawdown
        benchmark_cum_max_value = (1+benchmark_cum_return).cummax()
        benchmark_drawdowns = ((1+benchmark_cum_return) - benchmark_cum_max_value) / benchmark_cum_max_value
        benchmark_max_drawdown = np.min(benchmark_drawdowns)
        benchmark_avg_drawdown = benchmark_drawdowns.mean()

        # compare with the benchmark
        relative_return = annual_returns_df.iloc[-1] - benchmark_annual_returns.iloc[-1]
        relative_vol = annual_vol - benchmark_annual_vol
        relative_max_drawdown = max_drawdown - benchmark_max_drawdown
        relative_avg_drawdown = avg_drawdown - benchmark_avg_drawdown
        excess_return_std = (returns_df - benchmark_returns).std() * np.sqrt(252*snapshot_per_day)
        info_ratio = relative_return / excess_return_std
        
    # organize the output
    performance = pd.Series()
    performance.loc['Begin'] = start_time
    performance.loc['End'] = end_time
    performance.loc['Duration'] = performance.End - performance.Begin
    performance.loc['Initial_Value'] = initial_cash
    performance.loc['Highest_Value'] = np.max(values_df)
    performance.loc['Lowest_Value'] = np.min(values_df)
    performance.loc['Final_Value'] = values_df.iloc[-1]
    performance.loc['Total_Return'] = performance['Final_Value'] / performance['Initial_Value'] - 1
    performance.loc['Total_Return_(Annual)'] = annual_returns_df.iloc[-1]
    performance.loc['Volatility_(Annual)'] = annual_vol
    performance.loc['Max_Drawdown'] = max_drawdown
    performance.loc['Avg_Drawdown'] = avg_drawdown
    performance.loc['Sharpe_Ratio_(Annual)'] = sharpe_ratio
    performance.loc['Sortino_Ratio_(Annual)'] = sortino_ratio

    if benchmark:
        performance.loc['Relative_Return_(Annual)'] = relative_return
        performance.loc['Relative_Vol_(Annual)'] = relative_vol
        performance.loc['Relative_Max_DD'] = relative_max_drawdown
        performance.loc['Relative_Avg_DD'] = relative_avg_drawdown
        performance.loc['Information_Ratio_(Annual)'] = info_ratio
    
    print(performance)
    values_df.index = pd.Series([local_time.replace(tzinfo=timezone('UTC')) for local_time in values_df.index])
    drawdowns.index = pd.Series([local_time.replace(tzinfo=timezone('UTC')) for local_time in drawdowns.index])
    performance.loc['values_data'] = values_df
    performance.loc['returns_data'] = returns_df
    performance.loc['annual_returns_data'] = annual_returns_df
    performance.loc['drawdowns_data'] = drawdowns
    if benchmark:
        performance.loc['benchmark_values_data'] = benchmark_values
    
    strategy_plot(performance, benchmark, freq='intraday')

    return performance


if __name__ == '__main__':
    pnls_df = pd.read_csv(r'.\rsi_values_5y.csv', header=None, index_col=0)[1]
    initial_cash = 1000000
    performance = performance_analysis_MF(pnls_df, initial_cash)
    
    
    
    