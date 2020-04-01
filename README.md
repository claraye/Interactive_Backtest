# Interactive_Backtest
An interactive backtesting engine, applicable for any trading strategy given PnLs or portfolio values.

- Example included: \
backtest_MFT.ipynb

- with metrics:\
  Highest_Value\
  Lowest_Value\
  Final_Value\
  Total_Return\
  Total_Return_(Annualized)\
  Volatility_(Annualized)\
  Max_Drawdown\
  Avg_Drawdown\
  Max_DD_Duration\
  Sharpe_Ratio\
  Sortino_Ratio\
  Relative_Return\
  Relative_Vol\
  Relative_Max_DD\
  Relative_Avg_DD \
  Information_Ratio

- with output like:\
![Sample backtest output](https://github.com/claraye/Interactive_Backtest/blob/master/backtest_MFT_sample.png)

- Added tools to enable interactive functionailities (using the Bokeh package):
  1. Pan tool: allows the user to pan the plot by left-dragging a mouse across the plot region. 
  2. Box zoom tool: allows the user to zoom the plot bounds of a rectangular region by left-dragging the mouse.
  3. Wheel zoom tool: allows the user to zoom the plot in and out, centered on the current mouse location.
  4. Crosshair tool: draws a crosshair annotation over the plot, centered on the current mouse position.
  5. Hover tool: displays informational tooltips whenever the cursor is directly over a graph.
