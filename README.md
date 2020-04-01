# Interactive_Backtest
An interactive backtesting engine, applicable for any trading strategy given PnLs or portfolio values.

- Example included: backtest_MFT.ipynb\
with output like:\
![Sample backtest output](https://github.com/claraye/Interactive_Backtest/blob/master/backtest_MFT_sample.png)

Add tools to enable interactive functionailities (using the Bokeh package):
1. Pan tool: allows the user to pan the plot by left-dragging a mouse across the plot region. 
2. Box zoom tool: allows the user to zoom the plot bounds of a rectangular region by left-dragging the mouse.
3. Wheel zoom tool: allows the user to zoom the plot in and out, centered on the current mouse location.
4. Crosshair tool: draws a crosshair annotation over the plot, centered on the current mouse position.
5. Hover tool: displays informational tooltips whenever the cursor is directly over a graph.
