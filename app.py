import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html  # Use updated imports
from dash.dependencies import Input, Output, State

# get Bootstrap
external_stylesheets = [{
    'href':"https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css",
    'rel':"stylesheet",
    'integrity':"sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC",
    'crossorigin':"anonymous"
}]
options = [
    {'label':'All','value':'All'},
    {'label':'2020','value':2020},
    {'label':'2021','value':2021},
    {'label':'2022','value':2022},
    {'label':'2023','value':2023}
]
# FORMAT NUMBER_______________________
def f_num(num):
    if abs(num) >= 1000000:
        return f'{num/1000000:.2f}M'
    elif abs(num) >= 1000:
        return f'{num/1000:.2f}K'
    else:
        return f'{num:.2f}'

# ALL DATASET___________________


orders = pd.read_csv("Orders.csv")
product = pd.read_csv("Products.csv")
location = pd.read_csv("Location.csv")
customer = pd.read_csv("Customers.csv")
# FILTER DATA_______________

# CHANGE DATATYPE--
orders['Order Date'] = pd.to_datetime(orders['Order Date'],format = '%d-%m-%Y')
orders['Ship Date'] = pd.to_datetime(orders['Ship Date'],format = '%d-%m-%Y')
a = 1
# create local host
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Store(id='filtered-data'), # to share filter data
    html.Div([
        html.Div([html.H1('Sales Dashboard',style = {'font-size':'34px','text-align': 'left', 'margin': '.6%','font-weight':'bold','color':'#561da1','font-family': 'sans-serif'})
        ],className = 'col-md=3',style = {'background-color': '#f5f4f0','width':'78.3%','margin-right':'20px'}),

        html.Div([
                dcc.Dropdown(id='picker', options=options, value='All',
                                style={'background-color': '#f0f0f0','color': '#8b63da',
                                         'border-radius': '1px','font-size': '16px','width':'18.4vw',
                                        'justify-content': 'center','align-items': 'center'
                                }
                )
        ],className = 'col-md-3',style = {'background-color': '#f5f4f0','width':'20%',"border-radius": "1px",'padding':'8px'})
    ],className = 'row',style={'background-color':'#d0d0d0'}),
    # SECOND ROW ---------------------------------------------------------------------------------------------
    html.Div([
        html.Div([
            dcc.Graph(id = 'chartk1')
        ],className = 'col-md-3',style = {'width':'32.33%','background-color':'#f5f4f0','padding':'5px',
                                          "border-radius": "1px",'height':'22vh','margin-right':'20px'}),

        html.Div([
            dcc.Graph(id = 'chartk2')
        ],className = 'col-md-3',style = {'width':'32.15%','background-color':'#f5f4f0','padding':'5px',
                                          "border-radius": "1px",'height':'22vh','margin-right':'20px'}),

        html.Div([
            dcc.Graph(id= 'chartk3')
        ],className = 'col-md-3',style = {'width':'32.33%','background-color':'#f5f4f0','padding':'5px', "border-radius": "1px",'height':'22vh'})
    ],className = 'row',style = {'background-color':'#d0d0d0','margin-top':'20px'}),

    # Third row__________________________________
    html.Div([
        html.Div([
            dcc.Graph(id='chart1')
        ],className = 'col-md-3',style = {'width':'20%','height':'60vh','background-color':'#f5f4f0','padding':'10px'}),

        html.Div([
            dcc.Graph(id='chart2')
        ],className = 'col-md-3',style = {'width':'33.42%','height':'60vh','background-color':'#f5f4f0','padding':'10px','margin-right':'20px'}),

        html.Div([
            dcc.Graph(id = 'chart3')
        ],className = 'col-md-3',style = { 'width':'45%','height':'60vh','background-color':'#f5f4f0','padding':'10px'})

    ],className = 'row',style = {'margin-top':'20px','background-color':'#d0d0d0'})
    ],className = 'Container',style={'background-color':'#d0d0d0',"padding": "20px"})


# Shared Filtering Callback
@app.callback(
    Output('filtered-data', 'data'),
    Input('picker', 'value')
)
def filter_data(year):
    if year == 'All':
        filtered_df = orders
    else:
        filtered_df = orders[orders['Order Date'].dt.year == int(year)]
    return filtered_df.to_dict('records')


# decorator function
@app.callback([Output('chartk1','figure'),
               Output('chartk2','figure'),
               Output('chartk3','figure'),
               Output('chart1','figure'),
               Output('chart2','figure'),
               Output('chart3','figure')],
              [Input('picker','value')])
def update_graph(year):
    global select_year
    select_year = year
    if year == 'All':
        orderf = orders
    else:
        orderf = orders[orders['Order Date'].dt.year == int(year)]

# FIRST KPIS

    reindex = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Total_sale = orderf['Sales'].sum().round(1)
    Total_sale = f_num(Total_sale)
    sales_by_month = orderf.pivot_table(values='Sales', index=orderf['Order Date'].dt.month, aggfunc='sum')
    # reindexing
    sales_by_month.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sales_by_month['for_sale'] = sales_by_month['Sales'].apply(f_num)

    highest = sales_by_month['Sales'].max()
    lowest = sales_by_month['Sales'].min()

    sales_by_month['Marker_Size'] = [18 if val in [highest, lowest] else 0 for val in sales_by_month['Sales']]
    sales_by_month['Color'] = ['#07175b' if val == highest else '#8b63da' if val == lowest else '#5a636a' for val in
                               sales_by_month['Sales']]

    figk1 = px.line(sales_by_month, x=sales_by_month.index, y='Sales', markers=True, title='Total Sale')
    # markes and line operations
    figk1.update_traces(
        marker=dict(size=sales_by_month['Marker_Size'], color=sales_by_month['Color'], line=dict(width=4)),
        line=dict(color='#c8c3ca'),
        hovertemplate='Month: %{x}<br>Sales: %{customdata}<extra></extra>', customdata=sales_by_month['for_sale'])

    figk1.update_layout(title_font=dict(color='#555555', size=15,family = 'Arial'),
                      xaxis=dict(title=None, tickfont=dict(color='gray', size=10),showgrid = False),
                      yaxis=dict(title=None, showticklabels=False,showgrid = False)
                      )
    # subtitle
    figk1.update_layout(
        annotations=[dict(text=Total_sale, x=0.02, y=1.1, xref="paper", yref="paper", showarrow=False,
                          font=dict(size=22, color='#555555',family = 'Arial'))])

    figk1.update_layout(plot_bgcolor='#f5f4f0', width=401, height=144, margin=dict(l=10, r=10, t=30, b=5),
                      paper_bgcolor='#f5f4f0')


# SECOND KPIs______________________

    total_profit = orderf['Profit'].sum().round(1)
    total_profit = f_num(total_profit)

    profit_by_month = orderf.pivot_table(values='Profit', index=orderf['Order Date'].dt.month, aggfunc='sum')
    profit_by_month.index = reindex
    profit_by_month['for_profit'] = profit_by_month['Profit'].apply(f_num)

    max_profit = profit_by_month['Profit'].max()
    min_profit = profit_by_month['Profit'].min()

    profit_by_month['marker'] = [18 if val in [max_profit, min_profit] else 0 for val in profit_by_month['Profit']]
    profit_by_month['color'] = ['#07175b' if val == max_profit else '#8b63da' if val == min_profit else '#5a636a' for
                                val in profit_by_month['Profit']]

    figk2 = px.line(x=profit_by_month.index, y=profit_by_month['Profit'], title='Total Profit', markers=True)

    figk2.update_traces(marker=dict(size=profit_by_month['marker'], color=profit_by_month['color'], line=dict(width=4)),
                      line=dict(color='#c8c3ca'),
                      hovertemplate='Month: %{x}<br>Profit: %{customdata}<extra></extra>',
                      customdata=profit_by_month['for_profit'])

    figk2.update_layout(title_font=dict(color='#555555', size=15,family = 'Arial'),
                      xaxis=dict(title=None, tickfont=dict(color='gray',size = 10),showgrid = False),
                      yaxis=dict(title=None, showticklabels=False,showgrid = False))

    figk2.update_layout(
        annotations=[dict(text=total_profit, x=0.02, y=1.1, xref="paper", yref="paper", showarrow=False,
                          font=dict(color='#555555', size=22,family = 'Arial'))])

    figk2.update_layout(plot_bgcolor='#f5f4f0', width=401, height=144,margin=dict(l=10, r=10, t=30, b=5),paper_bgcolor = '#f5f4f0')

# Thired KPIs______________________

    total_qty = orderf.groupby(orderf['Order Date'].dt.month).agg({'Quantity': 'sum'})
    total_qty.index = reindex

    total_qty['for_qty'] = total_qty['Quantity'].apply(f_num)
    Total_Q = total_qty['Quantity'].sum()
    Total_QTY = f_num(Total_Q)

    max_qty = total_qty['Quantity'].max()
    min_qty = total_qty['Quantity'].min()

    total_qty['marker_size'] = [18 if val in [max_qty, min_qty] else 0 for val in total_qty['Quantity']]
    total_qty['color'] = ['#07175b' if val == max_qty else '#8b63da' if val == min_qty else '#5a636a' for val in
                          total_qty['Quantity']]

    figk3 = px.line(total_qty, x=total_qty.index, y='Quantity', markers=True, title='Total QTY')
    figk3.update_traces(marker=dict(size=total_qty['marker_size'], color=total_qty['color'], line=dict(width=4)),
                      line=dict(color='#c8c3ca'))
    figk3.update_layout(title_font=dict(color='#555555', size=15,family = 'Arial'),
                      yaxis=dict(title=None, showticklabels=False,showgrid = False),
                      xaxis=dict(title=None, tickfont=dict(color='gray',size = 10),showgrid = False))

    figk3.update_layout(
        annotations=[dict(text=Total_QTY,x = 0.02, y = 1.1, xref = "paper", yref = "paper",
                          showarrow = False, font = dict(color='#555555',size=22,family = 'Arial'))])
    figk3.update_layout(plot_bgcolor='#f5f4f0', width=401, height=144, margin=dict(l=10, r=10, t=30, b=5),paper_bgcolor = '#f5f4f0')


# FIRST CHART____________
    total_profits = pd.merge(orderf, product, how='left')

    total_profit = total_profits.pivot_table(values='Profit', index='Sub-Category', aggfunc='sum')
    total_profit['for_profit'] = total_profit['Profit'].apply(f_num)
    total_profit['color'] = ['#561da1' if val > 0 else '#07175b'for val in total_profit['Profit']]
    # Create the bar graph-------------
    fig = px.bar(total_profit, y=total_profit.index, x=total_profit['Profit'],color_discrete_sequence=[total_profit['color']])
    fig.update_traces(hovertemplate='Profit: %{customdata}<extra></extra>',customdata= total_profit['Profit'].apply(f_num))
    fig.update_layout(xaxis=dict(title=None, showticklabels=False,tickfont = dict(color = '#555555',size = 10),showgrid = False),
                      yaxis=dict(title=None, tickfont=dict(color='#555555',size = 10),showgrid = False))
    fig.update_layout(plot_bgcolor='#f5f4f0', width=240, height=395,paper_bgcolor = '#f5f4f0')
    fig.update_layout(showlegend=False,margin=dict(l=10, r=10, t=40, b=10),
                      title = 'Profit & Sales by Subcategory',title_font = dict(size = 14,family = 'Arial'))

# SECOND CHART____________________
    if year == 'All':
        orde_data = orders
    else:
        orde_data = orders[(orders['Order Date'].dt.year == int(year)) | (orders['Order Date'].dt.year == int(year) - 1)]

    orde_mer = pd.merge(orde_data, product, how='left')
    orde_pi = orde_mer.pivot_table(values = 'Sales',columns = orde_mer['Order Date'].dt.year,index ='Sub-Category' ,aggfunc = 'sum')
    orde_pio = orde_pi.sort_values(orde_pi.columns[-1],ascending = True)

    import plotly.graph_objects as go
    fig1 = go.Figure()
    # creating outer 2022 year bar
    fig1.add_trace(go.Bar(y=orde_pio.index, x= None if year in ['All',2020] else orde_pio[orde_pio.columns[0]], orientation='h',
                         marker=dict(color='#561da1', opacity=.6, line=dict(color=None)),
                         name= f"{(year-1) if isinstance(year,int) else 2020 if year == 'All' else None }",width=0.85))
    fig1.update_traces(hovertemplate='Sales: %{customdata}<extra></extra>',customdata = None if year in ['All',2020] else orde_pio[orde_pio.columns[0]].apply(f_num))
    # creating inner 2023 year bar
    fig1.add_trace(
        go.Bar(y=orde_pio.index, x= orde_pio[orde_pio.columns[-1]], orientation='h', marker=dict(color='#07175b',
                line=dict(color='#07175b')),name= f'{year}', width=0.3))
    fig1.update_traces(hovertemplate='Sales: %{customdata}<extra></extra>',customdata=orde_pio[orde_pio.columns[-1]].apply(f_num))
    # highlighted points

    fig1.update_layout(barmode='overlay',
                      xaxis=dict(title=None, tickfont=dict(color='#555555', size=10),showgrid = False),
                      yaxis=dict(title=None, tickfont=dict(color='#555555', size=10),showgrid = False),
                      height=600)
    fig1.update_layout(legend=dict(orientation="h"))
    fig1.update_layout(legend=dict(x = .6,y = 1.1),showlegend= False if year in ['All',2020] else True)

    fig1.update_layout(plot_bgcolor='#f5f4f0', width=415, height=395,margin=dict(l=10, r=10, t=40, b=10),paper_bgcolor = '#f5f4f0')

# CREATE THIRED CHART__________________________
    orderf['week'] = orderf['Order Date'].dt.isocalendar().week
    sweek = orderf.pivot_table(values='Sales', index='week', aggfunc='sum')
    sweek['for sales'] = sweek['Sales'].apply(f_num)

    pweek = orderf.pivot_table(values='Profit', index='week', aggfunc='sum')
    pweek['for profit'] = pweek['Profit'].apply(f_num)

    from plotly.subplots import make_subplots
    fig2 = make_subplots(rows=2, cols=1)

    # Chart for sales
    fig2.add_trace(go.Scatter(x=sweek.index,y=sweek['Sales'],mode='lines',line_shape='vh',line=dict(color='#561da1', width=2.5),
        hovertemplate='Week: %{x}<br>Sales: %{customdata}<extra></extra>',customdata=sweek['for sales']), row=1, col=1)

    # Add average line for sales
    average_sales = np.mean(sweek['Sales'])
    fig2.add_shape(type="line",x0=min(sweek.index),x1=max(sweek.index),y0=average_sales,y1=average_sales,
                   line=dict(color="gray", width=1.5, dash="dash"), row=1, col=1)

    # Chart for profit
    fig2.add_trace(go.Scatter(x=pweek.index,y=pweek['Profit'],mode='lines',line_shape='vh',line=dict(color='#561da1', width=2.5),
        hovertemplate='Week: %{x}<br>Profit: %{customdata}<extra></extra>',customdata=pweek['for profit']), row=2, col=1)

    # Add average line for profit
    average_profit = np.mean(pweek['Profit'])
    fig2.add_shape(type="line",x0=min(pweek.index),x1=max(pweek.index),y0=average_profit,y1=average_profit,
                   line=dict(color="gray", width=1.5, dash="dash"),row=2, col=1)


    fig2.update_layout(
        xaxis=dict(tickvals=[1, 10, 20, 30, 40, 50],tickfont=dict(size=10,family = 'Arial'),showgrid = False),
        yaxis=dict(title='Sales',tickfont=dict(size=10,family = 'Arial'),title_font = dict(size = 13,family = 'Arial'),showgrid = False),

        xaxis2=dict(tickvals=[1, 10, 20, 30, 40, 50],tickfont=dict(size=10,family = 'Arial'),showgrid = False),
        yaxis2=dict( title='Profit',tickfont=dict(size=10,family = 'Arial'),title_font = dict(size = 13,family = 'Arial'),showgrid = False),
        plot_bgcolor='#f5f4f0',width=560,height=395,margin = dict(l=10, r=10, t=40, b=10),paper_bgcolor = '#f5f4f0',
        title = 'Sales & Profit by week',title_font = dict(size = 14),showlegend = False
    )
    return figk1,figk2,figk3,fig,fig1,fig2



if __name__ == '__main__':
    app.run_server(debug=True)
