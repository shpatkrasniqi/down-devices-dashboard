from turtle import title
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

from datetime import date
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import pathlib

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
app.title = "Down Devices Dashboard"

#Filter stored the data
def filter_df(df, startDate=None, endDate=None, buildings=None, areas=None, apis=None):
    """
    :param: df: DataFrame to be filtered.
    :param: startDate: start date from selection.
    :param: endDate: end date from selection.
    :param: buildings: building from selection.
    :param: area: area from heatmap.
    :param: apis: data source from selection.

    :return: filtered DataFrame.
    """

    dff = df.copy()

    #filter date
    if (startDate == None) or (endDate == None) :
        pass
    else:
        selsected_s = pd.Timestamp(startDate)
        selected_e = pd.Timestamp(endDate)

        selected_inter = pd.Interval(selsected_s,selected_e, closed='both')

        interval = pd.IntervalIndex.from_arrays(dff.startTime, dff.endTime, closed='both')

        dff = dff[interval.overlaps(selected_inter)]

    #filter building
    if buildings == None:
        pass
    else:
        dff = dff.loc[dff['buildingName'].isin([buildings])] 

    #filter area
    if areas == None:
        pass
    else:
        dff = dff.loc[dff['area'].isin([areas])] #filter area

    #filter data source
    if apis == None:
        pass
    else:
        dff = dff.loc[dff['dataSource'].isin([apis])] 

    return dff

#Build dbc Cards
def card_builder(header, myId):
    """
    :param: header: card header text.
    :param: myId: card ID.

    :return: Card content.
    """
    card_content = [
        dbc.CardHeader(header),
        dbc.CardBody(html.H5(id=myId, className="card-text")),
    ]
    return card_content

#Find Score
def score_finder(dff):
    """
    :param: df: DataFrame to be used.

    :return: Average Down Days in percentage (score).
    """
    start = dff['startTime'].min()
    end = dff['endTime'].max()

    avg_down_days = (dff.loc[dff['status'].isin(['down'])]['durationH'].mean()) / 24
    days = (end - start) / np.timedelta64(1,'D')

    avg_down_days_p = (avg_down_days / days)*100

    avg_down_days_p = format(avg_down_days_p,".2f")
    if np.isnan(avg_down_days):
        return 'No Data'
    else:
        return  f'{avg_down_days_p} %'

#Get down time percentages
def get_dt_percent(df_intervals,df):

    """
    :param: df_intervals: intervals created from DataFrame.
    :param: df: DataFrame to be Used.

    :return: list of down time percentages and list of df_intervals.
    """

    dff = df.copy()
    interval = pd.IntervalIndex.from_arrays(dff.startTime, dff.endTime, closed='both')
    
    perc_list = []
    inter_list = []
    
    for i in df_intervals:
        dff2 = dff[interval.overlaps(i)]
        start = dff2['startTime'].min()
        end = dff2['endTime'].max()
        
        avg_down_days = (dff2.loc[dff2['status'].isin(['down'])]['durationH'].mean()) / 24
        days = (end - start) / np.timedelta64(1,'D')
        avg_down_days_p = ((avg_down_days / days)*100).round(2)
        
        inter_list.append(str(i))
        perc_list.append(avg_down_days_p)
    
    return perc_list, inter_list

BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

#Read Data
df = pd.read_csv(DATA_PATH.joinpath('device_down_data_KL.csv'))

df.startTime = df.startTime.astype('datetime64')
df.endTime = df.endTime.astype('datetime64')

buildings = df['buildingName'].unique()
maxDate = df['endTime'].max().date()
minDate = df['startTime'].min().date()



app.layout = html.Div([
    dcc.Store(id='stored-data', storage_type='memory'),
    html.Div([
        dbc.Container([
            dbc.Row([
                html.Label('Select date:'),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=minDate,
                    max_date_allowed=maxDate,
                    initial_visible_month=date(2021, 11, 1),
                    display_format='DD-MM-YYYY',
                    start_date=minDate,
                    end_date=maxDate
                    ),
                ]),
            html.Br(),
        
            dbc.Row([
                html.Label('Select building:'),
                dcc.Dropdown(id='building-dropdown', options=buildings, value=buildings[0]),
                ]),

            html.Br(),

            dbc.Row([
                html.Label('Select data source:'),
                dcc.Dropdown(id='ds-dropdown'),
                ]),

            html.Br(),

            dbc.Row([
                html.Label('Select area:'),
                dcc.Dropdown( id='area-dropdown'),
                ]),
        ]),
    ], style={'padding':10, 'flex':2}),

    html.Div([
        dbc.Container([
            dbc.Row(html.H3('DOWN DEVICES DASHBOARD')),
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(card_builder('Average Down Time', 'adt-output'))),
                dbc.Col(dbc.Card(card_builder('Average Down Time %', 'adtP-output'))),
                dbc.Col(dbc.Card(card_builder('Current Communication Status', 'ccs-output'))),
                dbc.Col(dbc.Card(card_builder('Building Score', 'bs-output'))),
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col(dbc.Card(card_builder('Data Source Communication Score', 'dscs-output'))),
                dbc.Col(dbc.Card(card_builder('Area Communication Score','acs-output'))),
                dbc.Col(dbc.Card(card_builder('Average of communication switch status', 'switch-output'))),
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.H5('Device Type Communication Score:'),
                    dash_table.DataTable(id='industrial-t-output',page_size=10,style_cell={'textAlign': 'left'})]),
                dbc.Col(dcc.Graph(id='graph')),
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.H5('Data Source Table:'),
                    dash_table.DataTable(id='ds-building-t-output',page_size=10,style_cell={'textAlign': 'left'})]),
            ]),

            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    html.H5('Down Devices Table:'),
                    dash_table.DataTable(id='down-device-t-output',page_size=15,style_cell={'textAlign': 'left'},)]),
            ]),
        ])
    ], style={'padding':10, 'flex':8}),

], style={'display':'flex', 'flex-direction':'row'})
    

#CALLBACKS
#------------------------------------------------------------------------------


#Store Data
@app.callback(
    Output('stored-data', 'data'),

    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('building-dropdown', 'value'),
    Input('ds-dropdown', 'value'),
    Input('area-dropdown', 'value')
)
def store_data(start_date, end_date, building, dataSource, area ):
    #Stores data taken form all the inputs
    data_dict={}
    data_dict['startDate'] = start_date
    data_dict['endDate'] = end_date
    data_dict['building'] = building
    data_dict['dataSource'] = dataSource
    data_dict['area'] = area

    return data_dict

#Avg Downtime and downtime in percentage
@app.callback(
    Output('adt-output', 'children'),
    Output('adtP-output', 'children'),

    Input('stored-data', 'data')
)
def update_adt_output(data):

    dff = filter_df(df, data['startDate'], data['endDate'])

    end = dff['endTime'].max()
    start = dff['startTime'].min()

    avg_down_days = (dff.loc[dff['status'].isin(['down'])]['durationH'].mean()) / 24
    days = (end - start) / np.timedelta64(1,'D')

    avg_down_days_p = (avg_down_days / (days))*100

    avg_down_days = format(avg_down_days,".2f")
    avg_down_days_p = format(avg_down_days_p,".2f")

    return  avg_down_days, f'{avg_down_days_p} %'

#Current communiacation status
@app.callback(
    Output('ccs-output', 'children'),
    Input('stored-data', 'data')
)
def update_ccs_output(data):
    dff = df.copy()

    current_time=pd.Timestamp(dff.endTime.max())

    interval = pd.IntervalIndex.from_arrays(dff.startTime, dff.endTime, closed='both')
    dff = dff[interval.contains(current_time)]

    dff = dff.drop(dff.loc[dff.duplicated(['deviceId'], keep='last') == True].index)

    down = dff['status'].value_counts()[1]

    ccd = (down / dff.shape[0])*100
    ccd = format(ccd,".2f")

    return f'{ccd} %'

#Building Score
@app.callback(
    Output('bs-output', 'children'),
    Output('area-dropdown', 'options'),
    Output('ds-dropdown', 'options'),

    Input('stored-data', 'data')
)
def update_bs_output(data):
    dff = filter_df(df, startDate=data['startDate'], endDate=data['endDate'], buildings=data['building'])
    score = score_finder(dff)

    #sets options to  input fields depending on selected building
    areas = dff['area'].unique()
    ds = dff['dataSource'].unique()

    return score, areas, ds
      
#Data Source Score
@app.callback(
    Output('dscs-output', 'children'),
    Input('stored-data', 'data')
)
def update_dscs_output(data):
    if data['dataSource'] == None:
        return 'Select Data Scource.'

    else:
        dff = filter_df(df, startDate=data['startDate'], endDate=data['endDate'], buildings=data['building'], apis=data['dataSource'])
        score = score_finder(dff)
        return score

#Area Communication Source Score
@app.callback(
    Output('acs-output', 'children'),
    Input('stored-data', 'data')
)
def update_acs_output(data):
    if data['area'] == None:
        return 'Select Area.'

    else:
        dff = filter_df(df, startDate=data['startDate'], endDate=data['endDate'], buildings=data['building'], areas=data['area'])
        score = score_finder(dff)
        return score

#Average of Communication switch
@app.callback(
    Output('switch-output', 'children'),
    Input('stored-data', 'data')
)
def update_switch_output(data):
    dff = filter_df(df, startDate=data['startDate'], endDate=data['endDate'])
    
    down = dff['status'].value_counts()
    deviceNr = dff['deviceId'].nunique()

    switch_avg = down[1] / deviceNr
    switch_avg = format(switch_avg,".2f")

    return switch_avg

#Industrial scores
@app.callback(
    Output('industrial-t-output', 'data'),
    Output('industrial-t-output', 'columns'),
    Input('stored-data', 'data')
)
def update_i_t_output(data):
    dff2 = filter_df(df, startDate=data['startDate'], endDate=data['endDate'], buildings=data['building'])
    dff2 = dff2.loc[dff2['status'] == 'down']

    start = dff2['startTime'].min()
    end = dff2['endTime'].max()
    days = (end - start) / np.timedelta64(1,'D')

    ind_types = dff2.groupby('industrialType')['durationH'].mean()/24

    ind_types = ((ind_types / days) * 100).round(2)

    ind_types = ind_types.astype('string')
    ind_types = ind_types + ' %'
    ind_types = ind_types.to_dict()
    indu_ser_data =  {'Device Type':list(ind_types.keys()), 'Average Down Time': list(ind_types.values())}

    indu_df = pd.DataFrame(indu_ser_data)

    pd.options.display.float_format = '{:.2f}'.format

    data_indu = indu_df.to_dict('rows')
    columns_indu = [{'name':i, 'id':i} for i in indu_df.columns]

    return data_indu, columns_indu

#Graph
@app.callback(
    Output('graph', 'figure'),
    Input('stored-data', 'data')
)
def update_graph_output(data):
    dff = filter_df(df, startDate=data['startDate'], endDate=data['endDate'], buildings=data['building'] )

    startD=pd.Timestamp(data['startDate'])
    endD=pd.Timestamp(data['endDate'])
    
    days = ((endD) -(startD)) / np.timedelta64(1,'D')


    if days <= 7:
        intervals = pd.interval_range(startD, endD, freq='D', closed='both')

        percentages = get_dt_percent(intervals, dff)

    else:
        intervals = pd.interval_range(startD, endD, freq='7D', closed='both')

        percentages = get_dt_percent(intervals, dff)

    graphData = {'Intervals':percentages[1], 'Percentage': percentages[0]}
    
    graphDf = pd.DataFrame(graphData)

    fig = px.line(graphDf, x="Intervals", y="Percentage", title='Communication Score Trend')
       
    return fig
    
#Data Source and Building table
@app.callback(
    Output('ds-building-t-output', 'data'),
    Output('ds-building-t-output', 'columns'),
    Input('stored-data', 'data')
)
def update_ds_b_t_output(data):
    dff = filter_df(df, buildings=data['building'])

    dffDown =  dff.loc[dff['status'] == 'down']
    bNames = dff.groupby('dataSource')['buildingName'].first()
    dff2 = bNames.to_frame()
    dff2['Down Time (%)'] = (dffDown.groupby('dataSource')['durationH'].sum() / dff.groupby('dataSource')['durationH'].sum())*100

    current_time=pd.Timestamp(dff.endTime.max())
    interval = pd.IntervalIndex.from_arrays(dff.startTime, dff.endTime, closed='both')
    dff = dff[interval.contains(current_time)]

    current_time=pd.Timestamp(dffDown.endTime.max())
    interval = pd.IntervalIndex.from_arrays(dffDown.startTime, dffDown.endTime, closed='both')
    dffDown = dffDown[interval.contains(current_time)]

    dff2['Current Down Status (%)'] = (dffDown.groupby('dataSource')['status'].count() / dff.groupby('dataSource')['status'].count())*100
    dff2 = dff2.reset_index()
    dff2 = dff2.rename(columns={'dataSource' : 'Data Source', 'buildingName':'Building Name'})

    dff2['Current Down Status (%)'] = dff2['Current Down Status (%)'].fillna(0).round(2)

    data = dff2.to_dict('rows')
    columns = [{'name':i, 'id':i} for i in dff2.columns]

    return data, columns

#Down Devices Table
@app.callback(
    Output('down-device-t-output', 'data'),
    Output('down-device-t-output', 'columns'),
    Input('stored-data', 'data')
)
def update_down_table_output(data):
    dff = filter_df(df, buildings=data['building'])

    areaS = dff.groupby('deviceId')['area'].first()
    dff2 = areaS.to_frame()
    dff2['Data Source'] = dff.groupby('deviceId')['dataSource'].first()
    dffDown = dff.loc[dff['status'] == 'down']
    dff2['Down Time (Hours)'] = dffDown.groupby('deviceId')['durationH'].sum().round()
    dff2['Down Frequency'] = dffDown.groupby('deviceId')['status'].count()
    dff2['Current Status'] = dff.groupby(['deviceId'])['status'].tail(1).values

    dff2 = dff2.reset_index()
    dff2 = dff2.rename(columns={'deviceId' : 'Device ID', 'area':'Area'})

    data = dff2.to_dict('rows')
    columns = [{'name':i, 'id':i} for i in dff2.columns]

    return data, columns


if __name__ == '__main__':
    app.run_server(debug=True)