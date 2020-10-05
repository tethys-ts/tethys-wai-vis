# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import urllib
import requests
import zstandard as zstd
import orjson
import flask
# from util import app_ts_summ, sel_ts_summ, ecan_ts_data

pd.options.display.max_columns = 10

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server,  url_base_pathname = '/')

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)
# server = app.server

##########################################
### Parameters

base_url = 'http://tethys-ts.xyz/tethys/data/'


def select_dataset(features, parameters, methods, processing_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, datasets):
    """

    """
    dataset = [d for d in datasets if (d['feature'] == features) and (d['parameter'] == parameters) and (d['method'] == methods) and (d['owner'] == owners) and (d['aggregation_statistic'] == aggregation_statistics) and (d['frequency_interval'] == frequency_intervals) and (d['utc_offset'] == utc_offsets) and (d['processing_code'] == processing_codes)][0]

    return dataset


ts_plot_height = 600
map_height = 700


lat1 = -43.45
lon1 = 171.9
zoom1 = 7

mapbox_access_token = "pk.eyJ1IjoibXVsbGVua2FtcDEiLCJhIjoiY2pudXE0bXlmMDc3cTNxbnZ0em4xN2M1ZCJ9.sIOtya_qe9RwkYXj5Du1yg"

###############################################
### App layout

map_layout = dict(mapbox = dict(layers = [], accesstoken = mapbox_access_token, style = 'outdoors', center=dict(lat=lat1, lon=lon1), zoom=zoom1), margin = dict(r=0, l=0, t=0, b=0), autosize=True, hovermode='closest', height=map_height)

# @server.route('/wai-vis')
# def main():
def serve_layout():

    dc = zstd.ZstdDecompressor()

    datasets = requests.get(base_url + 'datasets').json()

    requested_datasets = datasets.copy()

    features = list(set([f['feature'] for f in requested_datasets]))
    features.sort()

    parameters = list(set([f['parameter'] for f in requested_datasets]))
    parameters.sort()

    methods = list(set([f['method'] for f in requested_datasets]))
    methods.sort()

    processing_codes = list(set([f['processing_code'] for f in requested_datasets]))
    processing_codes.sort()

    owners = list(set([f['owner'] for f in requested_datasets]))
    owners.sort()

    aggregation_statistics = list(set([f['aggregation_statistic'] for f in requested_datasets]))
    aggregation_statistics.sort()

    frequency_intervals = list(set([f['frequency_interval'] for f in requested_datasets]))
    frequency_intervals.sort()

    utc_offsets = list(set([f['utc_offset'] for f in requested_datasets]))
    utc_offsets.sort()

    init_dataset = [d for d in requested_datasets if (d['feature'] == 'waterway') and (d['parameter'] == 'streamflow') and (d['processing_code'] == 'quality_controlled_data')][0]

    init_dataset_id = init_dataset['dataset_id']

    dataset_table_cols = {'license': 'Data License', 'precision': 'Data Precision', 'units': 'Units'}

    ### prepare summaries and initial states
    max_date = pd.Timestamp.now()
    start_date = max_date - pd.DateOffset(years=1)

    # init_summ = sel_ts_summ(ts_summ, 'River', 'Flow', 'Recorder', 'Primary', 'ECan', str(start_date.date()), str(max_date.date()))
    #
    # new_sites = init_summ.drop_duplicates('ExtSiteID')

    init_summ_r = requests.post(base_url + 'sampling_sites', params={'dataset_id': init_dataset_id, 'compression': 'zstd'})

    init_summ = orjson.loads(dc.decompress(init_summ_r.content))
    init_summ = [s for s in init_summ if (pd.Timestamp(s['stats']['to_date']) > start_date) and (pd.Timestamp(s['stats']['from_date']) < max_date)]

    init_sites = [{'label': s['ref'], 'value': s['site_id']} for s in init_summ]

    init_site_id = [s['value'] for s in init_sites if s['label'] == '70105'][0]

    init_lon = [l['geometry']['coordinates'][0] for l in init_summ]
    init_lat = [l['geometry']['coordinates'][1] for l in init_summ]
    init_names = [l['ref'] + '<br>' + l['name'] for l in init_summ]

    init_table = [{'Site ID': s['ref'], 'Site Name': s['name'], 'Min Value': s['stats']['min'], 'Mean Value': s['stats']['mean'], 'Max Value': s['stats']['max'], 'Start Date': s['stats']['from_date'], 'End Date': s['stats']['to_date'], 'Last Modified Date': s['modified_date']} for s in init_summ]

    # init_ts_r = requests.get(base_url + 'time_series_results', params={'dataset_id': init_dataset_id, 'site_id': init_site_id, 'compression': 'zstd', 'from_date': start_date.round('s').isoformat(), 'to_date': max_date.round('s').isoformat()})
    # dc = zstd.ZstdDecompressor()
    # df1 = pd.DataFrame(orjson.loads(dc.decompress(init_ts_r.content)))

    layout = html.Div(children=[
    html.Div([
        html.P(children='Filter datasets (select from top to bottom):'),
        html.Label('Feature'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in features], value='waterway', id='features'),
        html.Label('Parameter'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in parameters], value='streamflow', id='parameters'),
        html.Label('Method'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in methods], value='sensor_recording', id='methods'),
        html.Label('Processing Code'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in processing_codes], value='quality_controlled_data', id='processing_codes'),
        html.Label('Data Owner'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in owners], value='ECan', id='owners'),
        html.Label('Aggregation Statistic'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in aggregation_statistics], value='mean', id='aggregation_statistics'),
        html.Label('Frequency Interval'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in frequency_intervals], value='1H', id='frequency_intervals'),
        html.Label('UTC Offset'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in utc_offsets], value='0H', id='utc_offsets'),
        html.Label('Date Range'),
        dcc.DatePickerRange(
            end_date=str(max_date.date()),
            display_format='DD/MM/YYYY',
            start_date=str(start_date.date()),
            id='date_sel'
#               start_date_placeholder_text='DD/MM/YYYY'
            ),
        html.Label('Site IDs'),
        dcc.Dropdown(options=init_sites, id='sites')
        # html.Label('Water quality below detection limit method'),
        # dcc.RadioItems(
        #     options=[
        #         {'label': 'Half dtl', 'value': 'half'},
        #         {'label': 'Trend analysis method', 'value': 'trend'}
        #     ],
        #     value='half',
        #     id='dtl')
        ],
    className='two columns', style={'margin': 20}),

    html.Div([
        html.P('Click on a site or "box select" multiple sites:', style={'display': 'inline-block'}),
        dcc.Graph(
                id = 'site-map',
                style={'height': map_height},
                figure=dict(
                        data = [dict(lat = init_lat,
                                    lon = init_lon,
                                    text = init_names,
                                    type = 'scattermapbox',
                                    hoverinfo = 'text',
                                    marker = dict(
                                            size=8,
                                            color='black',
                                            opacity=1
                                            )
                                    )
                                ],
                        layout=map_layout),
                config={"displaylogo": False}),

#         html.A(
#             'Download Dataset Summary Data',
#             id='download-summ',
#             download="dataset_summary.csv",
#             href="",
#             target="_blank",
#             style={'margin': 50}),
#
        dash_table.DataTable(
            id='dataset_table',
            columns=[{"name": v, "id": v, 'deletable': True} for k, v in dataset_table_cols.items()],
            data=[],
            sort_action="native",
            sort_mode="multi",
            style_cell={
                'minWidth': '80px', 'maxWidth': '200px',
                'whiteSpace': 'normal'}
            )

    ], className='four columns', style={'margin': 20}),
#
    html.Div([

# 		html.P('Select Dataset for time series plot:', style={'display': 'inline-block'}),
# 		dcc.Dropdown(options=[{'value:': 5, 'label': init_dataset}], value=5, id='sel_dataset'),
        dcc.Graph(
            id = 'selected-data',
            figure = dict(
                data = [dict(x=0, y=0)],
                layout = dict(
                        paper_bgcolor = '#F4F4F8',
                        plot_bgcolor = '#F4F4F8',
                        height = ts_plot_height
                        )
                ),
            config={"displaylogo": False}
            ),
        html.A(
            'Download Time Series Data',
            id='download-tsdata',
            download="tsdata.csv",
            href="",
            target="_blank",
            style={'margin': 50}),
        dash_table.DataTable(
            id='summ_table',
            columns=[{"name": i, "id": i, 'deletable': True} for i in init_table[0].keys()],
            data=init_table,
            sort_action="native",
            sort_mode="multi",
            style_cell={
                'minWidth': '80px', 'maxWidth': '200px',
                'whiteSpace': 'normal'
            }
            )
    ], className='six columns', style={'margin': 10, 'height': 900}),
    html.Div(id='ts_data', style={'display': 'none'}),
    html.Div(id='datasets', style={'display': 'none'}, children=orjson.dumps(datasets).decode()),
    html.Div(id='dataset_id', style={'display': 'none'}, children=init_dataset_id),
    html.Div(id='sites_summ', style={'display': 'none'}, children=orjson.dumps(init_summ).decode())
#     dcc.Graph(id='map-layout', style={'display': 'none'}, figure=dict(data=[], layout=map_layout))
], style={'margin':0})

    return layout


app.layout = serve_layout

########################################
### Callbacks


@app.callback(
    [Output('parameters', 'options'), Output('methods', 'options'), Output('processing_codes', 'options'), Output('owners', 'options'), Output('aggregation_statistics', 'options'), Output('frequency_intervals', 'options'), Output('utc_offsets', 'options')],
    [Input('features', 'value')],
    [State('datasets', 'children')])
def update_parameters(features, datasets):

    def make_options(val):
        l1 = [{'label': v, 'value': v} for v in val]
        return l1

    datasets1 = orjson.loads(datasets)
    datasets2 = [d for d in datasets1 if d['feature'] == features]

    parameters = list(set([d['parameter'] for d in datasets2]))
    parameters.sort()

    methods = list(set([d['method'] for d in datasets2]))
    methods.sort()

    processing_codes = list(set([d['processing_code'] for d in datasets2]))
    processing_codes.sort()

    owners = list(set([d['owner'] for d in datasets2]))
    owners.sort()

    aggregation_statistics = list(set([d['aggregation_statistic'] for d in datasets2]))
    aggregation_statistics.sort()

    frequency_intervals = list(set([d['frequency_interval'] for d in datasets2]))
    frequency_intervals.sort()

    utc_offsets = list(set([d['utc_offset'] for d in datasets2]))
    utc_offsets.sort()

    return make_options(parameters), make_options(methods), make_options(processing_codes), make_options(owners), make_options(aggregation_statistics), make_options(frequency_intervals), make_options(utc_offsets)


@app.callback(
    Output('dataset_id', 'children'), [Input('features', 'value'), Input('parameters', 'value'), Input('methods', 'value'), Input('processing_codes', 'value'), Input('owners', 'value'), Input('aggregation_statistics', 'value'), Input('frequency_intervals', 'value'), Input('utc_offsets', 'value')], [State('datasets', 'children')])
def update_dataset_id(features, parameters, methods, processing_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, datasets):
    try:
        dataset = select_dataset(features, parameters, methods, processing_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, orjson.loads(datasets))
        dataset_id = dataset['dataset_id']

        print(features, parameters, methods, processing_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets)
        return dataset_id
    except:
        print('No available dataset_id')


@app.callback(
    Output('sites_summ', 'children'),
    [Input('dataset_id', 'children'), Input('date_sel', 'start_date'), Input('date_sel', 'end_date')])
def update_summ_data(dataset_id, start_date, end_date):
    if dataset_id is None:
        print('No new sites_summ')
    else:
        summ_r = requests.post(base_url + 'sampling_sites', params={'dataset_id': dataset_id, 'compression': 'zstd'})

        dc = zstd.ZstdDecompressor()
        summ_data1 = orjson.loads(dc.decompress(summ_r.content).decode())
        summ_data2 = [s for s in summ_data1 if (pd.Timestamp(s['stats']['to_date']) > pd.Timestamp(start_date)) and (pd.Timestamp(s['stats']['from_date']) < pd.Timestamp(end_date))]
        summ_json = orjson.dumps(summ_data2).decode()

        return summ_json


@app.callback(
    Output('sites', 'options'), [Input('sites_summ', 'children')])
def update_site_list(sites_summ):
    if sites_summ is None:
        print('No sites available')
        return []
    else:
        sites_summ1 = orjson.loads(sites_summ)
        sites_options = [{'label': s['ref'], 'value': s['site_id']} for s in sites_summ1]

        return sites_options


@app.callback(
        Output('site-map', 'figure'),
        [Input('sites_summ', 'children')],
        [State('site-map', 'figure')])
def update_display_map(sites_summ, figure):
    if sites_summ is None:
        print('Clear the sites')
        data1 = figure['data'][0]
        if 'hoverinfo' in data1:
            data1.pop('hoverinfo')
        data1.update(dict(size=8, color='black', opacity=0))
        fig = dict(data=[data1], layout=figure['layout'])
    else:
        sites_summ1 = orjson.loads(sites_summ)

        lon1 = [l['geometry']['coordinates'][0] for l in sites_summ1]
        lat1 = [l['geometry']['coordinates'][1] for l in sites_summ1]
        names1 = [l['ref'] + '<br>' + l['name'] for l in sites_summ1]

        data = [dict(
            lat = lat1,
            lon = lon1,
            text = names1,
            type = 'scattermapbox',
            hoverinfo = 'text',
            marker = dict(size=8, color='black', opacity=1)
        )]

        fig = dict(data=data, layout=figure['layout'])

    return fig


@app.callback(
        Output('sites', 'value'),
        [Input('site-map', 'selectedData'), Input('site-map', 'clickData')],
        [State('sites_summ', 'children')])
def update_sites_values(selectedData, clickData, sites_summ):
    # print(clickData)
    # print(selectedData)
    if selectedData:
        site1_index = selectedData['points'][0]['pointIndex']
        # sites1 = [s['text'].split('<br>')[0] for s in selectedData['points']]
    elif clickData:
        site1_index = clickData['points'][0]['pointIndex']
        # sites1 = [clickData['points'][0]['text'].split('<br>')[0]]
    else:
        site1_index = None

    if site1_index:
        site1_id = orjson.loads(sites_summ)[site1_index]['site_id']
    else:
        site1_id = ''

    # print(sites1_id)

    return site1_id


@app.callback(
    Output('summ_table', 'data'),
    [Input('sites_summ', 'children'), Input('sites', 'value'), Input('site-map', 'selectedData'), Input('site-map', 'clickData')])
def update_table(sites_summ, sites, selectedData, clickData):
    if sites_summ:
        new_summ = orjson.loads(sites_summ)

        if sites:
            summ_table = [{'Site ID': s['ref'], 'Site Name': s['name'], 'Min Value': s['stats']['min'], 'Mean Value': s['stats']['mean'], 'Max Value': s['stats']['max'], 'Start Date': s['stats']['from_date'], 'End Date': s['stats']['to_date'], 'Last Modified Date': s['modified_date']} for s in new_summ if s['site_id'] in sites]
        else:
            summ_table = [{'Site ID': s['ref'], 'Site Name': s['name'], 'Min Value': s['stats']['min'], 'Mean Value': s['stats']['mean'], 'Max Value': s['stats']['max'], 'Start Date': s['stats']['from_date'], 'End Date': s['stats']['to_date'], 'Last Modified Date': s['modified_date']} for s in new_summ]

        return summ_table


@app.callback(
    Output('ts_data', 'children'),
    [Input('sites', 'value'), Input('date_sel', 'start_date'), Input('date_sel', 'end_date'), Input('dataset_id', 'children')])
def get_data(sites, start_date, end_date, dataset_id):
    if dataset_id:
        if sites:
            ts_r = requests.get(base_url + 'time_series_results', params={'dataset_id': dataset_id, 'site_id': sites, 'compression': 'zstd', 'from_date': start_date+'T00:00', 'to_date': end_date+'T00:00'})
            dc = zstd.ZstdDecompressor()
            ts1 = dc.decompress(ts_r.content).decode()

            return ts1



@app.callback(
    Output('selected-data', 'figure'),
    [Input('ts_data', 'children')],
    [State('sites', 'value'), State('dataset_id', 'children'), State('date_sel', 'start_date'), State('date_sel', 'end_date')])
def display_data(ts_data, sites, dataset_id, start_date, end_date):

    base_dict = dict(
            data = [dict(x=0, y=0)],
            layout = dict(
                title='Click-drag on the map to select sites',
                paper_bgcolor = '#F4F4F8',
                plot_bgcolor = '#F4F4F8'
                )
            )

    if not sites:
        return base_dict

    if not dataset_id:
        return base_dict

    ts1 = orjson.loads(ts_data)

    x1 = [t['from_date'] for t in ts1]
    y1 = [t['result'] for t in ts1]

    set1 = go.Scattergl(
            x=x1,
            y=y1,
            showlegend=False,
            # name=s,
#                line={'color': col3[s]},
            opacity=0.8)

    layout = dict(title = 'Time series data', paper_bgcolor = '#F4F4F8', plot_bgcolor = '#F4F4F8', xaxis = dict(range = [start_date, end_date]), showlegend=True, height=ts_plot_height)

    fig = dict(data=[set1], layout=layout)

    return fig


@app.callback(
    Output('dataset_table', 'data'),
    [Input('dataset_id', 'children')],
    [State('datasets', 'children')])
def update_table(dataset_id, datasets):
    if dataset_id:
        dataset_table_cols = {'license': 'Data License', 'precision': 'Data Precision', 'units': 'Units'}

        datasets1 = orjson.loads(datasets)

        dataset1 = [d for d in datasets1 if d['dataset_id'] == dataset_id][0]

        dataset_table1 = {}
        [dataset_table1.update({v: dataset1[k]}) for k, v in dataset_table_cols.items()]

        return [dataset_table1]


@app.callback(
    Output('download-tsdata', 'href'),
    [Input('ts_data', 'children')],
    [State('sites', 'value'), State('dataset_id', 'children')])
def download_tsdata(ts_data, sites, dataset_id):
    if dataset_id:
        if sites:
            ts_data1 = pd.DataFrame(orjson.loads(ts_data))
            ts_data1['from_date'] = pd.to_datetime(ts_data1['from_date'])

            csv_string = ts_data1.to_csv(index=False, encoding='utf-8')
            csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
            return csv_string


# @app.callback(
#     Output('download-summ', 'href'),
#     [Input('summ_data', 'children')])
# def download_summ(summ_data):
#     new_summ = pd.read_json(summ_data, orient='split')[table_cols]
#
#     csv_string = new_summ.to_csv(index=False, encoding='utf-8')
#     csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
#     return csv_string


if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0', port=80)


# @server.route("/wai-vis")
# def my_dash_app():
#     return app.index()
