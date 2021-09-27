# -*- coding: utf-8 -*-
import io
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
from typing import Optional, List, Any
from flask_caching import Cache
import xarray as xr
import codecs
import pickle
# from tethysts.utils import get_results_obj_s3, result_filters, process_results_output
# from util import app_ts_summ, sel_ts_summ, ecan_ts_data

pd.options.display.max_columns = 10

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# server = flask.Flask(__name__)
# app = dash.Dash(__name__, server=server,  url_base_pathname = '/')

app = dash.Dash(__name__,  url_base_pathname = '/')
server = app.server

##########################################
### Parameters

# base_url = 'http://tethys-ts.xyz/tethys/data/'
# base_url = 'http://127.0.0.1:8080/tethys/data/'
# base_url = 'host.docker.internal/tethys/data/'
# base_url = 'https://api.tethys-ts.xyz/tethys/data/'
base_url = 'http://tethys-api-ext:80/tethys/data/'
# tethys = Tethys()

cache_config = {
    # "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "FileSystemCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 60*60,
    'CACHE_DIR': '/cache'
    # 'CACHE_DIR': '/media/nvme1/data/UC/wrf'
}

cache = Cache(server, config=cache_config)


# def dataset_filter(dataset_id: Optional[str] = None, feature: Optional[str] = None, parameter: Optional[str] = None, method: Optional[str] = None, product_code: Optional[str] = None, owner: Optional[str] = None, aggregation_statistic: Optional[str] = None, frequency_interval: Optional[str] = None, utc_offset: Optional[str] = None):
#     """

#     """
#     q_dict = {}

#     if isinstance(dataset_id, str):
#         q_dict.update({'dataset_id': dataset_id})
#     if isinstance(feature, str):
#         q_dict.update({'feature': feature})
#     if isinstance(parameter, str):
#         q_dict.update({'parameter': parameter})
#     if isinstance(method, str):
#         q_dict.update({'method': method})
#     if isinstance(product_code, str):
#         q_dict.update({'product_code': product_code})
#     if isinstance(owner, str):
#         q_dict.update({'owner': owner})
#     if isinstance(aggregation_statistic, str):
#         q_dict.update({'aggregation_statistic': aggregation_statistic})
#     if isinstance(frequency_interval, str):
#         q_dict.update({'frequency_interval': frequency_interval})
#     if isinstance(utc_offset, str):
#         q_dict.update({'utc_offset': utc_offset})

#     return q_dict

def encode_obj(obj, encoding="base64"):
    """

    """
    e1 = codecs.encode(pickle.dumps(obj), encoding).decode()

    return e1


def decode_obj(obj, encoding="base64"):
    """

    """
    d1 = pickle.loads(codecs.decode(obj.encode(), encoding))

    return d1


def select_dataset(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, datasets):
    """

    """
    dataset = [d for d in datasets if (d['feature'] == features) and (d['parameter'] == parameters) and (d['method'] == methods) and (d['owner'] == owners) and (d['aggregation_statistic'] == aggregation_statistics) and (d['frequency_interval'] == frequency_intervals) and (d['utc_offset'] == utc_offsets) and (d['product_code'] == product_codes)][0]

    return dataset


def build_table(site_summ, dataset):
    """

    """
    table1 = [{'Station reference': s['ref'], 'Station Name': s['name'], 'Min Value': s['stats']['min'], 'Max Value': s['stats']['max'], 'Units': dataset['units'], 'Precision': dataset['precision'], 'Start Date': s['time_range']['from_date'], 'End Date': s['time_range']['to_date'], 'lon': s['geometry']['coordinates'][0], 'lat': s['geometry']['coordinates'][1]} for s in site_summ]

    return table1


table_cols = ['Station reference', 'Station Name', 'Min Value', 'Max Value', 'Units', 'Precision', 'Start Date', 'End Date', 'lon', 'lat']

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

    datasets = requests.get(base_url + 'get_datasets').json()

    # tethys = Tethys()

    # datasets = tethys.datasets.copy()

    datasets = [ds for ds in datasets if ds['method'] != 'simulation']

    requested_datasets = datasets.copy()

    features = list(set([f['feature'] for f in requested_datasets]))
    features.sort()

    parameters = list(set([f['parameter'] for f in requested_datasets]))
    parameters.sort()

    methods = list(set([f['method'] for f in requested_datasets]))
    methods.sort()

    product_codes = list(set([f['product_code'] for f in requested_datasets]))
    product_codes.sort()

    owners = list(set([f['owner'] for f in requested_datasets]))
    owners.sort()

    aggregation_statistics = list(set([f['aggregation_statistic'] for f in requested_datasets]))
    aggregation_statistics.sort()

    frequency_intervals = list(set([f['frequency_interval'] for f in requested_datasets]))
    frequency_intervals.sort()

    utc_offsets = list(set([f['utc_offset'] for f in requested_datasets]))
    utc_offsets.sort()

    # init_dataset = [d for d in requested_datasets if (d['feature'] == 'waterway') and (d['parameter'] == 'streamflow') and (d['product_code'] == 'quality_controlled_data') and (d['owner'] == 'Environment Canterbury')][0]

    # init_dataset_id = init_dataset['dataset_id']

    dataset_table_cols = {'license': 'Data License', 'attribution': 'Attribution'}

    ### prepare summaries and initial states
    max_date = pd.Timestamp.now()
    start_date = max_date - pd.DateOffset(years=1)

    # init_summ = sel_ts_summ(ts_summ, 'River', 'Flow', 'Recorder', 'Primary', 'ECan', str(start_date.date()), str(max_date.date()))
    #
    # new_sites = init_summ.drop_duplicates('ExtSiteID')

    # init_summ_r = requests.post(base_url + 'get_stations', params={'dataset_id': init_dataset_id, 'compression': 'zstd'})

    # init_summ = orjson.loads(dc.decompress(init_summ_r.content))
    # init_summ = [s for s in init_summ if (pd.Timestamp(s['stats']['to_date']).tz_localize(None) > start_date) and (pd.Timestamp(s['stats']['from_date']).tz_localize(None) < max_date)]

    # init_sites = [{'label': s['ref'], 'value': s['station_id']} for s in init_summ]

    # init_site_id = [s['value'] for s in init_sites if s['label'] == '70105'][0]

    # init_lon = [l['geometry']['coordinates'][0] for l in init_summ]
    # init_lat = [l['geometry']['coordinates'][1] for l in init_summ]
    # init_names = [l['ref'] + '<br>' + l['name'] if 'name' in l else l['ref'] for l in init_summ]

    # init_table = build_table(init_summ, init_dataset)

    # init_ts_r = requests.get(base_url + 'time_series_results', params={'dataset_id': init_dataset_id, 'site_id': init_site_id, 'compression': 'zstd', 'from_date': start_date.round('s').isoformat(), 'to_date': max_date.round('s').isoformat()})
    # dc = zstd.ZstdDecompressor()
    # df1 = pd.DataFrame(orjson.loads(dc.decompress(init_ts_r.content)))

    layout = html.Div(children=[
    html.Div([
        html.P(children='Filter datasets (select from top to bottom):'),
        html.Label('Feature'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in features], value=None, id='features'),
        html.Label('Parameter'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in parameters], value=None, id='parameters'),
        html.Label('Method'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in methods], value=None, id='methods'),
        html.Label('Product Code'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in product_codes], value=None, id='product_codes'),
        html.Label('Data Owner'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in owners], value=None, id='owners'),
        html.Label('Aggregation Statistic'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in aggregation_statistics], value=None, id='aggregation_statistics'),
        html.Label('Frequency Interval'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in frequency_intervals], value=None, id='frequency_intervals'),
        html.Label('UTC Offset'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in utc_offsets], value=None, id='utc_offsets'),
        html.Label('Date Range'),
        dcc.DatePickerRange(
            end_date=str(max_date.date()),
            display_format='DD/MM/YYYY',
            start_date=str(start_date.date()),
            id='date_sel'
#               start_date_placeholder_text='DD/MM/YYYY'
            ),
        html.Label('Station reference ID'),
        # dcc.Dropdown(options=init_sites, id='sites')
        dcc.Dropdown(options=[], id='sites')
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
        html.P('Click on a station to view the time series data:', style={'display': 'inline-block'}),
        dcc.Graph(
            id = 'site-map',
            style={'height': map_height},
            figure=dict(
                    data = [dict(
                                # lat = init_lat,
                                # lon = init_lon,
                                # text = init_names,
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

    ], className='three columns', style={'margin': 20}),
#
    html.Div([

# 		html.P('Select Dataset for time series plot:', style={'display': 'inline-block'}),
# 		dcc.Dropdown(options=[{'value:': 5, 'label': init_dataset}], value=5, id='sel_dataset'),
        dcc.Loading(
                id="loading-1",
                type="default",
                children=dcc.Graph(
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
                            )
                ),
        # html.A(
        #     'Download Time Series Data',
        #     id='download-tsdata',
        #     download="tsdata.csv",
        #     href="",
        #     target="_blank",
        #     style={'margin': 50}),
        html.Button("Download CSV", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),
        # html.Button("Download netCDF", id="btn_netcdf"),
        # dcc.Download(id="download-netcdf"),
        dash_table.DataTable(
            id='summ_table',
            columns=[{"name": i, "id": i} for i in table_cols],
            data=[],
            sort_action="native",
            sort_mode="multi",
            style_cell={
                'minWidth': '80px', 'maxWidth': '200px',
                'whiteSpace': 'normal'
            }
            )
    ], className='six columns', style={'margin': 10, 'height': 900}),
    html.Div(id='ts_data', style={'display': 'none'}),
    # dcc.Store(id='tethys', data=encode_obj(tethys)),
    html.Div(id='datasets', style={'display': 'none'}, children=orjson.dumps(datasets).decode()),
    html.Div(id='dataset_id', style={'display': 'none'}),
    html.Div(id='sites_summ', style={'display': 'none'})
    # dcc.Store(id='map_layout', data=orjson.dumps(map_layout).decode())
#     dcc.Graph(id='map-layout', style={'display': 'none'}, figure=dict(data=[], layout=map_layout))
], style={'margin':0})

    return layout


app.layout = serve_layout

########################################
### Callbacks


# @app.callback(
#     [Output('features', 'options'), Output('parameters', 'options'), Output('methods', 'options'), Output('product_codes', 'options'), Output('owners', 'options'), Output('aggregation_statistics', 'options'), Output('frequency_intervals', 'options'), Output('utc_offsets', 'options')],
#     [Input('features', 'value'), Input('parameters', 'value'), Input('methods', 'value'), Input('product_codes', 'value'), Input('owners', 'value'), Input('aggregation_statistics', 'value'), Input('frequency_intervals', 'value'), Input('utc_offsets', 'value')],
#     [State('datasets', 'children')])
# def update_parameters(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, datasets):

#     def make_options(val):
#         l1 = [{'label': v, 'value': v} for v in val]
#         return l1

#     datasets1 = orjson.loads(datasets)
#     datasets2 = [d for d in datasets1 if d['feature'] == features]

#     # print(datasets2)

#     # features = list(set([d['feature'] for d in datasets2]))
#     # features.sort()

#     parameters = list(set([d['parameter'] for d in datasets2]))
#     parameters.sort()

#     methods = list(set([d['method'] for d in datasets2]))
#     methods.sort()

#     product_codes = list(set([d['product_code'] for d in datasets2]))
#     product_codes.sort()

#     owners = list(set([d['owner'] for d in datasets2]))
#     owners.sort()

#     aggregation_statistics = list(set([d['aggregation_statistic'] for d in datasets2]))
#     aggregation_statistics.sort()

#     frequency_intervals = list(set([d['frequency_interval'] for d in datasets2]))
#     frequency_intervals.sort()

#     utc_offsets = list(set([d['utc_offset'] for d in datasets2]))
#     utc_offsets.sort()

#     return make_options(parameters), make_options(methods), make_options(product_codes), make_options(owners), make_options(aggregation_statistics), make_options(frequency_intervals), make_options(utc_offsets)


# @app.callback(
#     [Output('methods', 'options'), Output('product_codes', 'options'), Output('owners', 'options'), Output('aggregation_statistics', 'options'), Output('frequency_intervals', 'options'), Output('utc_offsets', 'options')],
#     [Input('parameters', 'value')],
#     [State('features', 'value')]
#     )
# def update_parameters2(parameters, features):
#     print(features)
#
#     def make_options(val):
#         l1 = [{'label': v, 'value': v} for v in val]
#         return l1
#
#     d_dict = dataset_filter(None, features, parameters)
#
#     datasets2 = requests.get(base_url + 'get_datasets', params=d_dict).json()
#
#     # features = list(set([d['feature'] for d in datasets2]))
#     # features.sort()
#
#     # parameters = list(set([d['parameter'] for d in datasets2]))
#     # parameters.sort()
#
#     methods = list(set([d['method'] for d in datasets2]))
#     methods.sort()
#
#     product_codes = list(set([d['product_code'] for d in datasets2]))
#     product_codes.sort()
#
#     owners = list(set([d['owner'] for d in datasets2]))
#     owners.sort()
#
#     aggregation_statistics = list(set([d['aggregation_statistic'] for d in datasets2]))
#     aggregation_statistics.sort()
#
#     frequency_intervals = list(set([d['frequency_interval'] for d in datasets2]))
#     frequency_intervals.sort()
#
#     utc_offsets = list(set([d['utc_offset'] for d in datasets2]))
#     utc_offsets.sort()
#
#     return make_options(methods), make_options(product_codes), make_options(owners), make_options(aggregation_statistics), make_options(frequency_intervals), make_options(utc_offsets)


# @app.callback(
#     [Output('features', 'value'), Output('parameters', 'options'), Output('methods', 'options'), Output('product_codes', 'options'), Output('owners', 'options'), Output('aggregation_statistics', 'options'), Output('frequency_intervals', 'options'), Output('utc_offsets', 'options')],
#     [Input('features', 'value'), Input('parameters', 'value'), Input('methods', 'value'), Input('product_codes', 'value'), Input('owners', 'value'), Input('aggregation_statistics', 'value'), Input('frequency_intervals', 'value'), Input('utc_offsets', 'value')]
#     )
# def update_parameters(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets):
#     print(features)
#
#     def make_options(val):
#         l1 = [{'label': v, 'value': v} for v in val]
#         return l1
#
#     d_dict = dataset_filter(None, features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets)
#
#     datasets2 = requests.get(base_url + 'get_datasets', param=d_dict).json()
#
#     features = list(set([d['feature'] for d in datasets2]))
#     features.sort()
#
#     parameters = list(set([d['parameter'] for d in datasets2]))
#     parameters.sort()
#
#     methods = list(set([d['method'] for d in datasets2]))
#     methods.sort()
#
#     product_codes = list(set([d['product_code'] for d in datasets2]))
#     product_codes.sort()
#
#     owners = list(set([d['owner'] for d in datasets2]))
#     owners.sort()
#
#     aggregation_statistics = list(set([d['aggregation_statistic'] for d in datasets2]))
#     aggregation_statistics.sort()
#
#     frequency_intervals = list(set([d['frequency_interval'] for d in datasets2]))
#     frequency_intervals.sort()
#
#     utc_offsets = list(set([d['utc_offset'] for d in datasets2]))
#     utc_offsets.sort()
#
#     return make_options(features), make_options(parameters), make_options(methods), make_options(product_codes), make_options(owners), make_options(aggregation_statistics), make_options(frequency_intervals), make_options(utc_offsets)

@app.callback(
    [Output('features', 'options'), Output('parameters', 'options'), Output('methods', 'options'), Output('product_codes', 'options'), Output('owners', 'options'), Output('aggregation_statistics', 'options'), Output('frequency_intervals', 'options'), Output('utc_offsets', 'options'), Output('dataset_id', 'children')], [Input('features', 'value'), Input('parameters', 'value'), Input('methods', 'value'), Input('product_codes', 'value'), Input('owners', 'value'), Input('aggregation_statistics', 'value'), Input('frequency_intervals', 'value'), Input('utc_offsets', 'value')], [State('datasets', 'children')])
@cache.memoize()
def update_dataset_table(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, datasets):

    def make_options(val):
        l1 = [{'label': v, 'value': v} for v in val]
        return l1

    datasets1 = orjson.loads(datasets)
    if isinstance(features, str):
        datasets1 = [d for d in datasets1 if d['feature'] == features]
    if isinstance(parameters, str):
        datasets1 = [d for d in datasets1 if d['parameter'] == parameters]
    if isinstance(methods, str):
        datasets1 = [d for d in datasets1 if d['method'] == methods]
    if isinstance(product_codes, str):
        datasets1 = [d for d in datasets1 if d['product_code'] == product_codes]
    if isinstance(owners, str):
        datasets1 = [d for d in datasets1 if d['owner'] == owners]
    if isinstance(aggregation_statistics, str):
        datasets1 = [d for d in datasets1 if d['aggregation_statistic'] == aggregation_statistics]
    if isinstance(frequency_intervals, str):
        datasets1 = [d for d in datasets1 if d['frequency_interval'] == frequency_intervals]
    if isinstance(utc_offsets, str):
        datasets1 = [d for d in datasets1 if d['utc_offset'] == utc_offsets]

    # print(datasets1)

    if isinstance(features, str) and isinstance(parameters, str) and isinstance(methods, str) and isinstance(product_codes, str) and isinstance(owners, str) and isinstance(aggregation_statistics, str) and isinstance(frequency_intervals, str) and isinstance(utc_offsets, str):
        dataset = select_dataset(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, datasets1)
        dataset_id = dataset['dataset_id']
    else:
        print('Could not create dataset_id')
        dataset_id = None

    print(dataset_id)

    features = list(set([d['feature'] for d in datasets1]))
    features.sort()

    parameters = list(set([d['parameter'] for d in datasets1]))
    parameters.sort()

    methods = list(set([d['method'] for d in datasets1]))
    methods.sort()

    product_codes = list(set([d['product_code'] for d in datasets1]))
    product_codes.sort()

    owners = list(set([d['owner'] for d in datasets1]))
    owners.sort()

    aggregation_statistics = list(set([d['aggregation_statistic'] for d in datasets1]))
    aggregation_statistics.sort()

    frequency_intervals = list(set([d['frequency_interval'] for d in datasets1]))
    frequency_intervals.sort()

    utc_offsets = list(set([d['utc_offset'] for d in datasets1]))
    utc_offsets.sort()

    return make_options(features), make_options(parameters), make_options(methods), make_options(product_codes), make_options(owners), make_options(aggregation_statistics), make_options(frequency_intervals), make_options(utc_offsets), dataset_id


@app.callback(
    Output('sites_summ', 'children'),
    [Input('dataset_id', 'children'), Input('date_sel', 'start_date'), Input('date_sel', 'end_date')])
@cache.memoize()
def update_summ_data(dataset_id, start_date, end_date):
    if dataset_id is None:
        print('No new sites_summ')
    else:
        # tethys = decode_obj(tethys_obj)
        summ_r = requests.post(base_url + 'get_stations', params={'dataset_id': dataset_id, 'compression': 'zstd'})

        dc = zstd.ZstdDecompressor()
        summ_data1 = orjson.loads(dc.decompress(summ_r.content).decode())

        # summ_data1 = tethys.get_stations(dataset_id)

        if 'from_date' in summ_data1[0]['stats']:
            [s.update({'time_range': {'from_date': s['stats']['from_date'], 'to_date': s['stats']['to_date']}}) for s in summ_data1]

        summ_data2 = [s for s in summ_data1 if (pd.Timestamp(s['time_range']['to_date']).tz_localize(None) > pd.Timestamp(start_date)) and (pd.Timestamp(s['time_range']['from_date']).tz_localize(None) < pd.Timestamp(end_date))]
        [s.update({'ref': ''}) for s in summ_data2 if not 'ref' in s]
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
        sites_options = [{'label': s['ref'], 'value': s['station_id']} for s in sites_summ1]

        return sites_options


@app.callback(
        Output('site-map', 'figure'),
        [Input('sites_summ', 'children')],
        [State('site-map', 'figure')])
@cache.memoize()
def update_display_map(sites_summ, figure):
    if sites_summ is None:
        # print('Clear the sites')
        data1 = figure['data'][0]
        if 'hoverinfo' in data1:
            data1.pop('hoverinfo')
        data1.update(dict(size=8, color='black', opacity=0))
        fig = dict(data=[data1], layout=figure['layout'])
    else:
        sites_summ1 = orjson.loads(sites_summ)

        lon1 = [l['geometry']['coordinates'][0] for l in sites_summ1]
        lat1 = [l['geometry']['coordinates'][1] for l in sites_summ1]
        [l.update({'name': ''}) for l in sites_summ1 if not 'name' in l]
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
        site1_id = orjson.loads(sites_summ)[site1_index]['station_id']
    else:
        site1_id = ''

    # print(sites1_id)

    return site1_id


@app.callback(
    Output('summ_table', 'data'),
    [Input('sites_summ', 'children'), Input('sites', 'value'), Input('site-map', 'selectedData'), Input('site-map', 'clickData')],
    [State('datasets', 'children'), State('dataset_id', 'children')])
def update_table(sites_summ, sites, selectedData, clickData, datasets, dataset_id):
    if sites_summ:
        new_summ = orjson.loads(sites_summ)
        datasets1 = orjson.loads(datasets)
        dataset1 = [d for d in datasets1 if d['dataset_id'] == dataset_id][0]

        if sites:
            new_summ1 = [s for s in new_summ if s['station_id'] in sites]
        else:
            new_summ1 = new_summ

        [l.update({'name': ''}) for l in new_summ1 if not 'name' in l]

        summ_table = build_table(new_summ1, dataset1)

        return summ_table


@app.callback(
    Output('ts_data', 'children'),
    [Input('sites', 'value'), Input('date_sel', 'start_date'), Input('date_sel', 'end_date'), Input('dataset_id', 'children')],
    )
@cache.memoize()
def get_data(sites, start_date, end_date, dataset_id):
    if dataset_id:
        if sites:
            # tethys = decode_obj(tethys_obj)
            ts_r = requests.get(base_url + 'get_results', params={'dataset_id': dataset_id, 'station_id': sites, 'compression': 'zstd', 'from_date': start_date+'T00:00', 'to_date': end_date+'T00:00', 'squeeze_dims': True})
            dc = zstd.ZstdDecompressor()
            ts1 = xr.Dataset.from_dict(orjson.loads(dc.decompress(ts_r.content).decode()))
            # ts1 = tethys.get_results(dataset_id, sites, from_date=start_date, to_date=end_date, squeeze_dims=True)

            ts1_obj = encode_obj(ts1)

            return ts1_obj



@app.callback(
    Output('selected-data', 'figure'),
    [Input('ts_data', 'children')],
    [State('sites', 'value'), State('dataset_id', 'children'), State('date_sel', 'start_date'), State('date_sel', 'end_date')])
@cache.memoize()
def display_data(ts_data, sites, dataset_id, start_date, end_date):

    base_dict = dict(
            data = [dict(x=0, y=0)],
            layout = dict(
                title='Click on the map to select a station',
                paper_bgcolor = '#F4F4F8',
                plot_bgcolor = '#F4F4F8'
                )
            )

    if not sites:
        return base_dict

    if not dataset_id:
        return base_dict

    ts1 = decode_obj(ts_data)

    # x1 = ts1['coords']['time']['data']
    x1 = pd.to_datetime(ts1.time)
    # if not isinstance(x1, (list, np.ndarray)):
    #     x1 = [x1]
    # data_vars = ts1['data_vars']
    # data_vars = list(ts1.variables)
    parameter = [t for t in ts1 if 'dataset_id' in ts1[t].attrs][0]
    y1 = ts1[parameter].values
    # if not isinstance(y1, (list, np.ndarray)):
    #     y1 = [y1]

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
# @cache.memoize()
def update_ds_table(dataset_id, datasets):
    if dataset_id:
        # dataset_table_cols = {'license': 'Data License', 'attribution': 'Attribution'}

        datasets1 = orjson.loads(datasets)

        dataset1 = [d for d in datasets1 if d['dataset_id'] == dataset_id][0]

        if 'attribution' in dataset1:
            attr = dataset1['attribution']
        else:
            attr = 'Data sourced from ' + dataset1['owner']

        dataset_table1 = {'Data License': dataset1['license'], 'Attribution': attr}
        # [dataset_table1.update({v: dataset1[k]}) for k, v in dataset_table_cols.items()]

        return [dataset_table1]


# @app.callback(
#     Output('download-netcdf', 'data'),
#     Input("btn_netcdf", "n_clicks"),
#     [State('ts_data', 'children'), State('sites', 'value'), State('dataset_id', 'children')], prevent_initial_call=True)
# def download_netcdf(n_clicks, ts_data, sites, dataset_id):
#     if dataset_id:
#         if sites:
#             ts1 = orjson.loads(ts_data)
#             ts2 = xr.Dataset.from_dict(ts1)
#             # b1 = io.BytesIO(ts2.to_netcdf())
#             b1 = ts2.to_netcdf()
#
#             return dcc.send_bytes(b1, 'tsdata.nc')


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    [State('ts_data', 'children'), State('sites', 'value'), State('dataset_id', 'children')], prevent_initial_call=True)
def download_csv(n_clicks, ts_data, sites, dataset_id):
    if dataset_id:
        if sites:
            ts1 = decode_obj(ts_data)
            x1 = pd.to_datetime(ts1.time)
            parameter = [t for t in ts1 if 'dataset_id' in ts1[t].attrs][0]
            y1 = ts1[parameter].values

            ts2 = pd.DataFrame({'from_date': x1, parameter: y1})

            # ts2['from_date'] = pd.to_datetime(ts2['from_date'])
            ts2.set_index('from_date', inplace=True)

            return dcc.send_data_frame(ts2.to_csv, "tsdata.csv")


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
    app.run_server(host='0.0.0.0', port=80)


# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8080)
