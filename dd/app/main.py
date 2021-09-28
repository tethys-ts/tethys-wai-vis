# -*- coding: utf-8 -*-
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
import numpy as np
import requests
import zstandard as zstd
import codecs
import pickle
import dash_leaflet as dl
import dash_leaflet.express as dlx
import copy

pd.options.display.max_columns = 10

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# server = flask.Flask(__name__)
# app = dash.Dash(__name__, server=server,  url_base_pathname = '/')

app = dash.Dash(__name__,  url_base_pathname = '/')
server = app.server

##########################################
### Parameters

# base_url = 'https://api.tethys-ts.xyz/tethys/data/'
base_url = 'http://tethys-api-ext:80/tethys/data/'

ds_table_cols = {'feature': 'Feature', 'parameter': 'Parameter', 'method': 'Method', 'owner': 'Owner', 'product_code': 'Product Code', 'aggregation_statistic': 'Agg Stat', 'frequency_interval': 'Freq Interval', 'utc_offset': 'UTC Offset'}

map_height = 500

lat1 = -43.45
lon1 = 171.9
zoom1 = 6

mapbox_access_token = "pk.eyJ1IjoibXVsbGVua2FtcDEiLCJhIjoiY2pudXE0bXlmMDc3cTNxbnZ0em4xN2M1ZCJ9.sIOtya_qe9RwkYXj5Du1yg"


extra_text = """
## Tethys Dataset Discovery

Included in this application are the public datasets stored in the [Tethys data management system](https://tethysts.readthedocs.io/). The preferred method for accessing the data is through the [Tethys Python package](https://tethysts.readthedocs.io/). The documentation describes the system as well as providing examples for accessing the data. The system and the Python package are currently under heavy development and may (will) contain errors. Please report all issues to the [tethysts Github repo](https://github.com/tethys-ts/tethysts/issues).

Please be aware of the data owner, license, and attribution before using the data in other work.
"""


###########################################
### Functions


def encode_obj(obj):
    """

    """
    cctx = zstd.ZstdCompressor(level=1)
    c_obj = codecs.encode(cctx.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)), encoding="base64").decode()


    return c_obj


def decode_obj(str_obj):
    """

    """
    dctx = zstd.ZstdDecompressor()
    obj1 = dctx.decompress(codecs.decode(str_obj.encode(), encoding="base64"))
    d1 = pickle.loads(obj1)

    return d1


def filter_datasets(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, start_date, end_date, datasets):
    """

    """
    dataset_list = datasets.copy()

    if features:
        dataset_list = [d for d in dataset_list if d['feature'] in features]
    if parameters:
        dataset_list = [d for d in dataset_list if d['parameter'] in parameters]
    if methods:
        dataset_list = [d for d in dataset_list if d['method'] in methods]
    if product_codes:
        dataset_list = [d for d in dataset_list if d['product_code'] in product_codes]
    if owners:
        dataset_list = [d for d in dataset_list if d['owner'] in owners]
    if aggregation_statistics:
        dataset_list = [d for d in dataset_list if d['aggregation_statistic'] in aggregation_statistics]
    if frequency_intervals:
        dataset_list = [d for d in dataset_list if d['frequency_interval'] in frequency_intervals]
    if utc_offsets:
        dataset_list = [d for d in dataset_list if d['utc_offset'] in utc_offsets]

    ## Date filter
    dataset_list2 = []

    for d in dataset_list:
        if 'time_range' in d:
            if pd.Timestamp(d['time_range']['to_date']) >  pd.Timestamp(start_date):
                if pd.Timestamp(d['time_range']['from_date']) < pd.Timestamp(end_date):
                    dataset_list2.append(d)

    return dataset_list2


def build_table(datasets):
    """

    """
    table1 = []

    for dataset in datasets:
        ds = {k: dataset[k] for k in ds_table_cols}
        ds['id'] = dataset['dataset_id']
        table1.append(ds)

    return table1


def build_md_ds(dataset):
    """

    """
    ds1 = {'**' + str(k) + '**': v for k, v in dataset.items()}
    # ds2 = orjson.dumps(ds1, option=orjson.OPT_INDENT_2).decode()

    ds3 = '\n\n'.join([str(k) + ': ' + str(v) for k, v in ds1.items()])

    return ds3







###############################################
### App layout

map_layout = dict(mapbox = dict(layers = [], accesstoken = mapbox_access_token, style = 'outdoors', center=dict(lat=lat1, lon=lon1), zoom=zoom1), margin = dict(r=0, l=0, t=0, b=0), autosize=True, hovermode='closest', height=map_height)

# @server.route('/wai-vis')
# def main():
def serve_layout():

    datasets = requests.get(base_url + 'get_datasets').json()

    # datasets = [ds for ds in datasets if ds['method'] != 'simulation']

    requested_datasets = sorted(datasets, key=lambda k: k['owner'])

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

    ## Fix negative longitudes
    # for d in requested_datasets:
    #     if 'extent' in d:
    #         extent = d['extent']
    #         coordinates = []
    #         for e in extent['coordinates'][0]:
    #             c1 = [abs(e[0]), e[1]]
    #             coordinates.append(c1)
    #         extent['coordinates'] = [coordinates]

    ### prepare summaries and initial states
    max_date = pd.Timestamp.now()
    start_date = max_date - pd.DateOffset(years=100)

    layout = html.Div(children=[
    html.Div([
        html.P(children='Filters:'),
        html.Label('Features'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in features], value=[], id='features', multi=True),
        html.Label('Parameters'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in parameters], value=[], id='parameters', multi=True),
        html.Label('Methods'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in methods], value=[], id='methods', multi=True),
        html.Label('Data Owners'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in owners], value=[], id='owners', multi=True),
        html.Label('Product Codes'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in product_codes], value=[], id='product_codes', multi=True),
        html.Label('Aggregation Statistics'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in aggregation_statistics], value=[], id='aggregation_statistics', multi=True),
        html.Label('Frequency Interval'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in frequency_intervals], value=[], id='frequency_intervals', multi=True),
        html.Label('UTC Offsets'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in utc_offsets], value=[], id='utc_offsets', multi=True),
        html.Label('Date Range'),
        dcc.DatePickerRange(
            end_date=str(max_date.date()),
            display_format='YYYY-MM-DD',
            start_date=str(start_date.date()),
            id='date_sel'
#               start_date_placeholder_text='DD/MM/YYYY'
            ),
        ],
    className='two columns', style={'margin': 10}),

    html.Div([
        html.P('... Datasets Available', style={'display': 'inline-block'}, id='ds_count'),
        dash_table.DataTable(
        id='ds_table',
        columns=[{"name": v, "id": k} for k, v in ds_table_cols.items()],
        data=[],
        row_selectable="single",
        selected_rows=[],
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        fixed_rows={'headers': True},
        style_table={'height': 350},  # defaults to 500
        style_cell={
            'minWidth': '60px', 'maxWidth': '160px',
            'whiteSpace': 'normal'
        }
        ),
        html.Div([
        dcc.Markdown('', id='meta', style={"overflow-y": "scroll", 'height': 400})
        # style={"whitespace": "pre", "overflow-x": "scroll"}
        ])

    ], className='five columns', style={'margin': 10}),

    html.Div([

        dl.Map(center=[lat1, lon1], zoom=zoom1, children=[
                dl.TileLayer(),
                dl.GeoJSON(data={}, id='extent', zoomToBoundsOnClick=True)
            ], style={'width': '100%', 'height': map_height, 'margin': "auto", "display": "block"}, id="map"),
        dcc.Markdown(extra_text, id='docs')
    ], className='four columns', style={'margin': 10}),
    dcc.Store(id='datasets_obj', data=encode_obj(requested_datasets)),
    dcc.Store(id='filtered_datasets_obj', data=encode_obj(requested_datasets)),
    dcc.Store(id='dataset_id', data='')
], style={'margin':0})

    return layout


app.layout = serve_layout

########################################
### Callbacks


@app.callback(
    Output('filtered_datasets_obj', 'data'), [Input('features', 'value'), Input('parameters', 'value'), Input('methods', 'value'), Input('product_codes', 'value'), Input('owners', 'value'), Input('aggregation_statistics', 'value'), Input('frequency_intervals', 'value'), Input('utc_offsets', 'value'), Input('date_sel', 'start_date'), Input('date_sel', 'end_date')],
    [State('datasets_obj', 'data')])
def update_filtered_datasets(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, start_date, end_date, datasets_obj):
    """

    """
    datasets = decode_obj(datasets_obj)

    dataset_list = filter_datasets(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, start_date, end_date, datasets)

    return encode_obj(dataset_list)


@app.callback(
    Output('ds_table', 'data'), [Input('filtered_datasets_obj', 'data')],
    )
def update_dataset_table(filtered_datasets_obj):
    """

    """
    datasets = decode_obj(filtered_datasets_obj)

    table1 = build_table(datasets)

    return table1


@app.callback(
    Output('ds_count', 'children'), [Input('filtered_datasets_obj', 'data')],
    )
def update_dataset_count(filtered_datasets_obj):
    """

    """
    datasets = decode_obj(filtered_datasets_obj)

    len1 = len(datasets)

    count_text = str(len1) + ' Datasets Available'

    return count_text


@app.callback(
    Output('dataset_id', 'data'),
    # Output('meta', 'children'),
    [Input('ds_table', 'selected_row_ids')]
    )
def update_dataset_id(ds_id):
    """

    """
    # print(ds_id)

    if isinstance(ds_id, list):
        ds_id = ds_id[0]
    elif ds_id is None:
        ds_id = ''

    return ds_id


@app.callback(
    Output('meta', 'children'),
    [Input('dataset_id', 'data')],
    [State('datasets_obj', 'data')])
def update_meta(ds_id, datasets_obj):
    """

    """
    if len(ds_id) > 1:
        datasets = decode_obj(datasets_obj)
        dataset = [d for d in datasets if d['dataset_id'] == ds_id][0]
        [dataset.pop(d) for d in ['extent', 'properties'] if d in dataset]

        text = build_md_ds(dataset)

        # text = orjson.dumps(dataset, option=orjson.OPT_INDENT_2).decode()
        # text = pprint.pformat(dataset, 2)
    else:
        text = 'Click on a dataset row'

    return text


@app.callback(
    Output('extent', 'data'),
    [Input('dataset_id', 'data')],
    [State('datasets_obj', 'data')])
def update_map(ds_id, datasets_obj):
    """

    """
    if len(ds_id) > 1:
        datasets = decode_obj(datasets_obj)
        dataset = [d for d in datasets if d['dataset_id'] == ds_id][0]
        # print(dataset)
        if 'extent' in dataset:
            extent1 = {'type': 'FeatureCollection',
                         'features': [{'type': 'Feature',
                           'geometry': dataset['extent']}]}
        else:
            extent1 = {}
    else:
        extent1 = {}

    return extent1


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=80)


# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8080)
