#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:55:56 2021

@author: mike
"""
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import requests
import dash_leaflet as dl
import dash_leaflet.express as dlx
import copy

import utils

##########################################
### Parameters

map_height = 400

lat1 = -41.2
lon1 = 172.5
zoom1 = 6

extra_text = """
## Tethys Dataset Discovery

Included in this application are the public datasets stored in the [Tethys data management system](https://tethysts.readthedocs.io/). The preferred method for accessing the data is through the [Tethys Python package](https://tethysts.readthedocs.io/). The documentation describes the system as well as providing examples for accessing the data. The system and the Python package are currently under heavy development and may (will) contain errors. Please report all issues to the [tethysts Github repo](https://github.com/tethys-ts/tethysts/issues).

Please be aware of the data owner, license, and attribution before using the data in other work.
"""

tabs_styles = {
    'height': '40px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '5px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '5px'
}


###############################################
### App layout


def layout1():

    datasets = requests.get(utils.base_url + 'get_datasets').json()

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
            dcc.Tabs(id='intro_tabs', value='info_tab', style=tabs_styles, children=[
                dcc.Tab(label='Introduction', value='info_tab', id='info_tab', style=tab_style, selected_style=tab_selected_style, children=[
                    dcc.Markdown(extra_text, id='docs')
                    ]),
                dcc.Tab(label='Filters', value='filter_tab', id='filter_tab', style=tab_style, selected_style=tab_selected_style, children=[
                    # html.P(children='Filters:'),
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
                    ])
                ])
            ], className='two columns', style={'margin': 10}),

    html.Div([
        html.P('... Datasets Available', style={'display': 'inline-block'}, id='ds_count'),
        dash_table.DataTable(
        id='ds_table',
        columns=[{"name": v, "id": k} for k, v in utils.ds_table_cols.items()],
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
            dcc.Tabs(id='data_tabs', value='dataset_tab', style=tabs_styles, children=[
                dcc.Tab(label='Dataset metadata', value='dataset_tab', id='dataset_tab', style=tab_style, selected_style=tab_selected_style, children=[
                    dcc.Markdown('Click on a dataset row above', id='dataset_meta', style={"overflow-y": "scroll", 'height': 400})
                    # style={"whitespace": "pre", "overflow-x": "scroll"}
                    ]),
                dcc.Tab(label='Station data', value='stn_tab', id='stn_tab', style=tab_style, selected_style=tab_selected_style, children=[
                    dcc.Markdown('Click on a station on the map', id='stn_meta', style={"overflow-y": "scroll", 'height': 400})
                    # style={"whitespace": "pre", "overflow-x": "scroll"}
                    ])
                ])
            ])

    ], className='fourish columns', style={'margin': 10}),

    html.Div([
        dcc.Tabs(id='map_tabs', value='extent_tab', style=tabs_styles, children=[
            dcc.Tab(label='Dataset Extent', value='extent_tab', id='extent_tab', style=tab_style, selected_style=tab_selected_style, children=[
                dl.Map(center=[lat1, lon1], zoom=zoom1, children=[
                        dl.TileLayer(),
                        dl.GeoJSON(data={}, id='extent_map', zoomToBoundsOnClick=True)
                    ], style={'width': '100%', 'height': 780, 'margin': "auto", "display": "block"}, id="map1"),
                ]),
            dcc.Tab(label='Station locations', value='stn_loc_tab', id='stn_loc_tab', style=tab_style, selected_style=tab_selected_style, children=[
                dl.Map(center=[lat1, lon1], zoom=zoom1, children=[
                        dl.TileLayer(),
                        dcc.Loading(id="loading_station_map",  type="default", children=
                            dl.GeoJSON(data={}, id='stn_map', cluster=True, zoomToBoundsOnClick=True)
                        )
                    ], style={'width': '100%', 'height': 780, 'margin': "auto", "display": "block"}, id="map2")
                ]),
            dcc.Tab(label='Time series plot', value='ts_tab', id='ts_tab', style=tab_style, selected_style=tab_selected_style, children=[
                dcc.Loading(
                    id="loading_1",
                    type="default",
                    children=dcc.Graph(
                        id = 'selected_data',
                        figure = dict(
                            data = [dict(x=0, y=0)],
                            layout = dict(
                                    paper_bgcolor = '#F4F4F8',
                                    plot_bgcolor = '#F4F4F8',
                                    height = 400
                                    )
                            ),
                        config={"displaylogo": False},
                        style={'width': '100%', 'height': 780, 'margin': "auto", "display": "block"}
                        )
                    )
                ])
            ]),
    ], className='fourish columns', style={'margin': 10}),
    dcc.Store(id='datasets_obj', data=utils.encode_obj(requested_datasets)),
    dcc.Store(id='filtered_datasets_obj', data=utils.encode_obj(requested_datasets)),
    dcc.Store(id='dataset_id', data=''),
    dcc.Store(id='station_id', data=''),
    dcc.Store(id='station_obj', data=''),
    dcc.Store(id='result_obj', data='')
], style={'margin':0})

    return layout

