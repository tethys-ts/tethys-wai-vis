#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:55:43 2021

@author: mike
"""
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd

from .app import app
from . import utils


################################################
### Callbacks

@app.callback(
    Output('filtered_datasets_obj', 'data'), [Input('features', 'value'), Input('parameters', 'value'), Input('methods', 'value'), Input('product_codes', 'value'), Input('owners', 'value'), Input('aggregation_statistics', 'value'), Input('frequency_intervals', 'value'), Input('utc_offsets', 'value'), Input('date_sel', 'start_date'), Input('date_sel', 'end_date')],
    [State('datasets_obj', 'data')])
def update_filtered_datasets(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, start_date, end_date, datasets_obj):
    """

    """
    datasets = utils.decode_obj(datasets_obj)

    dataset_list = utils.filter_datasets(features, parameters, methods, product_codes, owners, aggregation_statistics, frequency_intervals, utc_offsets, start_date, end_date, datasets)

    return utils.encode_obj(dataset_list)


@app.callback(
    Output('ds_table', 'data'), [Input('filtered_datasets_obj', 'data')],
    )
def update_dataset_table(filtered_datasets_obj):
    """

    """
    datasets = utils.decode_obj(filtered_datasets_obj)

    table1 = utils.build_table(datasets)

    return table1


@app.callback(
    Output('ds_count', 'children'), [Input('filtered_datasets_obj', 'data')],
    )
def update_dataset_count(filtered_datasets_obj):
    """

    """
    datasets = utils.decode_obj(filtered_datasets_obj)

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
    Output('dataset_meta', 'children'),
    [Input('dataset_id', 'data')],
    [State('datasets_obj', 'data')])
def update_meta(ds_id, datasets_obj):
    """

    """
    if len(ds_id) > 1:
        datasets = utils.decode_obj(datasets_obj)
        dataset = [d for d in datasets if d['dataset_id'] == ds_id][0]
        # [dataset.pop(d) for d in ['extent', 'properties'] if d in dataset]

        text = utils.build_md_ds(dataset)

        # text = orjson.dumps(dataset, option=orjson.OPT_INDENT_2).decode()
        # text = pprint.pformat(dataset, 2)
    else:
        text = 'Click on a dataset row above'

    return text


@app.callback(
    Output('extent_map', 'data'),
    [Input('dataset_id', 'data')],
    [State('datasets_obj', 'data')])
def update_map_extent(ds_id, datasets_obj):
    """

    """
    if len(ds_id) > 1:
        datasets = utils.decode_obj(datasets_obj)
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


@app.callback(
    Output('stations_obj', 'data'),
    [Input('dataset_id', 'data')]
    )
def update_station_obj(ds_id):
    """

    """
    if len(ds_id) > 1:
        stns = utils.get_stations(utils.base_url, ds_id)

        stns_obj = utils.encode_obj(stns)
    else:
        stns_obj = ''

    return stns_obj


@app.callback(
    Output('stn_map', 'data'),
    [Input('stations_obj', 'data')]
    )
def update_stations_map(station_obj):
    """

    """
    if len(station_obj) > 1:
        stns = utils.decode_obj(station_obj)

        stns_gj = utils.stns_to_geojson(stns)

    else:
        stns_gj = {}

    return stns_gj


@app.callback(
    Output('station_id', 'data'),
    [Input('stn_map', 'click_feature')]
    )
def update_station_id(feature):
    """

    """
    # print(ds_id)
    if feature is not None:
        if 'name' in feature['properties']:
            stn_id = feature['properties']['name']
        else:
            stn_id = ''
    else:
        stn_id = ''

    return stn_id


@app.callback(
    Output('station_obj', 'data'),
    [Input('station_id', 'data')],
    [State('stations_obj', 'data')])
def update_stn_obj(stn_id, stations_obj):
    """

    """
    if len(stn_id) > 1:
        stns = utils.decode_obj(stations_obj)
        stn = [d for d in stns if d['station_id'] == stn_id][0]
        data = utils.encode_obj(stn)

    else:
        data = ''

    return data


@app.callback(
    Output('stn_meta', 'children'),
    [Input('station_obj', 'data')])
def update_stn_meta(station_obj):
    """

    """
    if len(station_obj) > 1:
        stn = utils.decode_obj(station_obj)

        text = utils.build_md_ds(stn)
    else:
        text = 'Click on a station on the map'

    return text


@app.callback(
    Output('result_obj', 'data'),
    [Input('station_obj', 'data')],
    [State('dataset_id', 'data')])
def update_result_obj(station_obj, ds_id):
    """

    """
    if (len(ds_id) > 1) and (len(station_obj) > 1):
        stn = utils.decode_obj(station_obj)
        from_date, to_date = utils.stn_date_range(stn)
        res = utils.get_results(utils.base_url, ds_id, stn['station_id'], from_date, to_date)

        res_obj = utils.encode_obj(res)
    else:
        res_obj = ''

    return res_obj


@app.callback(
    Output('selected_data', 'figure'),
    [Input('result_obj', 'data')]
    )
def update_results_plot(result_obj):
    """

    """
    base_dict = dict(
            data = [dict(x=0, y=0)],
            layout = dict(
                title='Click on the map to select a station',
                paper_bgcolor = '#F4F4F8',
                plot_bgcolor = '#F4F4F8',
                height = 400
                )
            )

    if len(result_obj) > 1:
        results = utils.decode_obj(result_obj)
        vars1 = list(results.variables)
        parameter = [v for v in vars1 if 'dataset_id' in results[v].attrs][0]

        results1 = results.isel(height=0, drop=True)

        fig = go.Figure()

        if 'geometry' in results.dims:
            grps = results1.groupby('geometry')

            for geo, grp in grps:
                if 'name' in grp:
                    name = str(grp['name'].values)
                elif 'ref' in grp:
                    name = str(grp['ref'].values)
                else:
                    name=None

                times = pd.to_datetime(grp['time'].values)

                fig.add_trace(go.Scattergl(
                    x=times,
                    y=grp[parameter].values,
                    showlegend=True,
                    name=name,
        #                line={'color': col3[s]},
                    opacity=0.8))
        else:
            results2 = results1[parameter].isel(lat=0, lon=0, drop=True)

            times = pd.to_datetime(results2['time'].values)

            fig.add_trace(go.Scattergl(
                x=times,
                y=results2.values,
                showlegend=False,
                # name=name,
    #                line={'color': col3[s]},
                opacity=0.8))

        # to_date = times.max()
        # from_date = to_date - pd.DateOffset(months=6)

        layout = dict(paper_bgcolor = '#F4F4F8', plot_bgcolor = '#F4F4F8', showlegend=True, height=780, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(l=20, r=20, t=20, b=20))

        fig.update_layout(**layout)
        fig.update_xaxes(
            type='date',
            # range=[from_date.date(), to_date.date()],
            # rangeslider=dict(visible=True),
            # rangeslider_range=[from_date, to_date],
            # rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(step="all", label='1y'),
                    # dict(count=1, label="1 year", step="year", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward")
                    ])
                )
            )

        fig.update_yaxes(autorange = True, fixedrange= False)

    else:
        fig = base_dict

    return fig
