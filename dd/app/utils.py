#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:00:41 2021

@author: mike
"""
import zstandard as zstd
import codecs
import pickle
import pandas as pd
import requests
import xarray as xr
import orjson

#####################################
### Parameters

ds_table_cols = {'feature': 'Feature', 'parameter': 'Parameter', 'method': 'Method', 'owner': 'Owner', 'product_code': 'Product Code', 'aggregation_statistic': 'Agg Stat', 'frequency_interval': 'Freq Interval', 'utc_offset': 'UTC Offset'}

# base_url = 'https://api.tethys-ts.xyz/tethys/data/'
base_url = 'http://tethys-api-ext:80/tethys/data/'

#####################################
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


def get_stations(base_url, dataset_id):
    """

    """
    resp_fn_stns = requests.post(base_url + 'get_stations', params={'dataset_id': dataset_id, 'compression': 'zstd'})

    if not resp_fn_stns.ok:
        raise ValueError(resp_fn_stns.raise_for_status())

    dctx = zstd.ZstdDecompressor()

    fn_stns = orjson.loads(dctx.decompress(resp_fn_stns.content))

    return fn_stns


def get_results(base_url, dataset_id, station_id):
    """

    """
    resp_fn_results = requests.get(base_url + 'get_results', params={'dataset_id': dataset_id, 'compression': 'zstd', 'station_id': station_id})

    if not resp_fn_results.ok:
        raise ValueError(resp_fn_results.raise_for_status())

    dctx = zstd.ZstdDecompressor()

    fn_results = orjson.loads(dctx.decompress(resp_fn_results.content))

    data2 = xr.Dataset.from_dict(fn_results)

    return data2


def stns_to_geojson(stns):
    """

    """
    gj = {'type': 'FeatureCollection', 'features': []}
    for s in stns:
        if 'name' in s:
            sgj = {'type': 'Feature', 'geometry': s['geometry'], 'properties': {'name': s['station_id'], 'tooltip': s['name']}}
        elif 'ref' in s:
            sgj = {'type': 'Feature', 'geometry': s['geometry'], 'properties': {'name': s['station_id'], 'tooltip': s['ref']}}
        else:
            sgj = {'type': 'Feature', 'geometry': s['geometry'], 'properties': {'name': s['station_id']}}
        gj['features'].append(sgj)

    return gj






