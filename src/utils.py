import pandas as pd


def get_mapping_locations_series(df: pd.Dataframe):
    '''
    Retrieve type of tennis series from a location
    
    :param pd.DataFrame df: datas about tennis, must have at least columns "Location" and "Series"
    :return dict: 
    '''
    mapping_location_series = {}
    for location in df["Location"].unique():
        type_serie = df[df["Location"]==location]["Series"].mode()[0]
        mapping_location_series[location] = type_serie
    return mapping_location_series


