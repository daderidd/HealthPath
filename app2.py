import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import sys
from pathlib import Path
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
import shapely.geometry
import numpy as np
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
sys.path.append('/Users/david/Dropbox/PhD/Data/Databases/Community design/20201106-switzerland-osm.shp')
sys.path.append('/Users/david/Dropbox/PhD/Scripts/Spatial analyses/')
sys.path.append('/Users/david/Dropbox/PhD/GitHub/deltagiraph/Data/')

data_folder = Path('./Data')

# Initialize app

app = dash.Dash(__name__,prevent_initial_callbacks=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}], )
server = app.server




url = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'
attribution = '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '
click_positions = []
if len(click_positions)>2:
    click_positions = []

app.layout = html.Div(
    [
    dl.Map(
        [
            dl.TileLayer(
                url=url,
                maxZoom=20,
                attribution=attribution
            ),
            dl.LayerGroup(
                id="position"
            )
        ]
        ,id="map", style={
            'width': '100%',
            'height': '50vh',
            'margin': "auto",
            "display": "block"}
        )
    ]
)


@app.callback(Output("position", "children"), [Input("map", "click_lat_lng")])
def map_click(dbl_click_lat_lng):
    click_positions.append(dbl_click_lat_lng)
    print(click_positions[-2:])
    popup = "Start ({:.3f}, {:.3f})".format(*dbl_click_lat_lng)
    return [dl.Marker(position=dbl_click_lat_lng, children=dl.Tooltip(popup))]





if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1')
