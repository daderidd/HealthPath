import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import sys
from pathlib import Path
import dash
from dash import dcc
from dash import html
from shapely.geometry import Point
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
# import plotly.express as px
import plotly
from dash.dependencies import Input, Output, State
import shapely.geometry
import numpy as np
import os
from toolbox import connect_poi
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

sys.path.append('/Users/david/Dropbox/PhD/Data/Databases/Community design/20201106-switzerland-osm.shp')
sys.path.append('/Users/david/Dropbox/PhD/Scripts/Spatial analyses/')
sys.path.append('/Users/david/Dropbox/PhD/GitHub/deltagiraph/Data/')

data_folder = Path('./Data')

# Initialize app

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}], )
server = app.server
app.title = 'HealthPath'


# Mapbox

# mapbox_access_token = mapbox_key()
# mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"


# Functions

def load_gdf(path):
    """Load data into GeoDataFrame"""
    data = gpd.read_file(path)
    return data


def read_network():
    G = nx.read_gpickle('./Data/Street_geneve/G_poi.gpickle')
    return G


def load_data(path, DATE_COLUMN=None):
    """Load data into DataFrame"""
    data = pd.read_csv(path)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    if DATE_COLUMN is not None:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def get_pois(selected_poi):
    """ Returns a GeoDataFrame containing all the POI categories in select_poi.

        This function takes a list of POIs as input and returns a GeoDataFrame containing
        all inputted types. If a POI input is not already existing in the POIs folder, the
        function queries OSM and creates the file in the expected folder."""
    poi_filenames = [poi + '.csv' for poi in selected_poi]
    poi_paths = [data_folder / 'POIs' / poi_filename for poi_filename in poi_filenames]
    place = 'Canton de Genève, Switzerland'
    print('get_pois is running')
    for poi, path, filename in zip(selected_poi, poi_paths, poi_filenames):
        if os.path.exists(path):
            print(path, 'File existed')
        else:
            print(path, 'Querying OSM for the missing tag')
            if 'bakery' in filename:
                type_poi = 'shop'
            elif 'tree' in filename:
                type_poi = 'natural'
            else:
                type_poi = 'amenity'
            tags = {type_poi: poi}
            print(tags)
            gdf = ox.geometries_from_place(place, tags)
            gdf.index = gdf.index.map(int)
            gdf['id'] = gdf.index
            if 'element_type' in gdf.columns:
                gdf = gdf[gdf.element_type == 'node']
            gdf['lon'] = gdf.geometry.x
            gdf['lat'] = gdf.geometry.y
            gdf.to_csv(path, index=False)
    df_poi = pd.concat((pd.read_csv(f) for f in poi_paths)).reset_index(drop=True)
    df_poi['category'] = 'Unhealthy'
    if 'natural' in df_poi.columns:
        df_poi.loc[df_poi.natural == 'tree', 'category'] = 'Healthy'
        df_poi.loc[df_poi.natural == 'tree', 'poi_category'] = df_poi.natural
    else:
        df_poi['natural'] = np.NaN
    if 'shop' in df_poi.columns:
        df_poi.loc[(df_poi.shop.isnull() == False), 'poi_category'] = df_poi.shop
    else:
        df_poi['shop'] = np.NaN
    if 'amenity' in df_poi.columns:
        df_poi.loc[(df_poi.amenity.isnull() == False), 'poi_category'] = df_poi.amenity
    else:
        df_poi['amenity'] = np.NaN

    geometry = [Point(xy) for xy in zip(df_poi.lon, df_poi.lat)]
    crs = 'epsg:4326'
    df_poi = gpd.GeoDataFrame(df_poi, crs=crs, geometry=geometry)
    df_poi = df_poi.to_crs('epsg:2056')
    return df_poi


def calculate_impedance(gdf_edges, good_pois, bad_pois, selected_poi, impedance_distance):
    print('calculate_impedance is running')
    unhealthy_cols = intersection(selected_poi, bad_pois)
    healthy_cols = intersection(selected_poi, good_pois)
    if all(elem in good_pois for elem in selected_poi):
        impedance = gdf_edges['length'].astype(float) - gdf_edges[healthy_cols].astype(int).sum(axis=1) * 10
    elif all(elem in bad_pois for elem in selected_poi):
        impedance = gdf_edges['length'].astype(float) + gdf_edges[unhealthy_cols].astype(int).sum(axis=1) \
                    * impedance_distance
    else:
        impedance = gdf_edges['length'].astype(float) - gdf_edges[healthy_cols].astype(int).sum(axis=1) * 10 \
                    + gdf_edges[unhealthy_cols].astype(int).sum(axis=1) * impedance_distance
    impedance.loc[impedance < 0] = 0
    return impedance


def getRouteEdges(route):
    """Input : OSMNX route
       Output : List of edges that make up the route
       Requires : Geodataframe of all edges (OSMNX street network)
    """
    route_nodes = list(zip(route[:-1], route[1:]))
    index = [
        gdf_edges[(gdf_edges["u"] == u) & (gdf_edges["v"] == v)].index[0] for u, v in route_nodes
    ]
    gdf_route_edges = gdf_edges.loc[index]
    return gdf_route_edges


def make_route(origin, destination, selected_poi_impedance):
    """
    Returns the shortest and health routes.

    This function returns two routes for the inputted road network, the selected start and end points and impedance.
    Shortest route : Impedance simply based on length.
    HealthPath : Impedance based on a modified impedance integrating the POIs.
    """
    origin_xy = list(zip(origin.lat, origin.lon))[0]
    dest_xy = list(zip(destination.lat, destination.lon))[0]
    orig = ox.get_nearest_node(G, origin_xy)
    dest = ox.get_nearest_node(G, dest_xy)
    route_health = ox.shortest_path(G, orig, dest, weight=selected_poi_impedance)
    route_short = ox.shortest_path(G, orig, dest, weight='length')
    ####
    gdf_route_edges_short = getRouteEdges(route_short)
    gdf_route_edges_health = getRouteEdges(route_health)

    return origin_xy, dest_xy, route_health, route_short, gdf_route_edges_short, gdf_route_edges_health


def linetopoints(gdf):
    lats = []
    lons = []
    names = []
    for feature, name in zip(gdf.geometry, gdf.name):
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            names = np.append(names, [name] * len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)
    return names, lons, lats


def calc_path_measures(G, route_health, route_short, poi):
    """
    This function computes few statistics for the generated routes:
        Total length (m) : obtained by summing the length of each segments of each route.
        N POIs (n): obtained by summing the number of encountered POIs along each route.
        Detour (m) : obtained by calculating the difference in distance between shortest path and HealthPath
        Diff POIs (n) : obtained by calculating the difference in number of POIs between shortest path and HealthPath
    """
    route_health_length = int(sum(ox.utils_graph.get_route_edge_attributes(G, route_health, 'length')))
    route_health_npoi = int(sum(ox.utils_graph.get_route_edge_attributes(G, route_health, poi)))

    route_short_length = int(sum(ox.utils_graph.get_route_edge_attributes(G, route_short, 'length')))
    route_short_npoi = int(sum(ox.utils_graph.get_route_edge_attributes(G, route_short, poi)))
    detour = route_health_length - route_short_length
    diff_poi = route_short_npoi - route_health_npoi
    return route_health_length, route_health_npoi, route_short_length, route_short_npoi, detour, diff_poi


# _poi = st.sidebar.selectbox('Sélectionner les lieux à éviter: ', ['Restaurants', 'Fast-Foods', 'Bars'])
# poi_dict = {'Restaurants': 'restaurant', 'Fast-Foods': 'fast_food', 'Bars': 'bar'}
# poi = poi_dict[_poi]
# poi_imped = poi + '_' + 'impedance'
# poi_filename = poi + '.csv'

# Load data
G = read_network()
# G_proj = ox.project_graph(G, to_crs = 'epsg:2056')
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
# gdf_nodes_proj,gdf_edges_proj = ox.graph_to_gdfs(G_proj)

regbl_df = pd.read_feather(data_folder / 'regbl_ge_2021.feather')
addresses = regbl_df.address.unique()

# Define list of options for address dropdown menus
options = [{'label': address, 'value': address} for address in addresses]

# Define variables
impedances = [50, 100, 150]
DEFAULT_COLORSCALE = [
    "#f2fffb",
    "#bbffeb",
    "#98ffe0",
    "#79ffd6",
    "#6df0c8",
    "#69e7c0",
    "#59dab2",
    "#45d0a5",
    "#31c194",
    "#2bb489",
    "#25a27b",
    "#1e906d",
    "#188463",
    "#157658",
    "#11684d",
    "#10523e",
]
# adresse_a = st.sidebar.text_input('Lieu de départ')
# adresse_b = st.sidebar.text_input("Lieu d'arrivée")

DEFAULT_OPACITY = 0.8

# df = pd.read_csv(
#     'https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/
#     raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')

app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                html.Img(id="logo", src=app.get_asset_url("logo_transparent.png"), width='150px'),
                html.H1(children="HealthPath - Reduce the caloric pressure around you"),
                html.P(
                    id="description",
                    children="HealthPath enables you to find itineraries going through the best natural \
                    and built environment for your health.",
                ),
            ],
        ),
        html.Div(
            id='app-container',
            children=[
                html.Div(
                    id='left-column',
                    children=[
                        html.Div(
                            id='address-container',
                            # style = {'zIndex':2},
                            children=[
                                html.H4('What is your itinerary?'),
                                html.Label(["Origin", dcc.Dropdown(id="dynamic-start",
                                                                   style=
                                                                   {
                                                                       'fontSize': '16px',
                                                                       'color': '#ffffff',
                                                                       # 'zIndex': 3,
                                                                       'backgroundColor': '#ffffff',

                                                                   },
                                                                   placeholder="Search an address"
                                                                   )]),

                                html.Label(["Destination", dcc.Dropdown(id="dynamic-end",
                                                                        style=
                                                                        {
                                                                            'fontSize': '16px',
                                                                            'color': '#ffffff',
                                                                            # 'zIndex': 3,
                                                                            'backgroundColor': '#ffffff',
                                                                        },
                                                                   placeholder="Search an address")]),
                            ],
                        ),

                        html.Div(
                            id="checklist-avoid",
                            # style={'zIndex': 3},

                            children=[
                                html.H4('What would you like to avoid along your itinerary?'),
                                dcc.Checklist(id='poi-selection-unhealthy',
                                              options=[
                                                  {'label': 'Fast-foods', 'value': 'fast_food'},
                                                  {'label': 'Restaurants', 'value': 'restaurant'},
                                                  {'label': 'Bars', 'value': 'bar'},
                                                  {'label': 'Bakeries', 'value': 'bakery'}

                                              ],
                                              value=['fast_food'],
                                              ),
                            ],
                        ),
                        html.Div(
                            id="slider-container",
                            # style = {'zIndex':4},
                            children=[
                                html.H4("How much do you want to avoid these?",
                                    id="slider-text"
                                ),
                                dcc.Slider(
                                    id="impedance-slider",
                                    min=min(impedances),
                                    max=max(impedances),
                                    value=100,
                                    marks={50: dict(label='A little', style={"color": "#7fafdf"}),
                                           100: dict(label='Moderately', style={"color": "#7fafdf"}),
                                           200: dict(label='A lot', style={"color": "#7fafdf"})},
                                    step=None,
                                ),
                            ],
                        ),
                        html.Div(
                            id="checklist-goto",
                            # style={'zIndex': 5},

                            children=[
                                html.H4('What would you like to see on your way?'),
                                dcc.Checklist(id='poi-selection-healthy',
                                              options=[
                                                  {'label': 'Trees', 'value': 'tree'},
                                              ],
                                              value=['tree'],
                                              ),
                            ],
                        ),
                        html.Button(id = 'itinerary-start', n_clicks = 0, children = 'Create itinerary'),
                        # dbc.Spinner(),

                        html.Div([html.H3(id='output_address')]
                        )
                    ],
                ),

                html.Div(
                    id='map-container',
                    # style = {'zIndex':1},
                    children=[dcc.Loading(
                        id="loading-1",
                        type="default",
                        children=html.Div(
                            id="loading-output-1")
                    ),
                        dcc.Graph(
                            id='map-route',
                            figure=dict(
                                data=[dict(
                                    type='scattermapbox',
                                    marker=[dict(size=0, color='white', opacity=0)]
                                )],
                                layout=dict(
                                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                                    mapbox=dict(
                                        layers=[],
                                        style='carto-darkmatter',
                                        center=dict(
                                            lat=46.22, lon=6.14
                                        ),
                                        pitch=0,
                                        zoom=10.8,
                                    ),
                                ),
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    dash.dependencies.Output("dynamic-start", "options"),
    [dash.dependencies.Input("dynamic-start", "search_value")],
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    return [o for o in options if search_value.lower() in o["label"]]


@app.callback(
    dash.dependencies.Output("dynamic-end", "options"),
    [dash.dependencies.Input("dynamic-end", "search_value")],
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    return [o for o in options if search_value.lower() in o["label"]]


# @app.callback(
#     Output("slider-text", "children"),
#     [Input("poi-selection-unhealthy", "value")])
# def return_poi(string):
#     return "How much do you want to avoid these?"


# @app.callback(
#     Output("output_address", "children"),
#     [Input("slider-container","value"),Input('poi-selection','value')])
# def transform_network(selected_poi,dist):
#     gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
#     poi_imped = selected_poi + '_' + 'impedance'
#     gdf_edges[poi_imped] = gdf_edges[poi].astype(int)*dist + gdf_edges['length'].astype(float)
#     G = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
#     return gdf_edges


# @app.callback(
#     [Output("output_address", "children"), Output("map-route", "figure")],
#     [Input("dynamic-start", "value"), Input("dynamic-end", 'value'), Input('poi-selection-unhealthy', 'value'),
#      Input('poi-selection-healthy', 'value'), Input("impedance-slider", "value")],
#     [State("map-route", "figure")])

@app.callback(
    [Output("output_address", "children"), Output("map-route", "figure")],
    [Input('itinerary-start','n_clicks')],
    [State("dynamic-start", "value"), State("dynamic-end", 'value'), State('poi-selection-unhealthy', 'value'),
     State('poi-selection-healthy', 'value'), State("impedance-slider", "value"),State("map-route", "figure")],
    prevent_initial_call = True)
def update_figure(n_clicks, start, end, selected_poi_unhealthy, select_poi_healthy, impedance_distance, figure):
    if start is None or end is None:
        return 'Veuillez indiquer une adresse de départ et de destination.', dash.no_update
    else:
        poi_dict = {'restaurant': 'restaurants', 'fast_food': 'fast-foods',
                    'bar': 'bars', 'tree': 'trees', 'bakery': 'bakeries'}
        selected_poi = selected_poi_unhealthy + select_poi_healthy
        selected_poi = sorted(selected_poi)
        bad_pois = ['fast_food', 'restaurant', 'cafe', 'bar', 'bakery']
        good_pois = ['tree']
        poi_impedance = 'impedance_' + '_'.join(i for i in selected_poi)

        # Calculate impedance for the selected options
        gdf_edges[poi_impedance] = calculate_impedance(gdf_edges, good_pois, bad_pois, selected_poi,
                                                       impedance_distance)
        edge_attribute = gdf_edges.set_index(['u', 'v', 'key'])[poi_impedance].to_dict()
        nx.set_edge_attributes(G, edge_attribute, poi_impedance)

        origin_address, dest_address = regbl_df[regbl_df.address == start], regbl_df[regbl_df.address == end]
        origin_address['type'], dest_address['type'] = 'Starting point', 'Destination'
        origin_address['color'], dest_address['color'] = 'Red', 'Blue'
        startend = pd.concat([origin_address, dest_address])
        origin_xy, dest_xy, route_health, route_short, gdf_route_edges_short, gdf_route_edges_health = make_route(
            origin_address, dest_address, poi_impedance)
        measures, messages = [], []
        for poi in selected_poi:
            route_health_length, route_health_npoi, route_short_length, route_short_npoi, detour, diff_poi \
                = calc_path_measures(G, route_health, route_short, poi)
            measure = [route_health_length, route_health_npoi, route_short_length, route_short_npoi, detour, diff_poi]
            if poi == 'tree':
                message = "HealthPath makes you walk by {} extra {} and walk just {} extra meters ".format(abs(diff_poi),
                    poi_dict[poi], detour)
            else:
                message = "HealthPath makes you avoid passing by {} {} and walk just {} extra meters ".format(diff_poi,
                    poi_dict[poi], detour)
            messages.append(message)
            measures.append(measure)
        message = '\n'.join(i for i in messages)
        x, y = gdf_route_edges_health.unary_union.centroid.xy

        gdf_poi = get_pois(selected_poi)

        names_health, lons_health, lats_health = linetopoints(gdf_route_edges_health)
        names_short, lons_short, lats_short = linetopoints(gdf_route_edges_short)
        # lats_cleaned,lons_cleaned = [x for x in list(lats_health) if str(x) != 'None'],[x for x in list(lons_health) if str(x) != 'None']
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            # One improvement would be to adjust markers' lon and lat to the start and end of the route, not the address
            mode="markers+text",
            lon=startend.lon,
            lat=startend.lat,
            marker=go.scattermapbox.Marker(size=15, color=startend['color']),
            text=startend.type,
            showlegend=False,
            textposition="bottom right",
            hoverinfo='text',
            textfont=dict(
                family="sans serif",
                size=16,
                color="crimson")
        ))

        fig.add_trace(go.Scattermapbox(
            name='Original itinerary',
            mode='lines',
            lon=lons_short,
            lat=lats_short,
            hovertext=names_short,
        ))
        fig.add_trace(go.Scattermapbox(
            name='HealthPath',
            mode='lines',
            lon=lons_health,
            lat=lats_health,
            hovertext=names_health,
            line=dict(width=4),
        ))

        fig.add_trace(go.Scattermapbox(
            name='Unhealthy features',
            mode='markers',
            marker_color='crimson',
            lat=gdf_poi[gdf_poi.category == 'Unhealthy'].lat,
            lon=gdf_poi[gdf_poi.category == 'Unhealthy'].lon,
            hovertext=gdf_poi.name,
        ))

        fig.add_trace(go.Scattermapbox(
            name='Healthy features',
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=3,
                color='green',
                opacity=0.7
            ),
            lat=gdf_poi[gdf_poi.category == 'Healthy'].lat,
            lon=gdf_poi[gdf_poi.category == 'Healthy'].lon,
            hovertext=gdf_poi.natural,
        ))

        fig.update_layout(
            mapbox=dict(
                center={'lon': x[0], 'lat': y[0]},  # Centroid of Healthpath
                style="carto-darkmatter",
                pitch=0,
                zoom=12.5),
            paper_bgcolor='#191e26',
            autosize=True,
            legend=dict(yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                title_font_family="Open Sans",
                font=dict(
                    family="Helvetica",
                    size=16,
                    color="white")),
            margin={"r": 0, "t": 0, "l": 0, "b": 0})

        return message, fig


if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1')
