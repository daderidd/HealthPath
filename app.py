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

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}], )
server = app.server

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


def transform_network(G, poi, poi_imped):
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    # gdf_edges[poi_imped] = gdf_edges[poi].astype(int)*100 + gdf_edges['length'].astype(float)
    # G = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
    return gdf_edges

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
def make_route(origin, destination,selected_poi_impedance):
    origin_xy = list(zip(origin.lat, origin.lon))[0]
    dest_xy = list(zip(destination.lat, destination.lon))[0]
    orig = ox.get_nearest_node(G, origin_xy)
    dest = ox.get_nearest_node(G, dest_xy)
    route_health = ox.shortest_path(G, orig, dest, weight=selected_poi_impedance)
    route_short = ox.shortest_path(G, orig, dest, weight='length')
    ####
    gdf_route_edges_short = getRouteEdges(route_short)
    gdf_route_edges_health = getRouteEdges(route_health)


    return origin_xy, dest_xy, route_health, route_short, gdf_route_edges_short,gdf_route_edges_health



def lineToPoints(gdf):
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
            names = np.append(names, [name]*len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)
    return names,lons,lats
def calc_path_measures(G, route_health, route_short,poi):
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
# gdf_edges = load_gdf(data_folder/'gdf_edges.geojson')
gdf_edges = ox.graph_to_gdfs(G,nodes = False)
regbl_df = load_data(data_folder/'regbl_ge.csv')
addresses = sorted(regbl_df.address.unique())
options = []
for address in addresses:
    option = {'label':address,'value':address}
    options.append(option)

# Define variables

METERS = [25, 50, 75, 100, 125, 150, 175, 200]
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

df = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')





app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                html.Img(id="logo", src=app.get_asset_url("logo_transparent.png"), height=120, width=120),
                html.H2(children="HealthPath - Reduce the caloric pressure around you"),
                html.P(
                    id="description",
                    children="HealthPath makes you able to find itineraries avoiding \
                    certain types of locations such has fast-foods, restaurants, bars.",
                ),
            ],
        ),
        html.Div(
            id = 'app-container',
            children = [
                html.Div(
                    id = 'left-column',
                    children = [
                        html.Div(
                            id = 'address-container',
                            children=[
                                html.H4('What is your itinerary?'),
                                html.Label(["Starting point", dcc.Dropdown(id="dynamic-start")]),
                                html.Label(["Destination", dcc.Dropdown(id="dynamic-end")]),
                            ],
                        ),

                        html.Div(
                            id="dropdown-amenity",
                            children=[
                                html.H4('What would you like to avoid on your way?'),
                                dcc.Dropdown(id = 'poi-selection',
                                    options=[
                                        {'label': 'Fast-foods', 'value': 'fast_food'},
                                        {'label': 'Restaurants', 'value': 'restaurant'},
                                        {'label': 'Bars', 'value': 'bar'}
                                    ],
                                    value='fast_food',
                                ),
                            ],
                        ),
                        html.Div(
                            id="slider-container",
                            children=[
                                html.H4(
                                    id="slider-text"
                                ),
                                dcc.Slider(
                                    id="years-slider",
                                    min=min(METERS),
                                    max=max(METERS),
                                    value=min(METERS),
                                    marks={str(dist): dict(label=str(dist)+'m', style={"color": "#7fafdf"}) for dist in
                                           METERS},
                                    step=None,
                                ),
                            ],
                        ),
                        html.Div([html.H4(id = 'output_address')]
                    ),
                ],
            ),

            html.Div(
                id='map-container',
                children=[
                    dcc.Graph(
                        id='map-route',
                        figure=dict(
                            data=[dict(
                                type='scattermapbox',
                                marker=[dict(size=0, color='white', opacity=0)]
                            )],
                            layout=dict(
                                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                                height = 600,
                                mapbox=dict(
                                    layers=[],
                                    style='carto-darkmatter',
                                    center=dict(
                                        lat=46.22, lon=6.14
                                    ),
                                    pitch=0,
                                    zoom=10.3,
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

@app.callback(
    Output("slider-text", "children"),
    [Input("poi-selection","value")])
def return_poi(selected_poi):
    poi_dict = {'restaurant':'restaurant', 'fast_food':'fast-food','bar':'bar'}
    text = "What additional distance are you ok to walk to avoid a {} ?".format(poi_dict[selected_poi])
    return text

@app.callback(
    [Output("output_address", "children"),Output("map-route", "figure")],
    [Input("dynamic-start","value"),Input("dynamic-end",'value'),Input('poi-selection','value')],
    [State("map-route", "figure")])

def update_figure(start,end,selected_poi,figure):
    poi_filename = selected_poi + '.csv'
    poi_imped = selected_poi + '_' + 'impedance'
    poi_dict = {'restaurant':'restaurants', 'fast_food':'fast-Foods','bar':'bars'}

    if start != '' and end != '':
        origin_address, dest_address = regbl_df[regbl_df.address == start], regbl_df[regbl_df.address == end]
        origin_address['type'],dest_address['type'] = 'Starting point','Destination'
        origin_address['color'], dest_address['color'] = 'Red', 'Blue'
        startend = pd.concat([origin_address,dest_address])
        origin_xy, dest_xy, route_health, route_short, gdf_route_edges_short,gdf_route_edges_health = make_route(origin_address, dest_address,poi_imped)
        route_health_length, route_health_npoi, route_short_length, route_short_npoi, detour, diff_poi = calc_path_measures(G, route_health, route_short,selected_poi)
        message = "HealthPath makes you avoid passing by {} {} and walk just {} extra meters ".format(diff_poi, poi_dict[selected_poi],detour)
        x, y = gdf_route_edges_health.unary_union.centroid.xy


        gdf_poi=load_data(data_folder/'POIs'/poi_filename)
        gdf_poi['type'] = poi_dict[selected_poi].capitalize()

        names_health,lons_health,lats_health = lineToPoints(gdf_route_edges_health)
        names_short,lons_short,lats_short = lineToPoints(gdf_route_edges_short)
        # lats_cleaned,lons_cleaned = [x for x in list(lats_health) if str(x) != 'None'],[x for x in list(lons_health) if str(x) != 'None']
        print(startend.lon,startend.lat)
        fig = go.Figure()

        fig.add_trace(go.Scattermapbox(
            # One improvement would be to adjust markers' lon and lat to the start and end of the route, not the address
            mode="markers+text",
            lon=startend.lon,
            lat = startend.lat,
            marker=go.scattermapbox.Marker(size= 15, symbol= ["marker",'marker']),
            text=startend.type,
            showlegend = False,
            textposition="bottom right",
            hoverinfo = 'text',
            textfont=dict(
                family="sans serif",
                size=14,
                color="crimson")
            ))

        fig.add_trace(go.Scattermapbox(
            name = 'Original itinerary',
            mode = 'lines',
            lon = lons_short,
            lat = lats_short,
            hovertext = names_short,
        ))
        fig.add_trace(go.Scattermapbox(
            name = 'HealthPath',
            mode = 'lines',
            lon = lons_health,
            lat = lats_health,
            hovertext=names_health,
        ))

        fig.add_trace(go.Scattermapbox(
            name = poi_dict[selected_poi].capitalize(),
            mode = 'markers',
            marker_color = 'crimson',
            lat=gdf_poi.lat,
            lon=gdf_poi.lon,
            hovertext =gdf_poi.name,
        ))


        fig.update_layout(
            mapbox = dict(
            center= {'lon': x[0], 'lat': y[0]}, #Centroid of Healthpath
            style="carto-darkmatter",
            pitch = 0,
            zoom = 12.5),
            paper_bgcolor= '#191e26',
            autosize=True,
            legend = dict(
            title_font_family="Open Sans",
            font=dict(
            family="Helvetica",
            size=12,
            color="white")),
            margin={"r": 0, "t": 0, "l": 0, "b": 0})

        return message,fig


if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1')
