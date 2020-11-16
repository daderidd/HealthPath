import pandas as pd
import geopandas as gpd
import networkx as nx
import importlib
import osmnx as ox
import overpy
import json
import spaghetti
import matplotlib_scalebar
from matplotlib_scalebar.scalebar import ScaleBar
import splot
import matplotlib.pyplot as plt
import sys
import matplotlib.lines as mlines
from shapely.geometry import Point, Polygon
import libpysal
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import folium_static
import folium
import time,datetime

sys.path.append('/Users/david/Dropbox/PhD/Data/Databases/Community design/20201106-switzerland-osm.shp')
sys.path.append('/Users/david/Dropbox/PhD/Scripts/Spatial analyses/')
sys.path.append('/Users/david/Dropbox/PhD/GitHub/deltagiraph/Data/')
import toolbox
importlib.reload(toolbox)


data_folder = Path('./Data')


st.markdown("# HealthPath : Calculateur d'itinéraire pour la santé")

@st.cache()
def load_gdf(path):
	"""Load data into GeoDataFrame"""
	data = gpd.read_file(path)
	return data
@st.cache(allow_output_mutation=True)
def read_network():
	G = nx.read_gpickle('./Data/Street_geneve/G_poi.gpickle')
	return G
@st.cache(allow_output_mutation=True)
def load_data(path,DATE_COLUMN = None):
	"""Load data into DataFrame"""
	data = pd.read_csv(path)
	lowercase = lambda x: str(x).lower()
	data.rename(lowercase, axis='columns', inplace=True)
	if DATE_COLUMN is not None:
		data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
	return data

@st.cache(allow_output_mutation=True)
def transform_network(G,poi,poi_imped):
	gdf_nodes,gdf_edges = ox.graph_to_gdfs(G)
	# gdf_edges[poi_imped] = gdf_edges[poi].astype(int)*100 + gdf_edges['length'].astype(float)
	# G = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
	return gdf_edges

@st.cache(allow_output_mutation=True)
def make_route(origin,destination):
	origin_xy = list(zip(origin_address.lat, origin_address.lon))[0]
	dest_xy = list(zip(dest_address.lat, dest_address.lon))[0]

	orig = ox.get_nearest_node(G, origin_xy)
	dest = ox.get_nearest_node(G, dest_xy)
	route_opti = ox.shortest_path(G, orig, dest, weight=poi_imped)
	route_short = ox.shortest_path(G, orig, dest, weight='length')
	####
	route_nodes = list(zip(route_short[:-1], route_short[1:]))
	index = [
	    gdf_edges[(gdf_edges["u"] == u) & (gdf_edges["v"] == v)].index[0] for u, v in route_nodes
	]
	gdf_route_edges = gdf_edges.loc[index]

	return origin_xy,dest_xy,route_opti,route_short,gdf_route_edges

@st.cache(allow_output_mutation=True)
def make_fig(G,route_opti,poi):
	route_opti_map = ox.plot_route_folium(G, route_opti)
	# route_short_map = ox.plot_route_folium(G, route_short)
	feature_group = folium.FeatureGroup(name=[poi])
	folium.Marker(origin_xy, popup='Start',icon=folium.Icon(color='red')).add_to(route_opti_map)
	folium.Marker(dest_xy, popup='End',icon=folium.Icon(color='blue')).add_to(route_opti_map)
	for index, row in gdf.iterrows():
	    marker = folium.Circle(location = ([row.lat,row.lon]), fill=True,fill_opacity = 1,fill_color = 'red', color = 'red',radius = 5)
	#     folium.Circle(location = ([row.lat,row.lon]), fill=True,fill_opacity = 1,fill_color = 'red', color = 'red',radius = radius).add_to(marker_cluster)   
	    feature_group.add_child(marker)
	route_opti_map.add_child(feature_group)
	folium.Choropleth(gdf_route_edges,line_weight = 3, line_color = 'grey').add_to(route_opti_map)
	# createLegend(route_opti_map,['Restaurant','Shortest path','Healthy path'],['Red','Blue','Green'])
	# call to render Folium map in Streamlit
	return route_opti_map

@st.cache(allow_output_mutation=True)
def calc_path_measures(G,route_opti,route_short):
	route_opti_length = int(sum(ox.utils_graph.get_route_edge_attributes(G, route_opti, 'length')))
	route_opti_npoi = int(sum(ox.utils_graph.get_route_edge_attributes(G, route_opti, poi)))

	route_short_length = int(sum(ox.utils_graph.get_route_edge_attributes(G, route_short, 'length')))
	route_short_npoi = int(sum(ox.utils_graph.get_route_edge_attributes(G, route_short, poi)))
	detour = route_opti_length - route_short_length
	diff_poi = route_short_npoi - route_opti_npoi
	return route_opti_length,route_opti_npoi,route_short_length,route_short_npoi,detour,diff_poi

## Choose POI

# get all the parks in some neighborhood
# constrain acceptable `leisure` tag values to `park`

_poi = st.sidebar.selectbox('Sélectionner les lieux à éviter: ',['Restaurants','Fast-Foods','Bars'])
poi_dict = {'Restaurants':'restaurant','Fast-Foods':'fast_food','Bars':'bar'}
poi = poi_dict[_poi]
poi_imped = poi+'_'+'impedance'
poi_filename = poi+'.csv'
##################
t0 = time.time()

G = read_network()
gdf_edges = load_gdf(data_folder/'gdf_edges.geojson')
# gdf_edges = gpd.GeoDataFrame(gdf_edges,geometry = gdf_edges['geometry'])
# gdf_nodes = load_data(data_folder/'gdf_nodes.csv')

t1 = time.time()
total = t1-t0
st.write(str(total),'Chargement du réseau') 

##############
# slider_impedance = st.sidebar.slider('Quelle distance feriez-vous pour éviter un {}'.format(poi), 0.0, 200.0, 100.0, step = 25.0)

t0 = time.time()

regbl_ge = load_data(data_folder/'regbl_ge.csv')
# G,gdf_edges = transform_network(G,poi, poi_imped)
# gdf_edges = transform_network(G,poi, poi_imped)

gdf = load_data(data_folder/'POIs'/poi_filename)


adresse_a = st.sidebar.text_input('Lieu de départ')
adresse_b = st.sidebar.text_input("Lieu d'arrivée")


t1 = time.time()
total = t1-t0
st.write(str(total),'Transformation du réseau') 



if adresse_a != '' and adresse_b != '':

	origin_address, dest_address = regbl_ge[regbl_ge.address == adresse_a],regbl_ge[regbl_ge.address == adresse_b]
	# st.markdown(origin_address)

	origin_xy,dest_xy,route_opti,route_short,gdf_route_edges = make_route(origin_address,dest_address)
	
	# compare the two routes

	route_opti_length,route_opti_npoi,route_short_length,route_short_npoi,detour,diff_poi = calc_path_measures(G,route_opti,route_short)

	# message_1 = "L'itinéraire santé est de {} mètres".format(route_opti_length)
	message_2 = "En empruntant l'itinéraire Santé, vous marchez {} mètres en plus et vous évitez de passer devant {} {}".format(detour,diff_poi, _poi.lower())
	# message_3 = 'HealthPath is passing {} {}'.format(route_opti_npoi, _poi.lower())
	# message_4 = 'Shortest path is passing {} {}'.format(route_short_npoi, _poi.lower())
	#Make fig and display it
	folium_static(make_fig(G,route_opti,poi))
	# st.markdown(message_1)
	st.markdown(message_2)
	# st.markdown(message_3)
	# st.markdown(message_4)

