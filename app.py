# --- BEGIN: DSA ROUTING / MAP SECTION ---
import streamlit as st
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import json
import time

# Try to import osmnx/networkx; if not available, we will fall back to ORS HTTP API
try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except Exception as e:
    OSMNX_AVAILABLE = False

# Optional: set your OpenRouteService API key here (fallback)
ORS_API_KEY = st.secrets.get("ORS_API_KEY", None)  # store securely in Streamlit secrets or provide below
# ORS_API_KEY = "your-openrouteservice-api-key-here"  # <- if not using secrets

st.markdown("## 🗺️ DSA Patrol Routing — Shortest Path (Dijkstra)")

with st.expander("How this works (short)"):
    st.write("""
    - Enter Source and Destination addresses (place names) anywhere in India.
    - We geocode them to lat/lon, build a local road graph (OpenStreetMap) and run Dijkstra to find the shortest path (edge weight = length).
    - If osmnx is unavailable, we call OpenRouteService as fallback (requires ORS API key).
    - The route is shown on a satellite map (ESRI tiles).  
    """)

# Inputs
col_a, col_b = st.columns(2)
with col_a:
    source_input = st.text_input("Source (address or place name)", value="Connaught Place, New Delhi")
with col_b:
    dest_input = st.text_input("Destination (address or place name)", value="India Gate, New Delhi")

buffer_km = st.slider("Routing buffer radius (km)", min_value=1, max_value=10, value=3,
                      help="Graph area radius around midpoint; increase if route is long / crosses wide area.")

run_route = st.button("Find Shortest Patrol Route (DSA)")

def geocode_address(address, user_agent="crime-patrol-app"):
    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    loc = geolocator.geocode(address)
    if loc:
        return (loc.latitude, loc.longitude)
    return None

def plot_route_folium(route_coords, source_pt, dest_pt, zoom=12):
    # ESRI Satellite tiles
    tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    m = folium.Map(location=[(source_pt[0]+dest_pt[0])/2, (source_pt[1]+dest_pt[1])/2], zoom_start=zoom, tiles=tiles, attr='Esri')
    folium.Marker(location=source_pt, tooltip="Source", icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker(location=dest_pt, tooltip="Destination", icon=folium.Icon(color='red', icon='flag')).add_to(m)
    folium.PolyLine(locations=route_coords, color='yellow', weight=5, opacity=0.8).add_to(m)
    return m

def route_with_osmnx(src_latlon, dst_latlon, buffer_km=3):
    # Build graph around midpoint with buffer radius (in meters)
    mid_lat = (src_latlon[0] + dst_latlon[0]) / 2
    mid_lon = (src_latlon[1] + dst_latlon[1]) / 2
    # distance in meters
    dist = int(buffer_km * 1000)
    # get drive network
    G = ox.graph_from_point((mid_lat, mid_lon), dist=dist, network_type='drive')
    # get nearest nodes to source/dest
    src_node = ox.nearest_nodes(G, src_latlon[1], src_latlon[0])
    dst_node = ox.nearest_nodes(G, dst_latlon[1], dst_latlon[0])
    # shortest path by length (Dijkstra)
    route = nx.shortest_path(G, src_node, dst_node, weight='length')  # Dijkstra under the hood
    # Extract coordinates
    route_coords = []
    for node in route:
        point = (G.nodes[node]['y'], G.nodes[node]['x'])
        route_coords.append(point)
    length_m = nx.shortest_path_length(G, src_node, dst_node, weight='length')
    return route_coords, length_m, G

def route_with_ors(src_latlon, dst_latlon, ors_key):
    # Calls OpenRouteService Directions API
    import openrouteservice
    client = openrouteservice.Client(key=ors_key)
    coords = ((src_latlon[1], src_latlon[0]), (dst_latlon[1], dst_latlon[0]))  # lon,lat
    resp = client.directions(coords, profile='driving-car', format='geojson')
    # parse geometry
    geom = resp['features'][0]['geometry']['coordinates']  # list of [lon,lat]
    route_coords = [(lat, lon) for lon, lat in geom]
    length_m = resp['features'][0]['properties']['summary']['distance']
    return route_coords, length_m, resp

if run_route:
    with st.spinner("Geocoding addresses…"):
        src = geocode_address(source_input)
        dst = geocode_address(dest_input)
    if not src or not dst:
        st.error("Could not geocode source or destination. Try more specific addresses.")
    else:
        st.success(f"Geocoded:\n Source: {src}\n Destination: {dst}")

        # Try osmnx method first
        if OSMNX_AVAILABLE:
            try:
                with st.spinner("Downloading map graph and computing Dijkstra route (OSM)… This can take 10–30s depending on area…"):
                    route_coords, length_m, G = route_with_osmnx(src, dst, buffer_km=buffer_km)
                st.success(f"Route found via OSM graph — length: {length_m:.0f} meters")
                m = plot_route_folium(route_coords, src, dst, zoom=12)
                st_map = st_folium(m, width=900, height=600)
                # Optional: show DSA details
                st.markdown("### DSA Notes (for viva)")
                st.write("- Graph nodes:", len(G.nodes))
                st.write("- Graph edges:", len(G.edges))
                st.write("- Algorithm used: Dijkstra (shortest path by edge length). Complexity: O(E + V log V) with priority queue.")
            except Exception as e:
                st.error(f"OSM routing failed: {e}")
                if ORS_API_KEY:
                    st.info("Falling back to OpenRouteService (ORS).")
                    try:
                        route_coords, length_m, resp = route_with_ors(src, dst, ORS_API_KEY)
                        st.success(f"Route found via ORS — length: {length_m:.0f} meters")
                        m = plot_route_folium(route_coords, src, dst, zoom=12)
                        st_folium(m, width=900, height=600)
                    except Exception as e2:
                        st.error(f"ORS fallback failed: {e2}")
                else:
                    st.info("No ORS API key provided. To use fallback routing, add ORS_API_KEY in Streamlit secrets or the code.")
        else:
            # osmnx not available — use ORS fallback if key present
            if ORS_API_KEY:
                try:
                    with st.spinner("Routing via OpenRouteService…"):
                        route_coords, length_m, resp = route_with_ors(src, dst, ORS_API_KEY)
                    st.success(f"Route found via ORS — length: {length_m:.0f} meters")
                    m = plot_route_folium(route_coords, src, dst, zoom=12)
                    st_folium(m, width=900, height=600)
                    st.markdown("### Note: OSMnx not available — using ORS directions API for routing.")
                    st.markdown("Algorithm: ORS directed shortest path (server-side).")
                except Exception as e:
                    st.error(f"ORS routing failed: {e}")
            else:
                st.error("OSMNX is not available in this environment and no ORS API key provided. To enable routing install osmnx or set ORS_API_KEY in Streamlit secrets.")
# --- END: DSA ROUTING / MAP SECTION ---
