import pickle 
import numpy as np
import os
import pandas as pd
import tqdm
import random
import time
import matplotlib.pyplot as plt
from os.path import join as pjoin
import geopandas as gpd
from cartoframes.viz import *
from shapely.geometry import MultiPoint, Point, Point, LineString, Polygon
import datetime
from utils.constants import GRID_UNIT_METER
import osmnx as ox
from core.astar import a_star
from shapely.ops import linemerge, nearest_points

class UrbanOSM:
    def __init__(self, sensor_df, osm_file_path, coefficients=[1, 0.9, 0.8], mcolors=['#5555FF', '#FF55FF', '#55ee55']):
        self.sensor_gdf = gpd.GeoDataFrame(
            sensor_df, geometry=gpd.points_from_xy(x=sensor_df.lng, y=sensor_df.lat)
        )
        self.sensor_gdf.crs = 'epsg:4326'
        self.sensor_gdf_3310 = self.sensor_gdf.to_crs('epsg:3310')
        self.total_bounds = self.sensor_gdf_3310.total_bounds
        self.id2geo = {str(sid):geo for sid, geo in zip(self.sensor_gdf['sid'], self.sensor_gdf.geometry)}
        if not os.path.isfile(osm_file_path):
            self._retrieve_street_network()
            ox.save_graphml(self.street_graph, filepath=osm_file_path)
            print('OSM street network saved at '+osm_file_path)
        else:
            self.street_graph = ox.load_graphml(filepath=osm_file_path)
        
        self.osm_nodes, osm_edges = ox.graph_to_gdfs(self.street_graph)
        self.osm_nodes['osmidn'] = self.osm_nodes.index
        self.osm_nodes['osmidstr'] = self.osm_nodes['osmidn'].astype(str)
        self.osm_nodes.crs = 'epsg:4326'
        self.osm_nodes_3310 = self.osm_nodes.to_crs('epsg:3310')

        osm_edges = osm_edges.reset_index()
        cond = np.array([str(type(s)) for s in osm_edges['highway']]) == "<class 'str'>"
        self.osm_edges = osm_edges[cond]
        self.osm_edges.crs = 'epsg:4326'
        self.osm_edges_3310 = self.osm_edges.to_crs('epsg:3310')
        self._set_ways()

        assert len(coefficients) == len(mcolors)
        self.coefficients = coefficients
        self.mcolors = mcolors

    
    def _set_ways(self):
        self.osm_motorway = self.osm_edges[self.osm_edges['highway'].isin(['motorway',])]
        self.osm_motorway.crs = 'epsg:4326'
        self.osm_primary = self.osm_edges[self.osm_edges['highway'].isin(['motorway_link', 'primary', 'primary_link'])]
        self.osm_secondary = self.osm_edges[self.osm_edges['highway'].isin(['secondary', 'secondary_link'])]
        self.osm_others = self.osm_edges[~self.osm_edges['highway'].isin(['motorway','residential',
                                                        'motorway_link', 'primary', 'primary_link',
                                                  'secondary', 'secondary_link'])]

    def _retrieve_street_network(self):
        '''
            Retrieve the street network for the location
        '''
        multipoint = MultiPoint(self.sensor_gdf.geometry)
        sensor_hull = multipoint.convex_hull

        sensor_hull_gdf = gpd.GeoDataFrame(geometry=[sensor_hull])
        sensor_hull_gdf.crs = 'epsg:4326'
        sensor_hull_gdf_3310 = sensor_hull_gdf.to_crs('epsg:3310')
        sensor_hull_3310 = sensor_hull_gdf_3310.iloc[0].geometry

        x1, y1, x2, y2 = sensor_hull_gdf.total_bounds

        sensor_center_latitude = (y1 + y2)/2
        sensor_center_longitude = (x1 + x2)/2 
        center_point = gpd.GeoDataFrame(geometry = [Point(sensor_center_longitude, sensor_center_latitude)])
        center_point.crs = 'epsg:4326'
        center_point = center_point.to_crs('epsg:3310')

        max_distance = self.sensor_gdf.to_crs('epsg:3310').distance(center_point.iloc[0].geometry).max()+GRID_UNIT_METER*2
        self.street_graph = ox.graph_from_point((sensor_center_latitude, sensor_center_longitude), dist=max_distance, network_type="drive")

    def match_sensors(self):
        '''
            Match sensor to OSM edges
            Update self.sensor_gdf
            Return Map to compare previous sensors and matched sensors
        '''
        new_items = []
        closest_line_list = []
            
        for _, item in tqdm.tqdm(self.sensor_gdf.iterrows(), total=len(self.sensor_gdf)): 
            closest_edge = self.osm_motorway.iloc[self.osm_motorway.distance(item.geometry).argmin()]
            closest_line = closest_edge.geometry
            closest_line_list.append(closest_line)
            closest_point_on_line, closest_point_on_point = nearest_points(closest_line, item.geometry)
            nitem = dict(item)
            nitem['geometry'] = closest_point_on_line
            nitem['u'] = str(closest_edge['u'])
            nitem['v'] = str(closest_edge['v'])
            nitem['uv'] = str(closest_edge['u']) + '-' + str(closest_edge['v'])
            new_items.append(nitem)
        
        new_sensor_gdf = gpd.GeoDataFrame(new_items)
        new_sensor_gdf.crs='epsg:4326'

        print('sensor gdf matched and updated')
        print('returning previous sensors as pink and matched sensors as red')
        map = Map([
            Layer(self.osm_others,  basic_style(color='#bbbbbb'), encode_data=False),
            Layer(self.osm_secondary,  basic_style(color='#777777'), encode_data=False),
            Layer(self.osm_primary,  basic_style(color='black'), encode_data=False),
            Layer(self.osm_motorway,  basic_style(color='blue'), encode_data=False),
            Layer(gpd.GeoDataFrame(geometry=closest_line_list)),
            Layer(self.sensor_gdf, basic_style(color='pink'),
                popup_click=[popup_element('sid')], popup_hover=[popup_element('sid')]),
            Layer(new_sensor_gdf, basic_style(color='red'),
                popup_click=[popup_element('sid')], popup_hover=[popup_element('sid')]),
        ])

        self.sensor_gdf = new_sensor_gdf
        self.sensor_gdf_3310 = self.sensor_gdf.to_crs('epsg:3310')

        return map

    def setup_graph(self):
        '''
            Setup graph
        '''
        self.osm_edges_non_res_3310 = self.osm_edges_3310[self.osm_edges_3310['highway'] != 'residential'].copy()
        self.path_dict = dict()
        for _, item in self.osm_edges_non_res_3310.iterrows():
            self.path_dict[(str(item['u']), str(item['v']))] = item.geometry
       
        osmid2pos = {osmidstr: (x, y) for osmidstr, y, x in 
                zip(self.osm_nodes_3310['osmidstr'], self.osm_nodes_3310.geometry.y, self.osm_nodes_3310.geometry.x)}
        self.graph_list = dict()
        for coef in self.coefficients:
            graph = dict()
            for _, item in self.osm_edges_non_res_3310.iterrows():
                us = str(item['u'])
                vs = str(item['v'])
                
                if us not in osmid2pos or vs not in osmid2pos:
                    continue
                dist = item['length']

                if item['highway'] == 'motorway':
                    dist *= coef

                graph.setdefault(us, {'pos': osmid2pos[us]})
                graph.setdefault(vs, {'pos': osmid2pos[vs]})
                graph[us][vs] = dist
            self.graph_list[coef] = graph

        appeared_nodes = []
        for graph in self.graph_list.values():
            appeared_nodes.extend(list(graph.keys()))
        appeared_nodes = list(set(appeared_nodes))

        nodes = self.osm_nodes_3310[self.osm_nodes_3310['osmidstr'].isin(appeared_nodes)]
        self.osmid2geo_3310 = {osmid:geo for osmid, geo in zip(self.osm_nodes_3310['osmidstr'], self.osm_nodes_3310['geometry'])}
        elem_list = nodes['osmidstr'].tolist()

        edges = self.osm_edges_non_res_3310.copy()
        edges['edgeid'] = range(len(edges))
        self.uv2edgeid = {(str(u), str(v)):eid for u, v, eid in zip(edges['u'], edges['v'], edges['edgeid'])}

        self.navigate_nodes = nodes 
        self.navigate_elem_list = elem_list
        self.navigate_edges = edges

        print('Used for navigation nodes and edges:', len(elem_list), len(edges))

    def _get_example_navigate_map(self, O, D, example_gdf_list, basemap='positron', show_grid=False):
        layers = [
                Layer(self.osm_motorway,  basic_style(color='black'))
            ] + [
                Layer(example_gdf, basic_style(color=self.mcolors[i], size = 5)) for i, example_gdf in enumerate(example_gdf_list)
            ] + [
                Layer(self.sensor_gdf, basic_style(color='gray'),
                popup_click=[popup_element('sid')], popup_hover=[popup_element('sid')]),
            ] + [
                Layer(self.navigate_nodes[self.navigate_nodes['osmidstr'].isin([O])]
                    , basic_style(color='red', size=20))
            ]  + [
                Layer(self.navigate_nodes[self.navigate_nodes['osmidstr'].isin([D])]
                    , basic_style(color='orange', size=20))
            ]
        
        if show_grid:
            layers = [Layer(self.grid_gdf[self.grid_gdf['num_nodes'] > 0], basic_style(color='#FFFFFFFF', opacity=0, stroke_width=2, stroke_color='green'))] + layers
        
        return Map(
            layers,
            basemap=basemap
        )
    
    def _get_grid_map(self, grid_gdf):
        return Map([
            Layer(grid_gdf[grid_gdf['num_nodes'] > 0], color_continuous_style('num_nodes', opacity=.5), popup_click=popup_element('idx')),
            Layer(self.sensor_gdf)
            
        ])

    def navigate_example(self, O='123031567', D='14956249', basemap={'style'}, show_grid=False):
        print(f'Sample navigation from {O} to {D}')
        example_gdf_list = []
         
        for ik, k in enumerate(self.graph_list):
            path = a_star(self.osmid2geo_3310, self.graph_list[k], O, D)
            path_list = []
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                path_list.append(self.path_dict[(u, v)])

            example_gdf = gpd.GeoDataFrame(geometry=path_list, crs='epsg:3310')
            if ik < 2:
                example_gdf.geometry = example_gdf.geometry.translate(ik*100, ik*100)
            example_gdf_list.append(example_gdf)
        
        return self._get_example_navigate_map(O, D, example_gdf_list, basemap, show_grid)
    

    def setup_grid(self):
        x1, y1, x2, y2 = self.total_bounds
        GRID_W = int((x2 - x1) / GRID_UNIT_METER) + 1
        GRID_H = int((y2 - y1) / GRID_UNIT_METER) + 1

        grid_width = (x2 - x1) / GRID_W
        grid_height = (y2 - y1) / GRID_H

        x1 = x1 - grid_width
        y1 = y1 - grid_height
        x2 = x2 + grid_width
        y2 = y2 + grid_height

        grid_items = []
        for j in range(GRID_H+2):
            for i in range(GRID_W+2):
                by, bx = y1 + j*grid_height, x1 + i*grid_width
                
                # Define the vertices of the square
                vertices = [(bx, by), (bx, by + grid_height), 
                            (bx + grid_width, by + grid_height), (bx + grid_width, by)]
                
                # Create a polygon object from the vertices
                square = Polygon(vertices)
                grid_items.append(square)

        grid_gdf = gpd.GeoDataFrame(geometry=grid_items, crs='epsg:3310')
        grid_gdf['idx'] = range(len(grid_gdf))

        grid_idx_elems = dict()
        for _, item in tqdm.tqdm(grid_gdf.iterrows(), total=len(grid_gdf)):
            idx = item['idx']
            grid_idx_elems[idx] = self.navigate_nodes[self.navigate_nodes.intersects(item.geometry)]['osmidstr'].tolist()
        
        self.grid_idx_elems = grid_idx_elems
        grid_gdf['num_nodes'] = [len(grid_idx_elems[idx]) for idx in grid_gdf['idx'].tolist()]

        # Rough path existence check

        grid_connectivity = np.eye(len(grid_gdf))
        for _, item in tqdm.tqdm(self.navigate_edges.iterrows(), total=len(self.navigate_edges)):
            connected_grids = np.arange(len(grid_gdf))[grid_gdf.intersects(item.geometry)]
            for k, idx in enumerate(connected_grids):
                for jdx in connected_grids[k+1:]:
                    grid_connectivity[idx, jdx] = grid_connectivity[jdx, idx] = 1

        new_connectivity = grid_connectivity.copy()
        prev_connectivity = np.zeros_like(new_connectivity)
        trial = 0
        while np.sum(new_connectivity != prev_connectivity) != 0:
            trial += 1
            prev_connectivity = new_connectivity > 0
            new_connectivity = (new_connectivity + new_connectivity @ new_connectivity) > 0
        self.grid_connectivity = new_connectivity
        self.grid_gdf = grid_gdf

        return self._get_grid_map(grid_gdf)
    
    def generate_paths(self, filename, max_trial=3):
        if os.path.isfile(filename):
            with open(filename, 'r') as fp:
                paths = fp.readlines()
            self.discover_path_list = [path.strip().split() for path in paths]
            print(f'Paths loaded from {filename}')
            return
        start_time = time.time()
        discover_path_list = []
        with open(filename, 'w') as fp:
            for i in tqdm.tqdm(range(len(self.grid_gdf))):
                for j in range(len(self.grid_gdf)):
                    if i == j or self.grid_connectivity[i, j] == 0:
                        continue
                    if len(self.grid_idx_elems[i]) < 5 or len(self.grid_idx_elems[j]) < 5:
                        continue

                    for graph in self.graph_list.values():
                        path = None
                        for _ in range(max_trial): #max trial
                            rn1 = random.choice(self.grid_idx_elems[i])
                            rn2 = random.choice(self.grid_idx_elems[j])
                            while rn1 == rn2:
                                rn2 = random.choice(self.grid_idx_elems[j])

                            path = a_star(self.osmid2geo_3310, graph, rn1, rn2)
                            if path:
                                discover_path_list.append(path)
                                fp.write(' '.join(path) + '\n')
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f"{len(discover_path_list)} paths discovered")

        self.discover_path_list = discover_path_list
    
    def setup_sensor_words(self):
        mitem_list = dict()
        for _, item in self.sensor_gdf_3310.iterrows():
            edge = self.navigate_edges[self.navigate_edges['highway'] == 'motorway']
            found_edge = edge.distance(item.geometry).argmin()
                    
            mitem = dict(edge.iloc[found_edge])
            
            mitem_list.setdefault(found_edge, mitem)
            mitem_list[found_edge].setdefault('sid2dist', dict())
            
            node_u = self.navigate_nodes[self.navigate_nodes['osmidstr'] == str(mitem['u'])].iloc[0]
            sid = item['sid']
            dist = node_u.geometry.distance(item.geometry)
            mitem_list[found_edge]['sid2dist'][sid] = dist
            
            
        synch_gdf = gpd.GeoDataFrame([mitem_list[k] for k in mitem_list])
        sensor_words = []
        total = 0
        for _, item in synch_gdf.iterrows():
            w_list = []
            d = item['sid2dist']
            for w in sorted(d, key=d.get, reverse=False):
                w_list.append('S' + str(w))
                total += 1
            sensor_words.append(' '.join(w_list))
        synch_gdf['sensors'] = sensor_words
        self.sensor_words = sensor_words
        self.eid2sw  = {eid:sw for eid, sw in zip(synch_gdf['edgeid'], synch_gdf['sensors'])}
        sid2eid = dict()
        for eid, sw in self.eid2sw.items():
            sids = sw.split()
            for sid in sids:
                sid2eid[sid] = eid
        self.sid2eid = sid2eid
    
    def setup_path_sentences(self):
        motorway_edges = self.navigate_edges[self.navigate_edges['highway'] == 'motorway']
        motorway_list = {u + '-' + v:-1 for u, v in zip(motorway_edges['u'].astype(str).tolist(), motorway_edges['v'].astype(str).tolist())}

        highway_path_list = []
        for path in tqdm.tqdm(self.discover_path_list):
            used_highway = False
            for i, node_u in enumerate(path[:-1]):
                node_v = path[i+1]
                if node_u + '-' + node_v in motorway_list:
                    used_highway = True
                    break
            if used_highway:
                highway_path_list.append(path)
        
        all_path_sentences = []
        for path in tqdm.tqdm(highway_path_list):
            path_sentence = str(path[0])
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                eid = self.uv2edgeid[u, v]
                if eid in self.eid2sw:
                    path_sentence += ' ' + self.eid2sw[eid]
                path_sentence += ' ' + str(v)
            all_path_sentences.append(path_sentence)
        
        self.highway_path_list = highway_path_list
        self.path_sentences = all_path_sentences
    
    def track_path(self, path_sentence, sid1, sid2):
        track_switch = False
        track_paths = []
        for node in path_sentence.split():

            if not track_switch and node == sid1:
                track_switch = True
                track_paths.append(sid1)

            if track_switch and node[0] != 'S':
                track_paths.append(node)

            if node == sid2:
                track_paths.append(sid2)
                break
        return track_paths

    def sid_dist(self, sid1, sid2, track_paths):
        between_sid_dist = 0
        sgeo1 = self.sensor_gdf_3310[self.sensor_gdf_3310['sid'] == int(sid1[1:])].iloc[0].geometry
        sgeo2 = self.sensor_gdf_3310[self.sensor_gdf_3310['sid'] == int(sid2[1:])].iloc[0].geometry
        if len(track_paths) == 2:
            between_sid_dist = sgeo1.distance(sgeo2)
        else:
            edge1 = self.navigate_edges[self.navigate_edges['edgeid'] == self.sid2eid[sid1]].iloc[0]
            edge2 = self.navigate_edges[self.navigate_edges['edgeid'] == self.sid2eid[sid2]].iloc[0]

            edge1_v = self.navigate_nodes[self.navigate_nodes['osmidstr'] == str(edge1.v)].iloc[0].geometry
            edge2_u = self.navigate_nodes[self.navigate_nodes['osmidstr'] == str(edge2.u)].iloc[0].geometry

            between_sid_dist += sgeo1.distance(edge1_v)
            between_sid_dist += edge2_u.distance(sgeo2)

            rest_paths = track_paths[1:-1]
            for i, node_u in enumerate(rest_paths[:-1]):
                node_v = rest_paths[i+1]
                node_u, node_v = int(node_u), int(node_v)
                between_sid_dist += self.navigate_edges[(self.navigate_edges['u'] == node_u) & (self.navigate_edges['v'] == node_v)].iloc[0].geometry.length
                
        return between_sid_dist
    
    def save_geojson(self, file_root):
        item_list = []
        geo_list = []
        uid = 0
        for sen in tqdm.tqdm(self.path_sentences):
            pathcpy = [s for s in sen.split()]
            path_sensors = [s for s in sen.split() if s[0] == 'S']
            path_nodes = [s for s in sen.split() if s[0] != 'S']

            path_list = []
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                path_list.append(self.path_dict[(u, v)])

            ls_list = []
            for ls in path_list[:-1]:
                ls_list.extend(list(ls.coords)[:-1])
            ls_list.extend(list(path_list[-1].coords))
            geo = LineString(ls_list)
            geo_list.append(geo)
            
            uid += 1
            mitem = {
                'path': ','.join(pathcpy),
                'path_sensors': ','.join(path_sensors),
                'path_nodes': ','.join(path_nodes),
                'key': f'UID{uid}',
                'geometry': geo
            }
            item_list.append(mitem)

            gdf = gpd.GeoDataFrame(item_list)
        gdf.crs = 'epsg:4326'

        arr = np.arange(len(gdf))
        np.random.shuffle(arr)

        for n in [1000, 3000, 5000, 10000]:
            gdf[['path_sensors', 'key', 'geometry']].iloc[arr[:n]].to_file(pjoin(file_root, f'{n}.json'))

        self.sensor_gdf['ssid'] = 'S' + self.sensor_gdf['sid'].astype(str)
        self.sensor_gdf.to_file(pjoin(file_root, 'sensors.json'))

        valdict = dict()
        for sen in gdf['path_sensors']:
            if len(sen) == 0:
                continue
            for s in sen.split(','):
                valdict.setdefault(s, 0)
                valdict[s] += 1
                
        self.sensor_gdf['count'] = [valdict[s] for s in self.sensor_gdf['ssid']]
        
        return Map([
            Layer(gdf[['path_sensors', 'key', 'geometry']].iloc[arr[:1000]], basic_style(opacity=.1)),
            Layer(self.sensor_gdf, color_continuous_style('count', palette='sunset'))
        ])
