from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from k_means_constrained import KMeansConstrained
from models import DataPoint, ClusteringRequestClusters, ClusteringRequestOutlets, ClusterData, ClusteringRequestWorkingHours, ClusteringResponseWorkingHours
from logging_config import logger
from middleware import LoggingMiddleware
#from clustering_utils import create_cluster, calculate_centroids, reassign_outliers

import numpy as np
import json 
import random
import math
import pandas as pd
from geopy.distance import geodesic


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

def generate_colors(n_clusters):
    color_list = []
    for _ in range(n_clusters):
        color_list.append('#%06X' % random.randint(0, 0xFFFFFF))
    return color_list

def prepare_clustering_data(data):
    X = np.array([(p.latitude, p.longitude) for p in data])
    return X, len(X)

def perform_clustering(X, total_data, n_clusters, size_min, size_max):
    try:
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=0, 
            n_jobs=-1
        )
        labels = clf.fit_predict(X)
        return labels , clf

    except Exception as e:  
        logger.error("Clustering failed:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

def build_clustering_response(labels, colors, data, clf):
    response = {
        "data": [],
        "cluster_definitions": []
    }

    cluster_defs = {}
    for i, label in enumerate(labels):
        color = colors[label.item()]
        center = clf.cluster_centers_[label.item()].tolist()

        cluster_id = label.item()
        if cluster_id not in cluster_defs:
            cluster_defs[cluster_id] = {
                "cluster": cluster_id,
                "color": color,
                "center": center
            }

        response["data"].append({
            "id_outlet": data[i].id_outlet,
            "latitude": data[i].latitude,
            "longitude": data[i].longitude,
            "cluster": label.item(),
            "color": color
        })

    response["cluster_definitions"] = list(cluster_defs.values())
    return {
        "status": "success",
        "data": response 
    } 

@app.post("/cluster/clusters")
async def perform_clustering_by_clusters(request: ClusteringRequestClusters):
    """
    Performs clustering based on the specified number of clusters.

    **Parameters:**

    * **request (ClusteringRequestClusters):**  The input request containing:
        * **data (list[DataPoint]):**  A list of data points with the following fields:
            * **id_outlet (int):**  The ID of the outlet.
            * **latitude (float):**  The latitude of the outlet.
            * **longitude (float):** The longitude of the outlet.
        * **n_clusters (int):**  The desired number of clusters.

    **Returns:**

    * **dict:**  A dictionary containing the clustering results:
        * **data (list):**  A list of data points with their assigned cluster information:
            * **id_outlet (int):**  The ID of the outlet.
            * **latitude (float):**  The latitude of the outlet.
            * **longitude (float):** The longitude of the outlet.
            * **cluster (int):**  The cluster ID assigned to the data point.
            * **color (str):**  A color representing the cluster.
        * **cluster_definitions (list):** A list of cluster definitions:
            * **cluster (int):**  The cluster ID.
            * **color (str):**  A color representing the cluster.
            * **center (list):** The coordinates of the cluster's center.
    
    **Example Response:**

    ```json
    {
        "data": [
            {
                "id_outlet": 123,
                "latitude": -6.2345,
                "longitude": 106.876,
                "cluster": 0,
                "color": "#FF0000"
            },
            # ... more data points ... 
        ],
        "cluster_definitions": [
            {
                "cluster": 0,
                "color": "#FF0000",
                "center": [ -6.21, 106.85 ]
            },
            # ... more cluster definitions ...
        ]
    }
    ```
    """

    # Input Validation
    if len(request.data) == 0:
        raise HTTPException(status_code=400, detail="No data points provided")

    if request.n_clusters <= 0:
        raise HTTPException(status_code=400, detail="Number of clusters must be greater than zero")
    
    X, total_data = prepare_clustering_data(request.data)
    size_min = total_data // request.n_clusters
    size_max = total_data

    labels, clf = perform_clustering(X, total_data, request.n_clusters, size_min, size_max)
    colors = generate_colors(request.n_clusters)

    response = build_clustering_response(labels, colors, request.data, clf)
    return response

@app.post("/cluster/outlets")
async def perform_clustering_by_outlets(request: ClusteringRequestOutlets):
    """
    Performs clustering based on the specified number of outlets.

    **Parameters:**

    * **request (ClusteringRequestOutlets):**  The input request containing:
        * **data (list[DataPoint]):**  A list of data points with the following fields:
            * **id_outlet (int):**  The ID of the outlet.
            * **latitude (float):**  The latitude of the outlet.
            * **longitude (float):** The longitude of the outlet.
        * **n_outlets (int):**  The desired number of outlet.

    **Returns:**

    * **dict:**  A dictionary containing the clustering results:
        * **data (list):**  A list of data points with their assigned cluster information:
            * **id_outlet (int):**  The ID of the outlet.
            * **latitude (float):**  The latitude of the outlet.
            * **longitude (float):** The longitude of the outlet.
            * **cluster (int):**  The cluster ID assigned to the data point.
            * **color (str):**  A color representing the cluster.
        * **cluster_definitions (list):** A list of cluster definitions:
            * **cluster (int):**  The cluster ID.
            * **color (str):**  A color representing the cluster.
            * **center (list):** The coordinates of the cluster's center.
    
    **Example Response:**

    ```json
    {
        "data": [
            {
                "id_outlet": 123,
                "latitude": -6.2345,
                "longitude": 106.876,
                "cluster": 0,
                "color": "#FF0000"
            },
            # ... more data points ... 
        ],
        "cluster_definitions": [
            {
                "cluster": 0,
                "color": "#FF0000",
                "center": [ -6.21, 106.85 ]
            },
            # ... more cluster definitions ...
        ]
    }
    ```
    """
    # Input Validation
    if len(request.data) == 0:
        raise HTTPException(status_code=400, detail="No data points provided")

    if request.n_outlets <= 0:
        raise HTTPException(status_code=400, detail="Number of outlets must be greater than zero")
    
    X, total_data = prepare_clustering_data(request.data)
    size_min = request.n_outlets
    size_max = total_data
    n_clusters = math.floor(total_data / request.n_outlets)

    labels, clf = perform_clustering(X, total_data, n_clusters, size_min, size_max)
    colors = generate_colors(n_clusters)

    response = build_clustering_response(labels, colors, request.data, clf)
    return response 

@app.post("/cluster/working_hours")
async def perform_clustering_by_working_hours(request: ClusteringRequestWorkingHours) -> ClusteringResponseWorkingHours:
    """
    Clusters outlets based on working hours using a greedy algorithm.

    **Parameters:**

    * **request (ClusteringRequestWorkingHours):** The input request containing:
        * **data (list[DataPoint]):**  A list of data points with the following fields:
            * **id_outlet (int):**  The ID of the outlet.
            * **latitude (float):**  The latitude of the outlet.
            * **longitude (float):** The longitude of the outlet.
            * **working_hours (int):** The working hours in minutes.
        * **n_outlets (int):**  The desired number of outlet.
            - n_hours: The maximum allowed total working hours in a cluster.

    **Returns:**

    * **dict:**  A dictionary containing the clustering results:
        * **data (list):**  A list of data points with their assigned cluster information:
            * **id_outlet (int):**  The ID of the outlet.
            * **latitude (float):**  The latitude of the outlet.
            * **longitude (float):** The longitude of the outlet.
            * **cluster (int):**  The cluster ID assigned to the data point.
            * **color (str):**  A color representing the cluster.
        * **cluster_definitions (list):** A list of cluster definitions:
            * **cluster (int):**  The cluster ID.
            * **color (str):**  A color representing the cluster.
            * **center (list):** The coordinates of the cluster's center.

    **Raises:**
        HTTPException: If the input data is invalid (e.g., no data points or invalid working hours).

    ```
    {
        "data": [
        {
            "id_outlet": 1,
            "latitude": -6.2345,
            "longitude": 106.876,
            "working_hours": 8
        },
        {
            "id_outlet": 2,
            "latitude": -6.1234,
            "longitude": 106.987,
            "working_hours": 10
        },
        {
            "id_outlet": 3,
            "latitude": -6.3210,
            "longitude": 106.789,
            "working_hours": 12
        },
        {
            "id_outlet": 4,
            "latitude": -6.2876,
            "longitude": 106.823,
            "working_hours": 6
        }],
        "n_hours": 15
    }
    ```

    ```
    {
    "status": "success",
    "data": [
        {
            "id_outlet": 1,
            "latitude": -6.2345,
            "longitude": 106.876,
            "cluster": 0,
            "color": "#FF5733"
        },
        {
            "id_outlet": 2,
            "latitude": -6.1234,
            "longitude": 106.987,
            "cluster": 1,
            "color": "#3498DB"
        },
        {
            "id_outlet": 3,
            "latitude": -6.3210,
            "longitude": 106.789,
            "cluster": 0,
            "color": "#FF5733"
        },
        {
            "id_outlet": 4,
            "latitude": -6.2876,
            "longitude": 106.823,
            "cluster": 1,
            "color": "#3498DB"
        }
    ],
    "cluster_definitions": [
        {
            "cluster": 0,
            "color": "#FF5733",
            "center": [-6.27775, 106.8325]
        },
        {
            "cluster": 1,
            "color": "#3498DB",
            "center": [-6.2055, 106.900]
        }
        ]
    }

    ```
    """
    outlets_df = pd.DataFrame([d.dict() for d in request.data])
    outlets_df = outlets_df.sort_values(by=['latitude', 'longitude']).reset_index(drop=True)

    # Extract necessary data
    outlets_df['assigned'] = False
    clusters = []

    # Function to calculate total visit duration for a cluster
    def calculate_total_time(cluster):
        return sum(outlets_df.loc[outlets_df['id_outlet'].isin(cluster), 'working_hours'])

    # Function to find the nearest outlet not yet assigned
    def find_nearest_outlet(current_location, unassigned_outlets):
        nearest_outlet = None
        nearest_distance = float('inf')
        for _, outlet in unassigned_outlets.iterrows():
            distance = geodesic(current_location, (outlet['latitude'], outlet['longitude'])).meters
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_outlet = outlet
        return nearest_outlet

    # Function to create a new cluster starting with a given outlet
    def create_cluster(start_outlet, max_time):
        cluster = [start_outlet['id_outlet']]
        current_time = start_outlet['working_hours']
        current_location = (start_outlet['latitude'], start_outlet['longitude'])
        unassigned_outlets = outlets_df[~outlets_df['assigned']]

        while current_time <= max_time:
            nearest_outlet = find_nearest_outlet(current_location, unassigned_outlets)
            if nearest_outlet is None:
                break
            
            if current_time + nearest_outlet['working_hours'] > max_time:
                break
            
            cluster.append(nearest_outlet['id_outlet'])
            current_time += nearest_outlet['working_hours']
            current_location = (nearest_outlet['latitude'], nearest_outlet['longitude'])
            outlets_df.loc[outlets_df['id_outlet'] == nearest_outlet['id_outlet'], 'assigned'] = True
            unassigned_outlets = outlets_df[~outlets_df['assigned']]
        
        return cluster

    # Create clusters iteratively
    max_time = request.n_hours  # Set maximum time for a cluster in minutes
    unassigned_outlets = outlets_df[~outlets_df['assigned']]

    while not unassigned_outlets.empty:
        start_outlet = unassigned_outlets.iloc[0]
        outlets_df.loc[outlets_df['id_outlet'] == start_outlet['id_outlet'], 'assigned'] = True
        cluster = create_cluster(start_outlet, max_time)
        clusters.append(cluster)
        unassigned_outlets = outlets_df[~outlets_df['assigned']]

    # Assign cluster IDs
    for cluster_id, cluster in enumerate(clusters):
        outlets_df.loc[outlets_df['id_outlet'].isin(cluster), 'cluster'] = cluster_id
    
    # Calculate cluster centroids
    def calculate_centroids(df):
        return df.groupby('cluster')[['latitude', 'longitude']].mean().to_dict('index')

    # Reassign outlets to nearest cluster if they are far from their centroid
    def reassign_outliers(df, centroids, max_distance=5000):  # Max distance in meters
        for idx, row in df.iterrows():
            current_cluster = row['cluster']
            centroid = (centroids[current_cluster]['latitude'], centroids[current_cluster]['longitude'])
            distance_to_centroid = geodesic((row['latitude'], row['longitude']), centroid).meters
            
            if distance_to_centroid > max_distance:
                nearest_cluster = None
                nearest_distance = float('inf')
                for cluster_id, centroid_coords in centroids.items():
                    distance = geodesic((row['latitude'], row['longitude']), (centroid_coords['latitude'], centroid_coords['longitude'])).meters
                    if distance < nearest_distance and calculate_total_time(df[df['cluster'] == cluster_id]['id_outlet'].tolist()) + row['working_hours'] <= max_time:
                        nearest_distance = distance
                        nearest_cluster = cluster_id
                
                if nearest_cluster is not None:
                    df.at[idx, 'cluster'] = nearest_cluster
                    centroids = calculate_centroids(df)
        
        return df

    # Calculate initial centroids
    centroids = calculate_centroids(outlets_df)

    # Reassign outliers
    adjusted_df = reassign_outliers(outlets_df.copy(), centroids)
    outlets_df = adjusted_df.copy()

    labels = adjusted_df['cluster'].values  # Update labels after outlier reassignment
    colors = generate_colors(len(centroids))

    # Create cluster definitions with color
    cluster_definitions = []
    for cluster_id in outlets_df['cluster'].unique():
        if cluster_id != -1:
            color = colors[int(cluster_id)]  # Get the color for this cluster
            centroid = centroids[cluster_id]
            cluster_definitions.append({
                'cluster': int(cluster_id),
                'color': color,
                'center': [centroid['latitude'], centroid['longitude']]
            })

    # Prepare response data
    response_data = []
    for i, row in adjusted_df.iterrows():
        response_data.append(
            ClusterData(
                id_outlet=row['id_outlet'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                cluster=row['cluster'],
                color=colors[int(row['cluster'])]
            )
        )

    return ClusteringResponseWorkingHours(
        status="success",
        data=response_data,
        cluster_definitions=cluster_definitions
    )

@app.get("/sample_output")
async def get_sample_output():
    with open("sample_10k.json", "r") as f:
        return json.load(f)

@app.get("/")
async def root():
    return {'message':"Welcome"}

