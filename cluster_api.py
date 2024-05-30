from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from k_means_constrained import KMeansConstrained
from models import DataPoint, ClusteringRequestClusters, ClusteringRequestOutlets, ClusterData
from logging_config import logger
from middleware import LoggingMiddleware
import numpy as np
import json 
import random
import math


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
    n_clusters = math.ceil(total_data / request.n_outlets)

    labels, clf = perform_clustering(X, total_data, n_clusters, size_min, size_max)
    colors = generate_colors(n_clusters)

    response = build_clustering_response(labels, colors, request.data, clf)
    return response 

@app.get("/sample_output")
async def get_sample_output():
    with open("sample_10k.json", "r") as f:
        return json.load(f)

@app.get("/")
async def root():
    return {'message':"Welcome"}

