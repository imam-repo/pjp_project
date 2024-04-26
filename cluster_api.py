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
        return labels 

    except Exception as e:  
        logger.error("Clustering failed:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@app.post("/cluster/clusters")
async def perform_kmeans_by_clusters(request: ClusteringRequestClusters):
    # Input Validation
    if len(request.data) == 0:
        raise HTTPException(status_code=400, detail="No data points provided")

    if request.n_clusters <= 0:
        raise HTTPException(status_code=400, detail="Number of clusters must be greater than zero")
    
    X, total_data = prepare_clustering_data(request.data)
    size_min = total_data // request.n_clusters
    size_max = total_data

    labels = perform_clustering(X, total_data, request.n_clusters, size_min, size_max)
    colors = generate_colors(request.n_clusters)

    response = []
    for i, label in enumerate(labels):
        
        color = colors[label.item()]
        response.append(ClusterData(
            id_outlet=request.data[i].id_outlet,
            latitude=request.data[i].latitude,
            longitude=request.data[i].longitude,
            cluster=label.item() ,
            color=color
        ))

    return {
        "status": "success",
        "data": response 
    } 

@app.post("/cluster/outlets")
async def perform_kmeans_by_outlets(request: ClusteringRequestOutlets):
    # Input Validation
    if len(request.data) == 0:
        raise HTTPException(status_code=400, detail="No data points provided")

    if request.n_outlets <= 0:
        raise HTTPException(status_code=400, detail="Number of outlets must be greater than zero")
    
    X, total_data = prepare_clustering_data(request.data)
    size_min = request.n_outlets
    size_max = total_data
    n_clusters = math.ceil(total_data / request.n_outlets)

    labels = perform_clustering(X, total_data, n_clusters, size_min, size_max)
    colors = generate_colors(n_clusters)

    response = []
    for i, label in enumerate(labels):
        
        color = colors[label.item()]
        response.append(ClusterData(
            id_outlet=request.data[i].id_outlet,
            latitude=request.data[i].latitude,
            longitude=request.data[i].longitude,
            cluster=label.item() ,
            color=color
        ))

    return {
        "status": "success",
        "data": response 
    } 

@app.get("/sample_output")
async def get_sample_output():
    with open("sample_10k.json", "r") as f:
        return json.load(f)

@app.get("/")
async def root():
    return {'message':"Welcome"}

