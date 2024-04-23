from fastapi import FastAPI, HTTPException, Request
from k_means_constrained import KMeansConstrained
from models import DataPoint, ClusteringRequest, ClusterData
from logging_config import logger
from middleware import LoggingMiddleware
import numpy as np
import json 
import random


app = FastAPI()
app.add_middleware(LoggingMiddleware)

def generate_colors(n_clusters):
    color_list = []
    for _ in range(n_clusters):
        color_list.append('#%06X' % random.randint(0, 0xFFFFFF))
    return color_list

@app.post("/cluster")
async def perform_kmeans(request: ClusteringRequest):
    # Input Validation
    if len(request.data) == 0:
        raise HTTPException(status_code=400, detail="No data points provided")

    if request.n_clusters <= 0:
        raise HTTPException(status_code=400, detail="Number of clusters must be greater than zero")
    
    X = np.array([(p.latitude, p.longitude) for p in request.data])

    total_data = len(X)
    size_min = total_data // request.n_clusters
    size_max = total_data

    try:
        clf = KMeansConstrained(
            n_clusters=request.n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=0,
            n_jobs=-1
        )
        labels = clf.fit_predict(X)

    except Exception as e:  
        logger.error("Clustering failed:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")
    
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

@app.get("/sample_output")
def get_sample_output():
    with open("sample_10k.json", "r") as f:
        return json.load(f)

@app.get("/")
def root():
    return {'message':"Welcome"}

