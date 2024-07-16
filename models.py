from fastapi import HTTPException 
from pydantic import BaseModel, PositiveInt, validator


# Data model for JSON input
class DataPoint(BaseModel):
    """
    Represents a single data point with outlet information, including working hours.
    """
    id_outlet: int
    latitude: float
    longitude: float
    working_hours: float

    # Add validators for latitude and longitude
    @validator('latitude')
    def latitude_within_range(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @validator('longitude')
    def longitude_within_range(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

class DataPointWithoutWorkingHours(BaseModel):
    """
    Represents a data point without working hours information.
    """
    id_outlet: int
    latitude: float
    longitude: float

    # Add validators for latitude and longitude
    @validator('latitude')
    def latitude_within_range(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @validator('longitude')
    def longitude_within_range(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

class ClusteringRequestClusters(BaseModel):
    """
    Represents a clustering request based on the number of clusters (excluding working hours).
    """
    data: list[DataPointWithoutWorkingHours]
    n_clusters: PositiveInt

class ClusteringRequestOutlets(BaseModel):
    """
    Represents a clustering request based on the number of outlets per cluster.
    """
    data: list[DataPointWithoutWorkingHours]
    n_outlets: PositiveInt
    
class ClusteringRequestWorkingHours(BaseModel):
    """
    Represents a clustering request based on working hours.
    """ 
    data: list[DataPoint] 
    n_hours: float

class ClusterData(BaseModel):  
    id_outlet: int
    latitude: float
    longitude: float
    cluster: int
    color: str   

class ClusteringResponse(BaseModel): 
    status: str
    data: list[ClusterData]  

class ClusteringResponseWorkingHours(BaseModel):
    """
    Represents a clustering response specifically for working hours clustering.

    Attributes:
        status (str): The status of the clustering process ("success" or "error").
        data (list[ClusterData]): List of clustered outlet data (includes working hours).
        total_clusters (int): The total number of clusters created.
    """
    status: str
    data: list[ClusterData]
    cluster_definitions: list[dict] = []