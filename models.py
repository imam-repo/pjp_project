from fastapi import HTTPException 
from pydantic import BaseModel, PositiveInt, validator


# Data model for JSON input
class DataPoint(BaseModel):
    id_outlet: int  # Include the ID field
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

class ClusteringRequest(BaseModel):
    data: list[DataPoint]
    n_clusters: PositiveInt

class ClusterData(BaseModel):  
    id_outlet: int
    latitude: float
    longitude: float
    cluster: int
    color: str   

class ClusteringResponse(BaseModel): 
    status: str
    data: list[ClusterData]  