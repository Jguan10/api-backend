# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import math
import os
from supabase import create_client, Client

app = FastAPI(title="Pet Sitter Matching API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Pydantic models
class OwnerRequest(BaseModel):
    zipCode: str
    service: str
    startDate: str
    endDate: Optional[str] = None
    needs: str  # Comma-separated string

class SitterMatch(BaseModel):
    name: str
    location: str
    distance: float
    score: float
    availability: str
    special_needs: str

class MatchResponse(BaseModel):
    matches: List[SitterMatch]

# Load sitter data from Supabase
def load_sitter_data():
    """
    Load sitter data from Supabase
    """
    try:
        # Fetch all sitters from Supabase
        response = supabase.table('sitters').select('*').execute()
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        
        return df
    except Exception as e:
        print(f"Error loading sitter data from Supabase: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading sitter data: {str(e)}")

# Haversine distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in miles
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in miles
    r = 3956
    
    return c * r

# Geocoding function using Supabase or external API
def geocode_zipcode(zipcode):
    """
    Convert zipcode to lat/lon
    First tries to get from Supabase zipcodes table, then falls back to default
    """
    try:
        # Try to get from Supabase zipcodes table
        response = supabase.table('zipcodes').select('latitude', 'longitude').eq('zip_code', zipcode).execute()
        
        if response.data and len(response.data) > 0:
            return (response.data[0]['latitude'], response.data[0]['longitude'])
        
        # Fallback to default NYC location
        return (40.7589, -73.9851)
    except Exception as e:
        print(f"Error geocoding zipcode: {e}")
        return (40.7589, -73.9851)

# Calculate special needs score using asymmetric matching
def calculate_special_needs_score(owner_needs: str, sitter_needs: str):
    """
    Calculate similarity between owner needs and sitter experience using asymmetric matching
    
    Args:
        owner_needs (str): Owner's special needs (comma-separated)
        sitter_needs (str): Sitter's special needs experience (comma-separated)
    
    Returns:
        float: Bonus points to add to score (0-20)
    """
    # Handle empty cases
    if not owner_needs or not sitter_needs or owner_needs.lower() in ['none', 'n/a', ''] or sitter_needs.lower() in ['none', 'n/a', '']:
        return 0
    
    try:
        # Convert comma-separated strings to sets (strip whitespace)
        owner_set = set(s.strip().lower() for s in owner_needs.split(',') if s.strip())
        sitter_set = set(s.strip().lower() for s in sitter_needs.split(',') if s.strip())
        
        # Handle empty sets after splitting
        if not owner_set or not sitter_set:
            return 0
        
        # Calculate asymmetric similarity (coverage of owner needs)
        matches = len(owner_set & sitter_set)
        coverage = matches / len(owner_set)
        
        # Convert to bonus points (0-20)
        bonus = coverage * 20
        
        return math.ceil(bonus)
        
    except Exception as e:
        print(f"Error calculating special needs score: {e}")
        return 0

# Calculate distance score
def calculate_distance_score(distance):
    """
    Convert distance to score (0-40 points)
    Closer distance = higher score
    """
    if distance <= 5:
        return 40
    elif distance <= 10:
        return 35
    elif distance <= 15:
        return 30
    elif distance <= 20:
        return 25
    elif distance <= 30:
        return 20
    elif distance <= 50:
        return 15
    else:
        return 10

# Calculate availability score
def calculate_availability_score(requested_start, requested_end, available_start, available_end):
    """
    Check if sitter is available for the requested dates
    Returns 40 points if available, 0 if not
    """
    try:
        req_start = datetime.strptime(requested_start, '%Y-%m-%d')
        req_end = datetime.strptime(requested_end, '%Y-%m-%d') if requested_end else req_start
        avail_start = datetime.strptime(available_start, '%Y-%m-%d')
        avail_end = datetime.strptime(available_end, '%Y-%m-%d')
        
        # Check if requested dates fall within available dates
        if avail_start <= req_start and req_end <= avail_end:
            return 40
        else:
            return 0
    except Exception as e:
        print(f"Error calculating availability score: {e}")
        return 0

# Calculate service match score
def calculate_service_score(requested_service, offered_services):
    """
    Check if sitter offers the requested service
    Returns 20 points if match, 0 if not
    """
    offered_list = [s.strip().lower() for s in offered_services.split(',')]
    if requested_service.lower() in offered_list:
        return 20
    return 0

@app.get("/")
def read_root():
    return {"message": "Pet Sitter Matching API", "status": "running"}

@app.post("/api/match", response_model=MatchResponse)
def match_sitters(request: OwnerRequest):
    """
    Match and rank sitters based on owner requirements
    """
    try:
        # Load sitter data from Supabase
        sitters_df = load_sitter_data()
        
        # Get owner location
        owner_lat, owner_lon = geocode_zipcode(request.zipCode)
        
        # Calculate scores for each sitter
        results = []
        
        for _, sitter in sitters_df.iterrows():
            # Calculate distance
            distance = haversine_distance(
                owner_lat, owner_lon,
                sitter['latitude'], sitter['longitude']
            )
            
            # Calculate component scores
            distance_score = calculate_distance_score(distance)
            availability_score = calculate_availability_score(
                request.startDate,
                request.endDate,
                sitter['availability_start'],
                sitter['availability_end']
            )
            service_score = calculate_service_score(request.service, sitter['services'])
            special_needs_score = calculate_special_needs_score(request.needs, sitter['special_needs'])
            
            # Total score (max 120 points)
            total_score = distance_score + availability_score + service_score + special_needs_score
            
            results.append({
                'name': sitter['name'],
                'location': sitter['zip_code'],
                'distance': round(distance, 2),
                'score': total_score,
                'availability': f"{sitter['availability_start']} to {sitter['availability_end']}",
                'special_needs': sitter['special_needs']
            })
        
        # Sort by score (descending) and take top 5
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
        
        return MatchResponse(matches=results_sorted)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error matching sitters: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)