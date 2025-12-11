# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import math

app = FastAPI(title="Pet Sitter Matching API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Load sitter data (you'll need to provide the actual CSV file)
# For now, this is a placeholder - replace with your actual data loading
def load_sitter_data():
    # TODO: Load your actual sitter data from CSV
    # df = pd.read_csv('sitters.csv')
    # For demonstration, creating sample data
    sample_data = {
        'name': ['Alice Johnson', 'Bob Smith', 'Carol Williams', 'David Brown', 'Emma Davis'],
        'zip_code': ['10001', '10002', '10003', '10004', '10005'],
        'latitude': [40.7589, 40.7614, 40.7489, 40.7569, 40.7639],
        'longitude': [-73.9851, -73.9776, -73.9680, -73.9862, -73.9723],
        'services': ['catBoarding,inHomeVisits', 'catBoarding', 'inHomeVisits', 'catBoarding,inHomeVisits', 'catBoarding'],
        'special_needs': ['Medication (Daily), Anxiety, Senior Cats', 'Kittens, Medication (Occasional)', 
                         'Anxiety, Blindness, Mobility Issues', 'Diabetes, Senior Cats, Medication (Daily)',
                         'Kittens, Deafness'],
        'availability_start': ['2025-01-01', '2025-01-15', '2025-01-01', '2025-01-20', '2025-01-01'],
        'availability_end': ['2025-12-31', '2025-06-30', '2025-12-31', '2025-12-31', '2025-12-31']
    }
    return pd.DataFrame(sample_data)

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

# Geocoding function (placeholder - you'll need actual geocoding)
def geocode_zipcode(zipcode):
    """
    Convert zipcode to lat/lon
    TODO: Implement actual geocoding using an API
    """
    # Placeholder mapping - replace with actual geocoding API
    zipcode_coords = {
        '10001': (40.7506, -73.9971),
        '10002': (40.7156, -73.9862),
        '10003': (40.7318, -73.9873),
        '90210': (34.0901, -118.4065),
    }
    return zipcode_coords.get(zipcode, (40.7589, -73.9851))  # Default to NYC

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
        # Load sitter data
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