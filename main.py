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
import traceback
import requests

app = FastAPI(title="Pet Sitter Matching API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(f"Supabase URL: {SUPABASE_URL}")
print(f"Supabase Key exists: {bool(SUPABASE_KEY)}")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Pydantic models
class OwnerRequest(BaseModel):
    zipCode: str
    service: str
    startDate: str
    endDate: Optional[str] = None
    needs: str

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
    Expected columns: name, zip_code, latitude, longitude, services, special_needs, availability_start, availability_end
    """
    try:
        print("Loading sitter data from Supabase...")
        response = supabase.table('sitters').select('*').execute()
        
        print(f"Supabase response: {response.data}")
        
        if not response.data or len(response.data) == 0:
            print("WARNING: No sitters found in database")
            return pd.DataFrame(columns=['name', 'zip_code', 'latitude', 'longitude', 
                                        'services', 'special_needs', 'availability_start', 'availability_end'])
        
        df = pd.DataFrame(response.data)
        print(f"Loaded {len(df)} sitters")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"Error loading sitter data from Supabase: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading sitter data: {str(e)}")

# Haversine distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in miles"""
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 3956
        return c * r
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return 999999

# Geocoding using free zippopotam.us API
def geocode_zipcode(zipcode):
    """
    Convert zipcode to lat/lon using zippopotam.us API
    Free, no API key needed
    """
    try:
        # Try zippopotam.us API
        url = f"http://api.zippopotam.us/us/{zipcode}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            lat = float(data['places'][0]['latitude'])
            lon = float(data['places'][0]['longitude'])
            print(f"Geocoded {zipcode} to ({lat}, {lon})")
            return (lat, lon)
        else:
            print(f"Zipcode {zipcode} not found in zippopotam.us, using default NYC")
            return (40.7589, -73.9851)
            
    except Exception as e:
        print(f"Error geocoding zipcode {zipcode}: {e}")
        # Fallback to default NYC location
        return (40.7589, -73.9851)

# Calculate special needs score
def calculate_special_needs_score(owner_needs: str, sitter_needs: str):
    """Calculate asymmetric similarity (0-20 points)"""
    if not owner_needs or not sitter_needs or owner_needs.lower() in ['none', 'n/a', ''] or sitter_needs.lower() in ['none', 'n/a', '']:
        return 0
    
    try:
        owner_set = set(s.strip().lower() for s in str(owner_needs).split(',') if s.strip())
        sitter_set = set(s.strip().lower() for s in str(sitter_needs).split(',') if s.strip())
        
        if not owner_set or not sitter_set:
            return 0
        
        matches = len(owner_set & sitter_set)
        coverage = matches / len(owner_set)
        bonus = coverage * 20
        
        return math.ceil(bonus)
    except Exception as e:
        print(f"Error calculating special needs score: {e}")
        return 0

# Calculate distance score
def calculate_distance_score(distance):
    """0-40 points based on distance"""
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
    """40 points if available, 0 if not"""
    try:
        req_start = datetime.strptime(str(requested_start), '%Y-%m-%d')
        req_end = datetime.strptime(str(requested_end), '%Y-%m-%d') if requested_end else req_start
        avail_start = datetime.strptime(str(available_start), '%Y-%m-%d')
        avail_end = datetime.strptime(str(available_end), '%Y-%m-%d')
        
        if avail_start <= req_start and req_end <= avail_end:
            return 40
        else:
            return 0
    except Exception as e:
        print(f"Error calculating availability score: {e}")
        return 0

# Calculate service match score
def calculate_service_score(requested_service, offered_services):
    """20 points if service matches"""
    try:
        offered_list = [s.strip().lower() for s in str(offered_services).split(',')]
        if str(requested_service).lower() in offered_list:
            return 20
        return 0
    except Exception as e:
        print(f"Error calculating service score: {e}")
        return 0

@app.get("/")
def read_root():
    return {"message": "Pet Sitter Matching API", "status": "running"}

@app.post("/api/match", response_model=MatchResponse)
def match_sitters(request: OwnerRequest):
    """Match and rank sitters"""
    try:
        print(f"\n=== Matching Request ===")
        print(f"Zip: {request.zipCode}")
        print(f"Service: {request.service}")
        print(f"Start: {request.startDate}")
        print(f"End: {request.endDate}")
        print(f"Needs: {request.needs}")
        
        # Load sitter data from Supabase
        sitters_df = load_sitter_data()
        
        if len(sitters_df) == 0:
            print("No sitters in database")
            return MatchResponse(matches=[])
        
        # Get owner location
        owner_lat, owner_lon = geocode_zipcode(request.zipCode)
        print(f"Owner location: {owner_lat}, {owner_lon}")
        
        # Calculate scores for each sitter
        results = []
        
        for idx, sitter in sitters_df.iterrows():
            try:
                # Calculate distance
                distance = haversine_distance(
                    owner_lat, owner_lon,
                    sitter['latitude'], sitter['longitude']
                )
                
                # Calculate component scores
                distance_score = calculate_distance_score(distance)
                availability_score = calculate_availability_score(
                    request.startDate,
                    request.endDate or request.startDate,
                    sitter['availability_start'],
                    sitter['availability_end']
                )
                service_score = calculate_service_score(request.service, sitter['services'])
                special_needs_score = calculate_special_needs_score(request.needs, sitter['special_needs'])
                
                # Total score (max 120 points)
                total_score = distance_score + availability_score + service_score + special_needs_score
                
                print(f"Sitter: {sitter['name']}, Score: {total_score} (dist:{distance_score}, avail:{availability_score}, svc:{service_score}, needs:{special_needs_score})")
                
                results.append({
                    'name': str(sitter['name']),
                    'location': str(sitter['zip_code']),
                    'distance': round(float(distance), 2),
                    'score': int(total_score),
                    'availability': f"{sitter['availability_start']} to {sitter['availability_end']}",
                    'special_needs': str(sitter.get('special_needs', 'None'))
                })
            except Exception as e:
                print(f"Error processing sitter {idx}: {e}")
                print(traceback.format_exc())
                continue
        
        # Sort by score and take top 5
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
        
        print(f"Returning {len(results_sorted)} matches")
        return MatchResponse(matches=results_sorted)
        
    except Exception as e:
        print(f"ERROR in match_sitters: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error matching sitters: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/debug/sitters")
def debug_sitters():
    """Debug endpoint to see what's in the database"""
    try:
        response = supabase.table('sitters').select('*').execute()
        return {"count": len(response.data), "data": response.data}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)