"""
FastAPI backend for Smart City Traffic AI System.
Deployed as Vercel serverless functions.
"""
from enum import Enum
import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.modules.input_preprocessing import InputPreprocessingModule
from src.modules.request_router import RequestRouter
from src.modules.ann_priority import ANNPriorityModule
from src.modules.logic_knowledge_base import LogicKnowledgeBase
from src.modules.csp_scheduler import CSPScheduler
from src.modules.search_navigation import SearchNavigationModule
from src.modules.final_response import FinalResponseModule

app = FastAPI(
    title="Smart City Traffic AI API",
    description="AI-powered traffic management system",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system (lazy loading)
_system = None

def get_system():
    global _system
    if _system is None:
        _system = SmartCityTrafficSystem()
        _system.initialize(verbose=False)
    return _system
class TimeSensitivity(str, Enum):
    Normal = "Normal"
    High = "High"

class VehicleType(str, Enum):
    Civilian = "Civilian"
    Ambulance = "Ambulance"
    Fire_Truck = "Fire_Truck"
    Police = "Police"

class SmartCityTrafficSystem:
    def __init__(self):
        self.input_module = InputPreprocessingModule()
        self.router = RequestRouter()
        self.ann_module = ANNPriorityModule()
        self.kb = LogicKnowledgeBase()
        self.csp = CSPScheduler()
        self.search = SearchNavigationModule()
        self.response_module = FinalResponseModule()
    
    def initialize(self, verbose=False):
        self.ann_module.load_training_data()
        self.ann_module.train_models(binary_epochs=100, mlp_epochs=200, verbose=False)
        return True
    
    def process(self, raw_input: dict):
        # Step 1: Input preprocessing
        traffic_request = self.input_module.process_request(raw_input)
        
        # Step 2: Routing
        routing_decision = self.router.route_request(traffic_request)
        modules_sequence = routing_decision['modules_sequence']
        
        # Step 3: Execute modules
        module_outputs = {}
        
        for module_name in modules_sequence:
            if module_name == 'ANN_Priority':
                priority = self.ann_module.predict_priority(
                    traffic_request.normalized_features, 'multiclass'
                )
                module_outputs['ann_priority'] = priority
            
            elif module_name == 'Logic_KnowledgeBase':
                predicted = module_outputs.get('ann_priority', {}).get('priority_level')
                policy = self.kb.validate_policy(traffic_request, predicted_priority=predicted)
                module_outputs['policy_validation'] = policy
            
            elif module_name == 'CSP_Scheduler':
                policy = module_outputs.get('policy_validation')
                control = self.csp.allocate_control(traffic_request, policy_result=policy)
                module_outputs['control_allocation'] = control
            
            elif module_name == 'Search_Navigation':
                control = module_outputs.get('control_allocation')
                route = self.search.find_route(
                    traffic_request,
                    control_plan=control if control and control.get('plan_type') != 'rejected' else None
                )
                module_outputs['route'] = route
        
        # Step 4: Final response
        final = self.response_module.generate_response(
            traffic_request, routing_decision, module_outputs
        )
        
        return final.to_dict()
    
class TimeSensitivity(str, Enum):
    Normal = "Normal"
    High = "High"

# Request models
class TrafficRequest(BaseModel):
    request_id: str = Field(..., example="REQ-001")
    vehicle_type: str = Field(..., example="Ambulance")
    request_category: str = Field(..., example="Emergency_Response_Request")
    current_location: str = Field(..., example="Central_Junction")
    destination: str = Field(..., example="City_Hospital")
    incident_severity: Optional[str] = "None"
    time_sensitivity: Optional[str] = "Normal"
    traffic_density: Optional[int] = 0
    priority_claim: Optional[int] = 0
    control_zone: Optional[str] = None
    description_note: Optional[str] = ""

@app.get("/")
async def root():
    return {
        "message": "Smart City Traffic AI System API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/api/categories")
async def get_categories():
    return {
        "categories": [
            {"id": "Route_Request", "name": "Route Request", "description": "Standard route guidance"},
            {"id": "Policy_Check", "name": "Policy Check", "description": "Validate authorization"},
            {"id": "Control_Allocation_Request", "name": "Control Allocation", "description": "Signal control assignment"},
            {"id": "Emergency_Response_Request", "name": "Emergency Response", "description": "Priority emergency routing"},
            {"id": "Integrated_City_Service_Request", "name": "Integrated Service", "description": "Full coordination"}
        ]
    }

@app.get("/api/locations")
async def get_locations():
    from src.config import ControlZone
    return {"locations": ControlZone.VALID_ZONES}

@app.post("/api/process")
async def process_request(request: TrafficRequest):
    try:
        system = get_system()
        result = system.process(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/demo")
async def run_demo():
    demos = [
        {
            "name": "Civilian Route",
            "request": {
                "request_id": "DEMO-001",
                "vehicle_type": "Civilian",
                "request_category": "Route_Request",
                "current_location": "North_Station",
                "destination": "Airport_Road",
                "traffic_density": 4
            }
        },
        {
            "name": "Emergency Response",
            "request": {
                "request_id": "DEMO-002",
                "vehicle_type": "Ambulance",
                "request_category": "Emergency_Response_Request",
                "current_location": "Central_Junction",
                "destination": "City_Hospital",
                "incident_severity": "High",
                "time_sensitivity": "High",
                "traffic_density": 9,
                "priority_claim": 3,
                "control_zone": "S1_Central_Junction"
            }
        }
    ]
    
    system = get_system()
    results = []
    for demo in demos:
        result = system.process(demo["request"])
        results.append({"name": demo["name"], "result": result})
    
    return {"demos": results}