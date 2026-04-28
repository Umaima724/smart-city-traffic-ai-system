"""
config.py
System-wide configuration and constants for the Smart City Traffic AI System.
"""

# Request Categories
class RequestCategory:
    ROUTE_REQUEST = "Route_Request"
    POLICY_CHECK = "Policy_Check"
    CONTROL_ALLOCATION_REQUEST = "Control_Allocation_Request"
    EMERGENCY_RESPONSE_REQUEST = "Emergency_Response_Request"
    INTEGRATED_CITY_SERVICE_REQUEST = "Integrated_City_Service_Request"

    VALID_CATEGORIES = [
        ROUTE_REQUEST,
        POLICY_CHECK,
        CONTROL_ALLOCATION_REQUEST,
        EMERGENCY_RESPONSE_REQUEST,
        INTEGRATED_CITY_SERVICE_REQUEST
    ]


# Vehicle Types
class VehicleType:
    CIVILIAN = "Civilian"
    AMBULANCE = "Ambulance"
    FIRE_TRUCK = "Fire_Truck"
    POLICE = "Police"

    VALID_TYPES = [CIVILIAN, AMBULANCE, FIRE_TRUCK, POLICE]

    @classmethod
    def is_emergency(cls, vehicle_type):
        """Check if vehicle type is emergency vehicle."""
        return vehicle_type in [cls.AMBULANCE, cls.FIRE_TRUCK, cls.POLICE]


# Severity Levels
class SeverityLevel:
    NONE = "None"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    VALID_LEVELS = [NONE, LOW, MEDIUM, HIGH]

    # Numeric mapping for ANN
    NUMERIC_MAP = {
        NONE: 0,
        LOW: 1,
        MEDIUM: 2,
        HIGH: 3
    }


# Priority Levels
class PriorityLevel:
    LOW = "Low"
    NORMAL = "Normal"
    HIGH = "High"
    CRITICAL = "Critical"

    VALID_LEVELS = [LOW, NORMAL, HIGH, CRITICAL]

    # Numeric mapping for ANN
    NUMERIC_MAP = {
        LOW: 0,
        NORMAL: 1,
        HIGH: 2,
        CRITICAL: 3
    }


# Time Sensitivity
class TimeSensitivity:
    NORMAL = "Normal"
    HIGH = "High"

    VALID_LEVELS = [NORMAL, HIGH]

    # Numeric mapping for ANN
    NUMERIC_MAP = {
        NORMAL: 0,
        HIGH: 1
    }


# Traffic Density (0-10 scale)
TRAFFIC_DENSITY_MIN = 0
TRAFFIC_DENSITY_MAX = 10


# Control Zones
class ControlZone:
    CENTRAL_JUNCTION = "Central_Junction"
    NORTH_STATION = "North_Station"
    RIVER_BRIDGE = "River_Bridge"
    EAST_MARKET = "East_Market"
    CITY_HOSPITAL = "City_Hospital"
    SOUTH_RESIDENTIAL = "South_Residential"
    WEST_TERMINAL = "West_Terminal"
    TRAFFIC_CONTROL_CENTER = "Traffic_Control_Center"
    POLICE_HQ = "Police_HQ"
    FIRE_STATION = "Fire_Station"
    STADIUM = "Stadium"
    AIRPORT_ROAD = "Airport_Road"
    INDUSTRIAL_ZONE = "Industrial_Zone"

    VALID_ZONES = [
        CENTRAL_JUNCTION, NORTH_STATION, RIVER_BRIDGE, EAST_MARKET,
        CITY_HOSPITAL, SOUTH_RESIDENTIAL, WEST_TERMINAL,
        TRAFFIC_CONTROL_CENTER, POLICE_HQ, FIRE_STATION,
        STADIUM, AIRPORT_ROAD, INDUSTRIAL_ZONE
    ]


# Signal Zones for CSP
class SignalZone:
    S1 = "S1_Central_Junction"
    S2 = "S2_North_Station"
    S3 = "S3_East_Market"
    S4 = "S4_River_Bridge"
    S5 = "S5_City_Hospital"

    VALID_ZONES = [S1, S2, S3, S4, S5]


# File Paths
DATA_DIR = "data"
CITY_GRAPH_UNWEIGHTED_PATH = f"{DATA_DIR}/city_graph_unweighted.json"
CITY_GRAPH_WEIGHTED_PATH = f"{DATA_DIR}/city_graph_weighted.json"
TRAFFIC_RULES_PATH = f"{DATA_DIR}/traffic_rules.json"
CSP_CONSTRAINTS_PATH = f"{DATA_DIR}/csp_constraints.json"
ANN_TRAINING_DATA_PATH = f"{DATA_DIR}/ann_training_data.json"


# ANN Configuration
ANN_INPUT_FEATURES = 6  # VehicleType, Severity, TimeSensitivity, TrafficDensity, Distance, PriorityClaim
ANN_BINARY_OUTPUT = 1   # Urgent (1) or Not Urgent (0)
ANN_MULTICLASS_OUTPUT = 4  # Low, Normal, High, Critical