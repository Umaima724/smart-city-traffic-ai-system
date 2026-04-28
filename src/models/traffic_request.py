"""
traffic_request.py
Data model for traffic requests in the Smart City Traffic AI System.
Defines the structure and properties of a traffic request.
"""


class TrafficRequest:
    """
    Represents a structured traffic request submitted to the system.
    
    This class encapsulates all input fields required for processing
    traffic routing, emergency response, and control allocation requests.
    
    Attributes:
        request_id (str): Unique identifier for the request
        vehicle_type (str): Type of vehicle (Civilian, Ambulance, Fire_Truck, Police)
        request_category (str): Category of request (Route_Request, Emergency_Response_Request, etc.)
        current_location (str): Starting location/node in city graph
        destination (str): Target location/node in city graph
        incident_severity (str): Severity level of incident (None, Low, Medium, High)
        time_sensitivity (str): Time sensitivity (Normal, High)
        traffic_density (int): Current traffic density on scale 0-10
        priority_claim (int): Claimed priority level 0-3 (Low=0, Normal=1, High=2, Critical=3)
        control_zone (str): Control zone identifier for signal operations
        description_note (str): Optional descriptive text about the request
    """
    
    # Required fields that must be present in every request
    REQUIRED_FIELDS = [
        'request_id',
        'vehicle_type',
        'request_category',
        'current_location',
        'destination'
    ]
    
    # Optional fields with default values
    OPTIONAL_FIELDS = {
        'incident_severity': 'None',
        'time_sensitivity': 'Normal',
        'traffic_density': 0,
        'priority_claim': 0,
        'control_zone': None,
        'description_note': ''
    }
    
    def __init__(self, **kwargs):
        """
        Initialize a TrafficRequest with provided fields.
        
        Args:
            **kwargs: Keyword arguments for request fields
            
        Raises:
            MissingFieldError: If a required field is not provided
            InvalidValueError: If a field value is outside acceptable range
        """
        # Set required fields
        for field in self.REQUIRED_FIELDS:
            if field not in kwargs:
                from src.utils.exceptions import MissingFieldError
                raise MissingFieldError(field)
            setattr(self, field, kwargs[field])
        
        # Set optional fields with defaults
        for field, default_value in self.OPTIONAL_FIELDS.items():
            setattr(self, field, kwargs.get(field, default_value))
        
        # Internal derived fields (set during processing)
        self._estimated_distance = None
        self._normalized_features = None
        self._validation_status = None
    
    @property
    def estimated_distance(self):
        """
        Get the estimated distance for this request.
        
        Returns:
            float or None: Estimated distance if calculated, None otherwise
        """
        return self._estimated_distance
    
    @estimated_distance.setter
    def estimated_distance(self, value):
        """
        Set the estimated distance for this request.
        
        Args:
            value (float): Estimated distance value
        """
        self._estimated_distance = value
    
    @property
    def is_emergency(self):
        """
        Check if this request is from an emergency vehicle.
        
        Returns:
            bool: True if emergency vehicle, False otherwise
        """
        from src.config import VehicleType
        return VehicleType.is_emergency(self.vehicle_type)
    
    @property
    def normalized_features(self):
        """
        Get the normalized feature vector for ANN processing.
        
        Returns:
            list or None: Normalized feature vector if prepared, None otherwise
        """
        return self._normalized_features
    
    @normalized_features.setter
    def normalized_features(self, features):
        """
        Set the normalized feature vector.
        
        Args:
            features (list): Normalized feature values
        """
        self._normalized_features = features
    
    def to_dict(self):
        """
        Convert the request to a dictionary representation.
        
        Returns:
            dict: Dictionary containing all request fields
        """
        return {
            'request_id': self.request_id,
            'vehicle_type': self.vehicle_type,
            'request_category': self.request_category,
            'current_location': self.current_location,
            'destination': self.destination,
            'incident_severity': self.incident_severity,
            'time_sensitivity': self.time_sensitivity,
            'traffic_density': self.traffic_density,
            'priority_claim': self.priority_claim,
            'control_zone': self.control_zone,
            'description_note': self.description_note,
            'estimated_distance': self.estimated_distance,
            'normalized_features': self.normalized_features
        }
    
    def __str__(self):
        """
        String representation of the traffic request.
        
        Returns:
            str: Formatted request description
        """
        return (
            f"TrafficRequest[{self.request_id}] "
            f"({self.vehicle_type} | {self.request_category} | "
            f"{self.current_location} -> {self.destination})"
        )
    
    def __repr__(self):
        """
        Detailed representation for debugging.
        
        Returns:
            str: Detailed request information
        """
        return (
            f"TrafficRequest(request_id='{self.request_id}', "
            f"vehicle_type='{self.vehicle_type}', "
            f"request_category='{self.request_category}', "
            f"current_location='{self.current_location}', "
            f"destination='{self.destination}', "
            f"incident_severity='{self.incident_severity}', "
            f"time_sensitivity='{self.time_sensitivity}', "
            f"traffic_density={self.traffic_density}, "
            f"priority_claim={self.priority_claim})"
        )