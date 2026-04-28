"""
vehicle.py
Data model for vehicles in the Smart City Traffic AI System.
Provides vehicle classification and authorization checking.
"""


class Vehicle:
    """
    Represents a vehicle in the traffic system.
    
    This class encapsulates vehicle properties and provides methods
    to check vehicle type classifications and authorization levels.
    
    Attributes:
        vehicle_id (str): Unique identifier for the vehicle
        vehicle_type (str): Type of vehicle (Civilian, Ambulance, Fire_Truck, Police)
        current_location (str): Current position in city graph
        destination (str): Target destination
    """
    
    def __init__(self, vehicle_id, vehicle_type, current_location, destination=None):
        """
        Initialize a Vehicle instance.
        
        Args:
            vehicle_id (str): Unique vehicle identifier
            vehicle_type (str): Type of vehicle
            current_location (str): Current location in city
            destination (str, optional): Target destination. Defaults to None.
        """
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.current_location = current_location
        self.destination = destination
    
    @property
    def is_emergency(self):
        """
        Check if this vehicle is an emergency vehicle.
        
        Returns:
            bool: True if emergency vehicle, False otherwise
        """
        from src.config import VehicleType
        return VehicleType.is_emergency(self.vehicle_type)
    
    @property
    def is_civilian(self):
        """
        Check if this vehicle is a civilian vehicle.
        
        Returns:
            bool: True if civilian vehicle, False otherwise
        """
        from src.config import VehicleType
        return self.vehicle_type == VehicleType.CIVILIAN
    
    def can_request_signal_override(self, zone):
        """
        Check if vehicle is authorized for signal override in a zone.
        
        According to traffic policy:
        - Emergency vehicles: Authorized for signal override
        - Civilian vehicles: NOT authorized for signal override
        
        Args:
            zone (str): Signal zone identifier
            
        Returns:
            bool: True if authorized, False otherwise
        """
        return self.is_emergency
    
    def can_request_emergency_route(self):
        """
        Check if vehicle is authorized for emergency routing.
        
        Returns:
            bool: True if authorized, False otherwise
        """
        return self.is_emergency
    
    def __str__(self):
        """
        String representation of the vehicle.
        
        Returns:
            str: Formatted vehicle description
        """
        emergency_tag = "[EMERGENCY]" if self.is_emergency else "[CIVILIAN]"
        return f"{emergency_tag} {self.vehicle_type}({self.vehicle_id}) at {self.current_location}"
    
    def __repr__(self):
        """
        Detailed representation for debugging.
        
        Returns:
            str: Detailed vehicle information
        """
        return (
            f"Vehicle(vehicle_id='{self.vehicle_id}', "
            f"vehicle_type='{self.vehicle_type}', "
            f"current_location='{self.current_location}', "
            f"destination='{self.destination}')"
        )