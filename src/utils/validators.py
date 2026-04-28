"""
validators.py
Validation helper functions for the Smart City Traffic AI System.
Provides centralized validation logic for all input fields.
"""


def validate_request_id(request_id):
    """
    Validate that request_id is a non-empty string.
    
    Args:
        request_id: Value to validate
        
    Returns:
        str: Validated request_id
        
    Raises:
        InvalidValueError: If request_id is invalid
    """
    from src.utils.exceptions import InvalidValueError
    
    if not isinstance(request_id, str) or not request_id.strip():
        raise InvalidValueError(
            'request_id',
            request_id,
            "a non-empty string"
        )
    return request_id.strip()


def validate_vehicle_type(vehicle_type):
    """
    Validate that vehicle_type is a recognized vehicle type.
    
    Args:
        vehicle_type: Value to validate
        
    Returns:
        str: Validated vehicle_type
        
    Raises:
        InvalidValueError: If vehicle_type is not recognized
    """
    from src.config import VehicleType
    from src.utils.exceptions import InvalidValueError
    
    if vehicle_type not in VehicleType.VALID_TYPES:
        raise InvalidValueError(
            'vehicle_type',
            vehicle_type,
            f"one of {VehicleType.VALID_TYPES}"
        )
    return vehicle_type


def validate_request_category(request_category):
    """
    Validate that request_category is a recognized category.
    
    Args:
        request_category: Value to validate
        
    Returns:
        str: Validated request_category
        
    Raises:
        InvalidValueError: If request_category is not recognized
    """
    from src.config import RequestCategory
    from src.utils.exceptions import InvalidValueError
    
    if request_category not in RequestCategory.VALID_CATEGORIES:
        raise InvalidValueError(
            'request_category',
            request_category,
            f"one of {RequestCategory.VALID_CATEGORIES}"
        )
    return request_category


def validate_location(location, field_name='location'):
    """
    Validate that a location is a non-empty string and exists in city.
    
    Args:
        location: Value to validate
        field_name (str): Name of the field for error messages
        
    Returns:
        str: Validated location
        
    Raises:
        InvalidValueError: If location is invalid
    """
    from src.config import ControlZone
    from src.utils.exceptions import InvalidValueError
    
    if not isinstance(location, str) or not location.strip():
        raise InvalidValueError(
            field_name,
            location,
            "a non-empty string"
        )
    
    # Check if location exists in valid zones
    if location not in ControlZone.VALID_ZONES:
        raise InvalidValueError(
            field_name,
            location,
            f"one of {ControlZone.VALID_ZONES}"
        )
    
    return location


def validate_severity(severity):
    """
    Validate incident severity level.
    
    Args:
        severity: Value to validate
        
    Returns:
        str: Validated severity
        
    Raises:
        InvalidValueError: If severity is not recognized
    """
    from src.config import SeverityLevel
    from src.utils.exceptions import InvalidValueError
    
    if severity not in SeverityLevel.VALID_LEVELS:
        raise InvalidValueError(
            'incident_severity',
            severity,
            f"one of {SeverityLevel.VALID_LEVELS}"
        )
    return severity


def validate_time_sensitivity(time_sensitivity):
    """
    Validate time sensitivity value.
    
    Args:
        time_sensitivity: Value to validate
        
    Returns:
        str: Validated time_sensitivity
        
    Raises:
        InvalidValueError: If time_sensitivity is not recognized
    """
    from src.config import TimeSensitivity
    from src.utils.exceptions import InvalidValueError
    
    if time_sensitivity not in TimeSensitivity.VALID_LEVELS:
        raise InvalidValueError(
            'time_sensitivity',
            time_sensitivity,
            f"one of {TimeSensitivity.VALID_LEVELS}"
        )
    return time_sensitivity


def validate_traffic_density(density):
    """
    Validate traffic density is an integer within 0-10 range.
    
    Args:
        density: Value to validate
        
    Returns:
        int: Validated density
        
    Raises:
        InvalidValueError: If density is out of range or wrong type
    """
    from src.config import TRAFFIC_DENSITY_MIN, TRAFFIC_DENSITY_MAX
    from src.utils.exceptions import InvalidValueError
    
    try:
        density = int(density)
    except (ValueError, TypeError):
        raise InvalidValueError(
            'traffic_density',
            density,
            f"an integer between {TRAFFIC_DENSITY_MIN} and {TRAFFIC_DENSITY_MAX}"
        )
    
    if density < TRAFFIC_DENSITY_MIN or density > TRAFFIC_DENSITY_MAX:
        raise InvalidValueError(
            'traffic_density',
            density,
            f"an integer between {TRAFFIC_DENSITY_MIN} and {TRAFFIC_DENSITY_MAX}"
        )
    
    return density


def validate_priority_claim(claim):
    """
    Validate priority claim is an integer within 0-3 range.
    
    Args:
        claim: Value to validate
        
    Returns:
        int: Validated claim
        
    Raises:
        InvalidValueError: If claim is out of range or wrong type
    """
    from src.utils.exceptions import InvalidValueError
    
    try:
        claim = int(claim)
    except (ValueError, TypeError):
        raise InvalidValueError(
            'priority_claim',
            claim,
            "an integer between 0 and 3 (Low=0, Normal=1, High=2, Critical=3)"
        )
    
    if claim < 0 or claim > 3:
        raise InvalidValueError(
            'priority_claim',
            claim,
            "an integer between 0 and 3 (Low=0, Normal=1, High=2, Critical=3)"
        )
    
    return claim


def validate_control_zone(zone):
    """
    Validate control zone if provided.
    
    Args:
        zone: Value to validate (can be None)
        
    Returns:
        str or None: Validated zone or None
        
    Raises:
        InvalidValueError: If zone is provided but invalid
    """
    from src.config import ControlZone
    from src.utils.exceptions import InvalidValueError
    
    if zone is None:
        return None
    
    if zone not in ControlZone.VALID_ZONES:
        raise InvalidValueError(
            'control_zone',
            zone,
            f"one of {ControlZone.VALID_ZONES} or None"
        )
    
    return zone


def validate_description(description):
    """
    Validate and clean description note.
    
    Args:
        description: Value to validate
        
    Returns:
        str: Cleaned description
    """
    if description is None:
        return ''
    return str(description).strip()