"""
exceptions.py
Custom exception classes for the Smart City Traffic AI System.
Provides specific error types for different failure scenarios.
"""


class SmartCityTrafficError(Exception):
    """Base exception for all system errors."""
    pass


class InvalidRequestError(SmartCityTrafficError):
    """
    Raised when a traffic request contains invalid or missing data.
    
    Attributes:
        field_name (str): The field that caused the error
        message (str): Explanation of the error
    """
    def __init__(self, field_name, message):
        self.field_name = field_name
        self.message = message
        super().__init__(f"Invalid request field '{field_name}': {message}")


class MissingFieldError(InvalidRequestError):
    """
    Raised when a required field is missing from the request.
    
    Attributes:
        field_name (str): The missing field name
    """
    def __init__(self, field_name):
        super().__init__(field_name, "This field is required but was not provided")


class InvalidValueError(InvalidRequestError):
    """
    Raised when a field contains a value outside acceptable range or set.
    
    Attributes:
        field_name (str): The field with invalid value
        provided_value: The actual value provided
        expected_values (str): Description of expected values
    """
    def __init__(self, field_name, provided_value, expected_values):
        self.provided_value = provided_value
        self.expected_values = expected_values
        super().__init__(
            field_name,
            f"Received '{provided_value}', expected {expected_values}"
        )


class UnauthorizedActionError(SmartCityTrafficError):
    """
    Raised when a vehicle requests an action it is not authorized for.
    
    Attributes:
        vehicle_type (str): Type of vehicle making request
        action (str): The unauthorized action
    """
    def __init__(self, vehicle_type, action):
        self.vehicle_type = vehicle_type
        self.action = action
        super().__init__(
            f"Vehicle type '{vehicle_type}' is not authorized for action '{action}'"
        )


class NoValidRouteError(SmartCityTrafficError):
    """
    Raised when no valid route can be found between two locations.
    
    Attributes:
        start (str): Starting location
        goal (str): Destination location
    """
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        super().__init__(f"No valid route found from '{start}' to '{goal}'")


class CSPNoSolutionError(SmartCityTrafficError):
    """
    Raised when the CSP solver cannot find a valid assignment.
    
    Attributes:
        variables (list): Variables that could not be assigned
    """
    def __init__(self, variables):
        self.variables = variables
        super().__init__(
            f"CSP solver could not find valid assignment for variables: {variables}"
        )


class DataLoadError(SmartCityTrafficError):
    """
    Raised when system data files cannot be loaded or parsed.
    
    Attributes:
        file_path (str): Path to the file that failed to load
    """
    def __init__(self, file_path):
        self.file_path = file_path
        super().__init__(f"Failed to load data from '{file_path}'")