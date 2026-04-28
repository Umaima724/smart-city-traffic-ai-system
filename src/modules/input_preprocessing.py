"""
input_preprocessing.py
Module 1: Input & Preprocessing Module for the Smart City Traffic AI System.

This module serves as the entry point of the system. Its role is to:
1. Receive a traffic request in structured form
2. Validate all submitted information against system rules
3. Normalize field values to standard internal formats
4. Build a standard internal request object (TrafficRequest)
5. Prepare the ANN feature vector for requests needing ML-based priority prediction

The module ensures that road identifiers, control-zone labels, and request
categories follow predefined system schemas. It acts as both a validation
layer and a preparation layer for downstream AI processing.

All inputs are assumed to be manually entered in structured form.
The system does NOT process free-text commands and does NOT use NLP.
Input preprocessing is limited to field validation, normalization,
and categorical mapping.

Author: [Your Name]
Group: [Your Group]
Course: AL-2002 Artificial Intelligence Lab
"""


import json
from src.config import (
    VehicleType,
    RequestCategory,
    SeverityLevel,
    TimeSensitivity,
    ControlZone,
    TRAFFIC_DENSITY_MIN,
    TRAFFIC_DENSITY_MAX,
    ANN_INPUT_FEATURES
)
from src.models.traffic_request import TrafficRequest
from src.utils.exceptions import (
    InvalidRequestError,
    MissingFieldError,
    InvalidValueError
)
from src.utils.validators import (
    validate_request_id,
    validate_vehicle_type,
    validate_request_category,
    validate_location,
    validate_severity,
    validate_time_sensitivity,
    validate_traffic_density,
    validate_priority_claim,
    validate_control_zone,
    validate_description
)
from src.utils.graph_loader import load_weighted_graph, get_edge_cost


class InputPreprocessingModule:
    """
    Input and Preprocessing Module for traffic request handling.
    
    This class encapsulates all functionality for receiving, validating,
    normalizing, and preparing traffic requests for further processing.
    
    Attributes:
        city_graph (dict): Loaded weighted city graph for distance estimation
        validation_errors (list): Accumulated validation errors
    """
    
    def __init__(self):
        """
        Initialize the Input Preprocessing Module.
        
        Loads the city graph for distance estimation and initializes
        the error tracking list.
        """
        self.city_graph = None
        self.validation_errors = []
        self._load_city_graph()
    
    def _load_city_graph(self):
        """
        Load the weighted city graph for distance calculations.
        
        This internal method attempts to load the city graph data.
        If loading fails, distance estimation will use fallback values.
        """
        try:
            self.city_graph = load_weighted_graph()
        except Exception:
            # Graph loading failure is not fatal for input processing
            # Distance estimation will use fallback
            self.city_graph = None
    
    # =====================================================================
    # FUNCTION 1: validate_request
    # =====================================================================
    def validate_request(self, raw_input):
        """
        Validate a raw traffic request against all system rules.
        
        This function performs comprehensive validation of all fields
        in a traffic request. It checks for required fields, valid
        data types, and acceptable value ranges. All validation errors
        are collected and reported together rather than failing on
        the first error.
        
        Args:
            raw_input (dict): Dictionary containing raw request fields.
                Expected keys:
                - request_id (str, required): Unique identifier
                - vehicle_type (str, required): Type of vehicle
                - request_category (str, required): Category of request
                - current_location (str, required): Starting location
                - destination (str, required): Target location
                - incident_severity (str, optional): Severity level
                - time_sensitivity (str, optional): Time sensitivity
                - traffic_density (int, optional): Traffic density 0-10
                - priority_claim (int, optional): Priority claim 0-3
                - control_zone (str, optional): Control zone identifier
                - description_note (str, optional): Description text
        
        Returns:
            dict: Dictionary of validated and cleaned field values
        
        Raises:
            InvalidRequestError: If validation fails, contains all errors
        
        Example:
            >>> raw = {
            ...     'request_id': 'REQ-001',
            ...     'vehicle_type': 'Ambulance',
            ...     'request_category': 'Emergency_Response_Request',
            ...     'current_location': 'Central_Junction',
            ...     'destination': 'City_Hospital',
            ...     'incident_severity': 'High',
            ...     'time_sensitivity': 'High',
            ...     'traffic_density': 8,
            ...     'priority_claim': 3
            ... }
            >>> module = InputPreprocessingModule()
            >>> validated = module.validate_request(raw)
        """
        self.validation_errors = []
        validated = {}
        
        # --- Validate Required Fields ---
        required_fields = TrafficRequest.REQUIRED_FIELDS
        
        for field in required_fields:
            if field not in raw_input or raw_input[field] is None:
                self.validation_errors.append(
                    MissingFieldError(field)
                )
        
        # If required fields are missing, cannot proceed further
        if self.validation_errors:
            error_msg = "Validation failed with missing required fields:\n"
            for err in self.validation_errors:
                error_msg += f"  - {err}\n"
            raise InvalidRequestError(
                'required_fields',
                error_msg.strip()
            )
        
        # --- Validate Individual Fields ---
        try:
            validated['request_id'] = validate_request_id(
                raw_input['request_id']
            )
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        try:
            validated['vehicle_type'] = validate_vehicle_type(
                raw_input['vehicle_type']
            )
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        try:
            validated['request_category'] = validate_request_category(
                raw_input['request_category']
            )
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        try:
            validated['current_location'] = validate_location(
                raw_input['current_location'],
                'current_location'
            )
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        try:
            validated['destination'] = validate_location(
                raw_input['destination'],
                'destination'
            )
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        # Check that current and destination are different
        if (validated.get('current_location') == 
            validated.get('destination')):
            self.validation_errors.append(
                InvalidValueError(
                    'destination',
                    validated.get('destination'),
                    "a location different from current_location"
                )
            )
        
        # --- Validate Optional Fields ---
        # incident_severity
        severity = raw_input.get('incident_severity', 'None')
        try:
            validated['incident_severity'] = validate_severity(severity)
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        # time_sensitivity
        time_sens = raw_input.get('time_sensitivity', 'Normal')
        try:
            validated['time_sensitivity'] = validate_time_sensitivity(
                time_sens
            )
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        # traffic_density
        density = raw_input.get('traffic_density', 0)
        try:
            validated['traffic_density'] = validate_traffic_density(density)
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        # priority_claim
        claim = raw_input.get('priority_claim', 0)
        try:
            validated['priority_claim'] = validate_priority_claim(claim)
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        # control_zone
        zone = raw_input.get('control_zone', None)
        try:
            validated['control_zone'] = validate_control_zone(zone)
        except InvalidValueError as e:
            self.validation_errors.append(e)
        
        # description_note
        desc = raw_input.get('description_note', '')
        validated['description_note'] = validate_description(desc)
        
        # --- Report All Errors ---
        if self.validation_errors:
            error_msg = "Validation failed with the following errors:\n"
            for err in self.validation_errors:
                error_msg += f"  - {err}\n"
            raise InvalidRequestError(
                'multiple_fields',
                error_msg.strip()
            )
        
        return validated
    
    # =====================================================================
    # FUNCTION 2: normalize_request
    # =====================================================================
    def normalize_request(self, validated_input):
        """
        Normalize validated request fields to standard internal formats.
        
        This function converts categorical string values to numeric
        representations suitable for AI processing. It ensures
        consistency across all modules by using standardized mappings.
        
        Normalization includes:
        - Converting severity strings to numeric values
        - Converting time sensitivity to binary numeric
        - Converting vehicle types to numeric codes
        - Converting priority claims to standardized scale
        
        Args:
            validated_input (dict): Dictionary from validate_request()
        
        Returns:
            dict: Normalized request with additional numeric fields
        
        Example:
            >>> validated = {
            ...     'vehicle_type': 'Ambulance',
            ...     'incident_severity': 'High',
            ...     'time_sensitivity': 'High',
            ...     'priority_claim': 3
            ... }
            >>> normalized = module.normalize_request(validated)
            >>> print(normalized['vehicle_type_numeric'])
            1
        """
        normalized = validated_input.copy()
        
        # Normalize vehicle type to numeric code
        vehicle_type_map = {
            VehicleType.CIVILIAN: 0,
            VehicleType.AMBULANCE: 1,
            VehicleType.FIRE_TRUCK: 2,
            VehicleType.POLICE: 3
        }
        normalized['vehicle_type_numeric'] = vehicle_type_map.get(
            validated_input['vehicle_type'], 0
        )
        
        # Normalize severity to numeric
        normalized['severity_numeric'] = SeverityLevel.NUMERIC_MAP.get(
            validated_input['incident_severity'], 0
        )
        
        # Normalize time sensitivity to numeric
        normalized['time_sensitivity_numeric'] = (
            TimeSensitivity.NUMERIC_MAP.get(
                validated_input['time_sensitivity'], 0
            )
        )
        
        # Normalize priority claim (already numeric, just ensure range)
        normalized['priority_claim_numeric'] = max(
            0, min(3, validated_input['priority_claim'])
        )
        
        # Traffic density is already numeric (0-10)
        normalized['traffic_density_normalized'] = (
            validated_input['traffic_density'] / 10.0
        )
        
        return normalized
    
    # =====================================================================
    # FUNCTION 3: estimate_distance
    # =====================================================================
    def estimate_distance(self, current_location, destination):
        """
        Estimate distance between two locations in the city graph.
        
        This function attempts to find the shortest path distance
        between current location and destination using the loaded
        weighted city graph. If the graph is unavailable, it returns
        a fallback estimate based on predefined distances.
        
        Args:
            current_location (str): Starting location/node
            destination (str): Target location/node
        
        Returns:
            float: Estimated distance in kilometers
        
        Example:
            >>> distance = module.estimate_distance(
            ...     'Central_Junction', 'City_Hospital'
            ... )
            >>> print(distance)
            6.0
        """
        # If graph is loaded, try to find shortest path
        if self.city_graph is not None:
            try:
                # Use UCS to find shortest path cost
                from src.modules.search_navigation import SearchNavigationModule
                search_module = SearchNavigationModule()
                _, cost = search_module.ucs(
                    self.city_graph,
                    current_location,
                    destination
                )
                return float(cost)
            except Exception:
                pass  # Fall through to fallback
        
        # Fallback: Predefined distance estimates
        fallback_distances = {
            ('Central_Junction', 'City_Hospital'): 6.0,
            ('Central_Junction', 'North_Station'): 3.0,
            ('Central_Junction', 'West_Terminal'): 4.0,
            ('Central_Junction', 'East_Market'): 3.0,
            ('Central_Junction', 'South_Residential'): 4.0,
            ('North_Station', 'River_Bridge'): 4.0,
            ('North_Station', 'Traffic_Control_Center'): 2.0,
            ('River_Bridge', 'Police_HQ'): 2.0,
            ('River_Bridge', 'Stadium'): 2.0,
            ('Stadium', 'Airport_Road'): 5.0,
            ('Stadium', 'East_Market'): 2.0,
            ('East_Market', 'City_Hospital'): 3.0,
            ('Airport_Road', 'South_Residential'): 2.0,
            ('City_Hospital', 'South_Residential'): 6.0,
            ('West_Terminal', 'Fire_Station'): 2.0,
            ('West_Terminal', 'Industrial_Zone'): 4.0,
            ('Police_HQ', 'Traffic_Control_Center'): 2.0,
        }
        
        # Check both directions
        direct = fallback_distances.get((current_location, destination))
        if direct is not None:
            return direct
        
        reverse = fallback_distances.get((destination, current_location))
        if reverse is not None:
            return reverse
        
        # Ultimate fallback: arbitrary estimate
        return 5.0
    
    # =====================================================================
    # FUNCTION 4: build_feature_vector
    # =====================================================================
    def build_feature_vector(self, normalized_request):
        """
        Build the ANN feature vector for priority prediction.
        
        This function prepares a 6-element feature vector from the
        normalized request data. The feature vector is used as input
        to the ANN Priority Prediction Module.
        
        Feature vector structure:
        [vehicle_type, severity, time_sensitivity, traffic_density, 
         distance, priority_claim]
        
        Args:
            normalized_request (dict): Normalized request from 
                                      normalize_request()
        
        Returns:
            list: 6-element feature vector as list of floats
        
        Raises:
            InvalidRequestError: If required normalized fields are missing
        
        Example:
            >>> normalized = {
            ...     'vehicle_type_numeric': 1,
            ...     'severity_numeric': 3,
            ...     'time_sensitivity_numeric': 1,
            ...     'traffic_density': 8,
            ...     'priority_claim_numeric': 3
            ... }
            >>> features = module.build_feature_vector(normalized)
            >>> print(features)
            [1.0, 3.0, 1.0, 8.0, 6.0, 3.0]
        """
        required_normalized_fields = [
            'vehicle_type_numeric',
            'severity_numeric',
            'time_sensitivity_numeric',
            'traffic_density',
            'priority_claim_numeric'
        ]
        
        # Check that all required fields exist
        for field in required_normalized_fields:
            if field not in normalized_request:
                raise InvalidRequestError(
                    field,
                    f"Missing normalized field '{field}' required for "
                    f"feature vector construction"
                )
        
        # Estimate distance if not already present
        if 'estimated_distance' not in normalized_request:
            distance = self.estimate_distance(
                normalized_request['current_location'],
                normalized_request['destination']
            )
            normalized_request['estimated_distance'] = distance
        
        # Construct feature vector
        feature_vector = [
            float(normalized_request['vehicle_type_numeric']),
            float(normalized_request['severity_numeric']),
            float(normalized_request['time_sensitivity_numeric']),
            float(normalized_request['traffic_density']),
            float(normalized_request['estimated_distance']),
            float(normalized_request['priority_claim_numeric'])
        ]
        
        # Validate feature vector length
        if len(feature_vector) != ANN_INPUT_FEATURES:
            raise InvalidRequestError(
                'feature_vector',
                f"Expected {ANN_INPUT_FEATURES} features, got "
                f"{len(feature_vector)}"
            )
        
        return feature_vector
    
    # =====================================================================
    # FUNCTION 5: process_request (Main Entry Point)
    # =====================================================================
    def process_request(self, raw_input):
        """
        Main entry point for processing a raw traffic request.
        
        This function orchestrates the complete input preprocessing
        pipeline: validation, normalization, distance estimation,
        and feature vector preparation. It returns a fully prepared
        TrafficRequest object ready for routing to AI modules.
        
        Processing pipeline:
        1. Validate all input fields
        2. Normalize categorical values to numeric
        3. Estimate route distance
        4. Build ANN feature vector
        5. Construct TrafficRequest object
        
        Args:
            raw_input (dict): Raw traffic request dictionary
        
        Returns:
            TrafficRequest: Fully prepared request object
        
        Raises:
            InvalidRequestError: If validation or processing fails
        
        Example:
            >>> raw = {
            ...     'request_id': 'REQ-001',
            ...     'vehicle_type': 'Ambulance',
            ...     'request_category': 'Emergency_Response_Request',
            ...     'current_location': 'Central_Junction',
            ...     'destination': 'City_Hospital',
            ...     'incident_severity': 'High',
            ...     'time_sensitivity': 'High',
            ...     'traffic_density': 8,
            ...     'priority_claim': 3
            ... }
            >>> module = InputPreprocessingModule()
            >>> request = module.process_request(raw)
            >>> print(request)
            TrafficRequest[REQ-001] (Ambulance | Emergency_Response_Request | 
            Central_Junction -> City_Hospital)
        """
        # Step 1: Validate the request
        validated = self.validate_request(raw_input)
        
        # Step 2: Normalize the validated request
        normalized = self.normalize_request(validated)
        
        # Step 3: Estimate distance
        distance = self.estimate_distance(
            normalized['current_location'],
            normalized['destination']
        )
        normalized['estimated_distance'] = distance
        
        # Step 4: Build feature vector
        feature_vector = self.build_feature_vector(normalized)
        normalized['feature_vector'] = feature_vector
        
        # Step 5: Create TrafficRequest object
        traffic_request = TrafficRequest(**validated)
        
        # Set derived fields
        traffic_request.estimated_distance = distance
        traffic_request.normalized_features = feature_vector
        
        return traffic_request
    
    # =====================================================================
    # FUNCTION 6: display_request_summary
    # =====================================================================
    def display_request_summary(self, traffic_request):
        """
        Display a formatted summary of a processed traffic request.
        
        This helper function prints a human-readable summary of the
        request, including all fields and derived values. Useful for
        debugging and demonstration purposes.
        
        Args:
            traffic_request (TrafficRequest): Processed request object
        
        Returns:
            str: Formatted summary string
        
        Example:
            >>> summary = module.display_request_summary(request)
            >>> print(summary)
        """
        summary_lines = [
            "=" * 50,
            "TRAFFIC REQUEST SUMMARY",
            "=" * 50,
            f"Request ID:        {traffic_request.request_id}",
            f"Vehicle Type:      {traffic_request.vehicle_type}",
            f"Emergency Vehicle: {traffic_request.is_emergency}",
            f"Request Category:  {traffic_request.request_category}",
            f"Current Location:  {traffic_request.current_location}",
            f"Destination:       {traffic_request.destination}",
            f"Incident Severity: {traffic_request.incident_severity}",
            f"Time Sensitivity:  {traffic_request.time_sensitivity}",
            f"Traffic Density:   {traffic_request.traffic_density}/10",
            f"Priority Claim:    {traffic_request.priority_claim}",
            f"Control Zone:      {traffic_request.control_zone or 'None'}",
            f"Description:       {traffic_request.description_note or 'None'}",
            "-" * 50,
            "DERIVED VALUES",
            "-" * 50,
            f"Estimated Distance: {traffic_request.estimated_distance:.2f} km",
            f"Feature Vector:     {traffic_request.normalized_features}",
            "=" * 50
        ]
        
        return '\n'.join(summary_lines)


# =====================================================================
# Standalone testing functionality
# =====================================================================
if __name__ == "__main__":
    """
    Standalone test for the Input Preprocessing Module.
    Run this file directly to test module functionality.
    """
    print("=" * 60)
    print("INPUT & PREPROCESSING MODULE - STANDALONE TEST")
    print("=" * 60)
    
    # Create module instance
    module = InputPreprocessingModule()
    
    # Test Case 1: Valid Emergency Request
    print("\n--- Test Case 1: Valid Emergency Request ---")
    test_request_1 = {
        'request_id': 'REQ-001',
        'vehicle_type': 'Ambulance',
        'request_category': 'Emergency_Response_Request',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital',
        'incident_severity': 'High',
        'time_sensitivity': 'High',
        'traffic_density': 8,
        'priority_claim': 3,
        'control_zone': 'Central_Junction',
        'description_note': 'Patient with cardiac emergency'
    }
    
    try:
        result = module.process_request(test_request_1)
        print(module.display_request_summary(result))
        print("\n✓ Test Case 1 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 1 FAILED: {e}")
    
    # Test Case 2: Valid Civilian Route Request
    print("\n--- Test Case 2: Valid Civilian Route Request ---")
    test_request_2 = {
        'request_id': 'REQ-002',
        'vehicle_type': 'Civilian',
        'request_category': 'Route_Request',
        'current_location': 'North_Station',
        'destination': 'Airport_Road',
        'traffic_density': 4,
        'description_note': 'Daily commute to airport'
    }
    
    try:
        result = module.process_request(test_request_2)
        print(module.display_request_summary(result))
        print("\n✓ Test Case 2 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 2 FAILED: {e}")
    
    # Test Case 3: Invalid Request - Missing Required Field
    print("\n--- Test Case 3: Missing Required Field ---")
    test_request_3 = {
        'request_id': 'REQ-003',
        'vehicle_type': 'Ambulance',
        # Missing request_category
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital'
    }
    
    try:
        result = module.process_request(test_request_3)
        print("✗ Test Case 3 FAILED: Should have raised exception")
    except InvalidRequestError as e:
        print(f"✓ Test Case 3 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 3 FAILED: Unexpected error: {e}")
    
    # Test Case 4: Invalid Request - Invalid Vehicle Type
    print("\n--- Test Case 4: Invalid Vehicle Type ---")
    test_request_4 = {
        'request_id': 'REQ-004',
        'vehicle_type': 'Bicycle',  # Invalid
        'request_category': 'Route_Request',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital'
    }
    
    try:
        result = module.process_request(test_request_4)
        print("✗ Test Case 4 FAILED: Should have raised exception")
    except InvalidRequestError as e:
        print(f"✓ Test Case 4 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 4 FAILED: Unexpected error: {e}")
    
    # Test Case 5: Invalid Request - Same Origin and Destination
    print("\n--- Test Case 5: Same Origin and Destination ---")
    test_request_5 = {
        'request_id': 'REQ-005',
        'vehicle_type': 'Civilian',
        'request_category': 'Route_Request',
        'current_location': 'Central_Junction',
        'destination': 'Central_Junction'  # Same as origin
    }
    
    try:
        result = module.process_request(test_request_5)
        print("✗ Test Case 5 FAILED: Should have raised exception")
    except InvalidRequestError as e:
        print(f"✓ Test Case 5 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 5 FAILED: Unexpected error: {e}")
    
    # Test Case 6: Invalid Request - Out of Range Values
    print("\n--- Test Case 6: Out of Range Values ---")
    test_request_6 = {
        'request_id': 'REQ-006',
        'vehicle_type': 'Civilian',
        'request_category': 'Route_Request',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital',
        'traffic_density': 15,  # Out of range (0-10)
        'priority_claim': 5     # Out of range (0-3)
    }
    
    try:
        result = module.process_request(test_request_6)
        print("✗ Test Case 6 FAILED: Should have raised exception")
    except InvalidRequestError as e:
        print(f"✓ Test Case 6 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 6 FAILED: Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)