"""
logic_knowledge_base.py
Module 4: Logic/Knowledge Base Module for the Smart City Traffic AI System.

This module is responsible for policy validation and rule-based reasoning.
It checks whether an action is allowed under traffic policy, whether emergency
priority can be granted, whether signal override conditions are satisfied,
and whether a requested control action is logically consistent with the
knowledge base.

In practical terms, this module protects the system from unsafe or unauthorized
actions. It ensures that emergency requests are validated before the system
adjusts signals or grants corridor priority. It is therefore the gatekeeper
before any constrained control allocation is performed.

The module implements all required predicates and rules from the project
specification:

PREDICATES:
- Vehicle(v), EmergencyVehicle(v), CivilianVehicle(v)
- Location(l), SignalZone(z), Hospital(h)
- Request(req), RequestType(req, type)
- CurrentLocation(v, l), Destination(v, l)
- IncidentSeverity(v, level), TimeSensitive(v)
- Priority(v, level)
- Authorized(v, action), AllowedAction(v, action)
- EmergencyCorridor(v), EmergencyRoute(v), SignalOverride(z)
- Approved(v, req), Rejected(v, req)

RULES (from project specification pages 7-8):
- EmergencyVehicle(v) ∧ IncidentSeverity(v, High) → Priority(v, Critical)
- EmergencyVehicle(v) ∧ TimeSensitive(v) → Priority(v, High)
- CivilianVehicle(v) → Priority(v, Normal)
- EmergencyVehicle(v) ∧ SignalZone(z) → Authorized(v, SignalOverride(z))
- CivilianVehicle(v) ∧ SignalZone(z) → ¬Authorized(v, SignalOverride(z))
- EmergencyVehicle(v) ∧ Destination(v, h) ∧ Hospital(h) → EmergencyCorridor(v)
- EmergencyCorridor(v) → Authorized(v, EmergencyRoute)
- Authorized(v, action) → AllowedAction(v, action)
- ¬AllowedAction(v, action) → Rejected(v, req)
- Priority(v, Critical) ∧ ¬Authorized(v, action) → ¬AllowedAction(v, action)
- Priority(v, Critical) ∧ Authorized(v, EmergencyRoute) → AllowedAction(v, SignalOverride)
- AllowedAction(v, action) → Approved(v, req)
- RequestType(req, Route_Request) → Approved(v, req)
- RequestType(req, Policy_Check) ∧ Authorized(v, action) → Approved(v, req)
- RequestType(req, Policy_Check) ∧ ¬Authorized(v, action) → Rejected(v, req)
- RequestType(req, Control_Allocation_Request) ∧ AllowedAction(v, action) → Approved(v, req)
- RequestType(req, Control_Allocation_Request) ∧ ¬AllowedAction(v, action) → Rejected(v, req)
- RequestType(req, Emergency_Response_Request) ∧ Priority(v, level) ∧ Authorized(v, EmergencyRoute) → Approved(v, req)
- RequestType(req, Integrated_City_Service_Request) ∧ Priority(v, Critical) ∧ Authorized(v, EmergencyRoute) ∧ AllowedAction(v, action) → Approved(v, req)

Author: [Your Name]
Group: [Your Group]
Course: AL-2002 Artificial Intelligence Lab
"""


import json
import os

from src.config import (
    VehicleType,
    RequestCategory,
    SeverityLevel,
    PriorityLevel,
    ControlZone,
     TimeSensitivity,
    TRAFFIC_RULES_PATH
)
from src.models.traffic_request import TrafficRequest
from src.utils.exceptions import (
    InvalidRequestError,
    UnauthorizedActionError,
    DataLoadError
)


class LogicKnowledgeBase:
    """
    Logic/Knowledge Base Module for policy validation and rule-based reasoning.
    
    This class implements predicate logic and rule-based inference for
    traffic management decisions. It serves as the policy gatekeeper,
    ensuring only authorized actions are approved.
    
    Attributes:
        rules (dict): Loaded traffic rules from JSON
        predicates (dict): Current predicate state for active request
        inference_log (list): Log of all inference steps performed
    """
    
    def __init__(self):
        """
        Initialize the Logic/Knowledge Base Module.
        
        Loads traffic rules from JSON file and initializes the predicate
        state and inference log.
        """
        self.rules = None
        self.predicates = {}
        self.inference_log = []
        self._load_rules()
    
    # =====================================================================
    # FUNCTION 1: _load_rules
    # =====================================================================
    def _load_rules(self):
        """
        Load traffic policy rules from JSON file.
        
        This internal function loads the structured rule database that
        defines vehicle authorizations, priority rules, and request
        type requirements.
        
        Raises:
            DataLoadError: If rules file cannot be loaded
        """
        try:
            if os.path.exists(TRAFFIC_RULES_PATH):
                with open(TRAFFIC_RULES_PATH, 'r') as f:
                    self.rules = json.load(f)
            else:
                # Use default rules if file not found
                self.rules = self._get_default_rules()
        except Exception as e:
            raise DataLoadError(f"{TRAFFIC_RULES_PATH} ({str(e)})")
    
    # =====================================================================
    # FUNCTION 2: _get_default_rules
    # =====================================================================
    def _get_default_rules(self):
        """
        Get default traffic rules if JSON file is unavailable.
        
        Returns:
            dict: Default rule structure
        """
        return {
            "vehicle_types": {
                "Civilian": {
                    "is_emergency": False,
                    "max_priority": "Normal",
                    "allowed_actions": ["Route_Request"],
                    "forbidden_actions": ["SignalOverride", "EmergencyRoute"]
                },
                "Ambulance": {
                    "is_emergency": True,
                    "max_priority": "Critical",
                    "allowed_actions": ["Route_Request", "SignalOverride", "EmergencyRoute"],
                    "forbidden_actions": []
                },
                "Fire_Truck": {
                    "is_emergency": True,
                    "max_priority": "Critical",
                    "allowed_actions": ["Route_Request", "SignalOverride", "EmergencyRoute"],
                    "forbidden_actions": []
                },
                "Police": {
                    "is_emergency": True,
                    "max_priority": "Critical",
                    "allowed_actions": ["Route_Request", "SignalOverride", "EmergencyRoute"],
                    "forbidden_actions": []
                }
            }
        }
    
    # =====================================================================
    # FUNCTION 3: reset_predicates
    # =====================================================================
    def reset_predicates(self):
        """
        Reset all predicates for a new reasoning session.
        
        Clears the predicate state and inference log to prepare for
        evaluating a new request.
        """
        self.predicates = {}
        self.inference_log = []
    
    # =====================================================================
    # FUNCTION 4: assert_predicate
    # =====================================================================
    def assert_predicate(self, predicate_name, value, reason=None):
        """
        Assert a predicate with a given value and optional reason.
        
        This function adds a predicate to the knowledge base state
        and logs the inference step.
        
        Args:
            predicate_name (str): Name of the predicate
            value: Value to assign (bool, str, etc.)
            reason (str, optional): Explanation for this assertion
        
        Returns:
            bool: True if assertion was new or changed, False otherwise
        """
        old_value = self.predicates.get(predicate_name)
        is_new = (predicate_name not in self.predicates or 
                  self.predicates[predicate_name] != value)
        
        self.predicates[predicate_name] = value
        
        self.inference_log.append({
            'predicate': predicate_name,
            'value': value,
            'old_value': old_value,
            'reason': reason or "Direct assertion",
            'is_new': is_new
        })
        
        return is_new
    
    # =====================================================================
    # FUNCTION 5: get_predicate
    # =====================================================================
    def get_predicate(self, predicate_name, default=None):
        """
        Get the value of a predicate.
        
        Args:
            predicate_name (str): Name of the predicate
            default: Default value if predicate not found
        
        Returns:
            Value of predicate or default
        """
        return self.predicates.get(predicate_name, default)
    
    # =====================================================================
    # FUNCTION 6: is_true
    # =====================================================================
    def is_true(self, predicate_name):
        """
        Check if a predicate is true.
        
        Args:
            predicate_name (str): Name of the predicate
        
        Returns:
            bool: True if predicate exists and is truthy
        """
        return bool(self.predicates.get(predicate_name, False))
    
    # =====================================================================
    # PREDICATE DEFINITIONS
    # =====================================================================
    
    def predicate_Vehicle(self, v):
        """Predicate: Vehicle(v) - v is a vehicle."""
        return self.assert_predicate(f"Vehicle({v})", True, 
                                     f"{v} is a registered vehicle")
    
    def predicate_EmergencyVehicle(self, v):
        """Predicate: EmergencyVehicle(v) - v is an emergency vehicle."""
        vehicle_type = self.get_predicate(f"vehicle_type({v})")
        is_emergency = vehicle_type in [VehicleType.AMBULANCE, 
                                        VehicleType.FIRE_TRUCK, 
                                        VehicleType.POLICE]
        return self.assert_predicate(f"EmergencyVehicle({v})", is_emergency,
                                     f"{v} vehicle type is {vehicle_type}")
    
    def predicate_CivilianVehicle(self, v):
        """Predicate: CivilianVehicle(v) - v is a civilian vehicle."""
        vehicle_type = self.get_predicate(f"vehicle_type({v})")
        is_civilian = (vehicle_type == VehicleType.CIVILIAN)
        return self.assert_predicate(f"CivilianVehicle({v})", is_civilian,
                                     f"{v} vehicle type is {vehicle_type}")
    
    def predicate_Location(self, l):
        """Predicate: Location(l) - l is a valid location."""
        is_valid = l in ControlZone.VALID_ZONES
        return self.assert_predicate(f"Location({l})", is_valid,
                                     f"{l} is a valid city location")
    
    def predicate_SignalZone(self, z):
        """Predicate: SignalZone(z) - z is a valid signal zone."""
        # Signal zones are derived from control zones with S_ prefix
        is_valid = (z in ControlZone.VALID_ZONES or 
                    any(z.startswith(prefix) for prefix in ['S1_', 'S2_', 'S3_', 'S4_', 'S5_']))
        return self.assert_predicate(f"SignalZone({z})", is_valid,
                                     f"{z} is a valid signal control zone")
    
    def predicate_Hospital(self, h):
        """Predicate: Hospital(h) - h is a hospital location."""
        is_hospital = (h == ControlZone.CITY_HOSPITAL)
        return self.assert_predicate(f"Hospital({h})", is_hospital,
                                     f"{h} is a hospital")
    
    def predicate_Request(self, req):
        """Predicate: Request(req) - req is a valid request."""
        return self.assert_predicate(f"Request({req})", True,
                                     f"{req} is a valid traffic request")
    
    def predicate_RequestType(self, req, req_type):
        """Predicate: RequestType(req, type) - req has type."""
        return self.assert_predicate(f"RequestType({req},{req_type})", True,
                                     f"{req} is of type {req_type}")
    
    def predicate_CurrentLocation(self, v, l):
        """Predicate: CurrentLocation(v, l) - v is at location l."""
        return self.assert_predicate(f"CurrentLocation({v},{l})", True,
                                     f"{v} is currently at {l}")
    
    def predicate_Destination(self, v, l):
        """Predicate: Destination(v, l) - v is going to location l."""
        return self.assert_predicate(f"Destination({v},{l})", True,
                                     f"{v} destination is {l}")
    
    def predicate_IncidentSeverity(self, v, level):
        """Predicate: IncidentSeverity(v, level) - v has severity level."""
        return self.assert_predicate(f"IncidentSeverity({v},{level})", True,
                                     f"{v} incident severity is {level}")
    
    def predicate_TimeSensitive(self, v):
        """Predicate: TimeSensitive(v) - v has time-sensitive request."""
        time_sens = self.get_predicate(f"time_sensitivity({v})")
        is_sensitive = (time_sens == "High")
        return self.assert_predicate(f"TimeSensitive({v})", is_sensitive,
                                     f"{v} time sensitivity is {time_sens}")
    
    def predicate_Priority(self, v, level):
        """Predicate: Priority(v, level) - v has priority level."""
        return self.assert_predicate(f"Priority({v},{level})", True,
                                     f"{v} assigned priority: {level}")
    
    def predicate_Authorized(self, v, action):
        """Predicate: Authorized(v, action) - v is authorized for action."""
        return self.assert_predicate(f"Authorized({v},{action})", True,
                                     f"{v} is authorized for {action}")
    
    def predicate_NotAuthorized(self, v, action):
        """Predicate: ¬Authorized(v, action) - v is NOT authorized."""
        return self.assert_predicate(f"Authorized({v},{action})", False,
                                     f"{v} is NOT authorized for {action}")
    
    def predicate_AllowedAction(self, v, action):
        """Predicate: AllowedAction(v, action) - action is allowed."""
        return self.assert_predicate(f"AllowedAction({v},{action})", True,
                                     f"{action} is allowed for {v}")
    
    def predicate_NotAllowedAction(self, v, action):
        """Predicate: ¬AllowedAction(v, action) - action is NOT allowed."""
        return self.assert_predicate(f"AllowedAction({v},{action})", False,
                                     f"{action} is NOT allowed for {v}")
    
    def predicate_EmergencyCorridor(self, v):
        """Predicate: EmergencyCorridor(v) - v gets emergency corridor."""
        return self.assert_predicate(f"EmergencyCorridor({v})", True,
                                     f"{v} is assigned emergency corridor")
    
    def predicate_EmergencyRoute(self, v):
        """Predicate: EmergencyRoute(v) - v gets emergency route."""
        return self.assert_predicate(f"EmergencyRoute({v})", True,
                                     f"{v} is assigned emergency route")
    
    def predicate_SignalOverride(self, z):
        """Predicate: SignalOverride(z) - signal override at zone z."""
        return self.assert_predicate(f"SignalOverride({z})", True,
                                     f"Signal override authorized at {z}")
    
    def predicate_Approved(self, v, req):
        """Predicate: Approved(v, req) - request is approved."""
        return self.assert_predicate(f"Approved({v},{req})", True,
                                     f"Request {req} APPROVED for {v}")
    
    def predicate_Rejected(self, v, req):
        """Predicate: Rejected(v, req) - request is rejected."""
        return self.assert_predicate(f"Approved({v},{req})", False,
                                     f"Request {req} REJECTED for {v}")
    
    # =====================================================================
    # FUNCTION 7: evaluate_priority_rules
    # =====================================================================
    def evaluate_priority_rules(self, vehicle_id, vehicle_type, 
                                incident_severity, time_sensitivity):
        """
        Evaluate priority assignment rules.
        
        RULES IMPLEMENTED:
        - EmergencyVehicle(v) ∧ IncidentSeverity(v, High) → Priority(v, Critical)
        - EmergencyVehicle(v) ∧ TimeSensitive(v) → Priority(v, High)
        - CivilianVehicle(v) → Priority(v, Normal)
        
        Args:
            vehicle_id (str): Vehicle identifier
            vehicle_type (str): Type of vehicle
            incident_severity (str): Severity level
            time_sensitivity (str): Time sensitivity level
        
        Returns:
            str: Assigned priority level (Low, Normal, High, Critical)
        """
        self.assert_predicate(f"vehicle_type({vehicle_id})", vehicle_type)
        
        # Rule: EmergencyVehicle(v) ∧ IncidentSeverity(v, High) → Priority(v, Critical)
        is_emergency = vehicle_type in [VehicleType.AMBULANCE, 
                                        VehicleType.FIRE_TRUCK, 
                                        VehicleType.POLICE]
        
        if is_emergency and incident_severity == SeverityLevel.HIGH:
            self.predicate_EmergencyVehicle(vehicle_id)
            self.predicate_IncidentSeverity(vehicle_id, incident_severity)
            self.predicate_Priority(vehicle_id, PriorityLevel.CRITICAL)
            return PriorityLevel.CRITICAL
        
        # Rule: EmergencyVehicle(v) ∧ TimeSensitive(v) → Priority(v, High)
        if is_emergency and time_sensitivity == TimeSensitivity.HIGH:
            self.predicate_EmergencyVehicle(vehicle_id)
            self.predicate_TimeSensitive(vehicle_id)
            self.predicate_Priority(vehicle_id, PriorityLevel.HIGH)
            return PriorityLevel.HIGH
        
        # Rule: CivilianVehicle(v) → Priority(v, Normal)
        if not is_emergency:
            self.predicate_CivilianVehicle(vehicle_id)
            self.predicate_Priority(vehicle_id, PriorityLevel.NORMAL)
            return PriorityLevel.NORMAL
        
        # Default for emergency without high severity or time sensitivity
        self.predicate_Priority(vehicle_id, PriorityLevel.NORMAL)
        return PriorityLevel.NORMAL
    
    # =====================================================================
    # FUNCTION 8: evaluate_authorization_rules
    # =====================================================================
    def evaluate_authorization_rules(self, vehicle_id, vehicle_type,
                                     destination, control_zone):
        """
        Evaluate authorization rules for actions.
        
        RULES IMPLEMENTED:
        - EmergencyVehicle(v) ∧ SignalZone(z) → Authorized(v, SignalOverride(z))
        - CivilianVehicle(v) ∧ SignalZone(z) → ¬Authorized(v, SignalOverride(z))
        - EmergencyVehicle(v) ∧ Destination(v, h) ∧ Hospital(h) → EmergencyCorridor(v)
        - EmergencyCorridor(v) → Authorized(v, EmergencyRoute)
        - Authorized(v, action) → AllowedAction(v, action)
        
        Args:
            vehicle_id (str): Vehicle identifier
            vehicle_type (str): Type of vehicle
            destination (str): Destination location
            control_zone (str): Control zone (can be None)
        
        Returns:
            dict: Authorization results for various actions
        """
        is_emergency = vehicle_type in [VehicleType.AMBULANCE, 
                                        VehicleType.FIRE_TRUCK, 
                                        VehicleType.POLICE]
        
        authorizations = {
            'signal_override': False,
            'emergency_route': False,
            'emergency_corridor': False
        }
        
        # Rule: EmergencyVehicle(v) ∧ SignalZone(z) → Authorized(v, SignalOverride(z))
        # Rule: CivilianVehicle(v) ∧ SignalZone(z) → ¬Authorized(v, SignalOverride(z))
        if control_zone is not None:
            self.predicate_SignalZone(control_zone)
            
            if is_emergency:
                self.predicate_EmergencyVehicle(vehicle_id)
                self.predicate_Authorized(vehicle_id, f"SignalOverride({control_zone})")
                authorizations['signal_override'] = True
            else:
                self.predicate_CivilianVehicle(vehicle_id)
                self.predicate_NotAuthorized(vehicle_id, f"SignalOverride({control_zone})")
                authorizations['signal_override'] = False
        
        # Rule: EmergencyVehicle(v) ∧ Destination(v, h) ∧ Hospital(h) → EmergencyCorridor(v)
        if is_emergency and destination == ControlZone.CITY_HOSPITAL:
            self.predicate_EmergencyVehicle(vehicle_id)
            self.predicate_Destination(vehicle_id, destination)
            self.predicate_Hospital(destination)
            self.predicate_EmergencyCorridor(vehicle_id)
            authorizations['emergency_corridor'] = True
            
            # Rule: EmergencyCorridor(v) → Authorized(v, EmergencyRoute)
            self.predicate_Authorized(vehicle_id, "EmergencyRoute")
            authorizations['emergency_route'] = True
        
        # Rule: Authorized(v, action) → AllowedAction(v, action)
        if authorizations['signal_override']:
            self.predicate_AllowedAction(vehicle_id, "SignalOverride")
        else:
            self.predicate_NotAllowedAction(vehicle_id, "SignalOverride")
        
        if authorizations['emergency_route']:
            self.predicate_AllowedAction(vehicle_id, "EmergencyRoute")
        else:
            self.predicate_NotAllowedAction(vehicle_id, "EmergencyRoute")
        
        return authorizations
    
    # =====================================================================
    # FUNCTION 9: evaluate_request_type_rules
    # =====================================================================
    def evaluate_request_type_rules(self, vehicle_id, vehicle_type,
                                    request_category, priority_level,
                                    authorizations):
        """
        Evaluate request type specific rules.
        
        RULES IMPLEMENTED:
        - RequestType(req, Route_Request) → Approved(v, req)
        - RequestType(req, Policy_Check) ∧ Authorized(v, action) → Approved(v, req)
        - RequestType(req, Policy_Check) ∧ ¬Authorized(v, action) → Rejected(v, req)
        - RequestType(req, Control_Allocation_Request) ∧ AllowedAction(v, action) → Approved(v, req)
        - RequestType(req, Control_Allocation_Request) ∧ ¬AllowedAction(v, action) → Rejected(v, req)
        - RequestType(req, Emergency_Response_Request) ∧ Priority(v, level) ∧ Authorized(v, EmergencyRoute) → Approved(v, req)
        - RequestType(req, Integrated_City_Service_Request) ∧ Priority(v, Critical) ∧ Authorized(v, EmergencyRoute) ∧ AllowedAction(v, action) → Approved(v, req)
        
        Args:
            vehicle_id (str): Vehicle identifier
            vehicle_type (str): Type of vehicle
            request_category (str): Category of request
            priority_level (str): Predicted/assigned priority
            authorizations (dict): Authorization results
        
        Returns:
            dict: Decision with status and reason
        """
        self.predicate_RequestType(vehicle_id, request_category)
        
        # Rule: RequestType(req, Route_Request) → Approved(v, req)
        if request_category == RequestCategory.ROUTE_REQUEST:
            self.predicate_Approved(vehicle_id, request_category)
            return {
                'status': 'Approved',
                'reason': 'Route requests are always approved'
            }
        
        # Rule: RequestType(req, Policy_Check) ∧ Authorized(v, action) → Approved(v, req)
        # Rule: RequestType(req, Policy_Check) ∧ ¬Authorized(v, action) → Rejected(v, req)
        if request_category == RequestCategory.POLICY_CHECK:
            if authorizations.get('signal_override', False):
                self.predicate_Approved(vehicle_id, request_category)
                return {
                    'status': 'Approved',
                    'reason': 'Policy check passed - vehicle is authorized'
                }
            else:
                self.predicate_Rejected(vehicle_id, request_category)
                return {
                    'status': 'Rejected',
                    'reason': 'Policy check failed - vehicle not authorized for this action'
                }
        
        # Rule: RequestType(req, Control_Allocation_Request) ∧ AllowedAction(v, action) → Approved(v, req)
        # Rule: RequestType(req, Control_Allocation_Request) ∧ ¬AllowedAction(v, action) → Rejected(v, req)
        if request_category == RequestCategory.CONTROL_ALLOCATION_REQUEST:
            if (authorizations.get('signal_override', False) and 
                authorizations.get('emergency_route', False)):
                self.predicate_Approved(vehicle_id, request_category)
                return {
                    'status': 'Approved',
                    'reason': 'Control allocation approved - all actions allowed'
                }
            else:
                self.predicate_Rejected(vehicle_id, request_category)
                return {
                    'status': 'Rejected',
                    'reason': 'Control allocation rejected - insufficient authorization'
                }
        
        # Rule: RequestType(req, Emergency_Response_Request) ∧ Priority(v, level) ∧ Authorized(v, EmergencyRoute) → Approved(v, req)
        if request_category == RequestCategory.EMERGENCY_RESPONSE_REQUEST:
            has_priority = priority_level in [PriorityLevel.HIGH, PriorityLevel.CRITICAL]
            has_emergency_route = authorizations.get('emergency_route', False)
            
            if has_priority and has_emergency_route:
                self.predicate_Approved(vehicle_id, request_category)
                return {
                    'status': 'Approved',
                    'reason': f'Emergency response approved - priority {priority_level} with emergency route'
                }
            else:
                self.predicate_Rejected(vehicle_id, request_category)
                missing = []
                if not has_priority:
                    missing.append('sufficient priority')
                if not has_emergency_route:
                    missing.append('emergency route authorization')
                return {
                    'status': 'Rejected',
                    'reason': f'Emergency response rejected - missing: {", ".join(missing)}'
                }
        
        # Rule: RequestType(req, Integrated_City_Service_Request) ∧ Priority(v, Critical) ∧ Authorized(v, EmergencyRoute) ∧ AllowedAction(v, action) → Approved(v, req)
        if request_category == RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST:
            is_critical = (priority_level == PriorityLevel.CRITICAL)
            has_emergency_route = authorizations.get('emergency_route', False)
            has_allowed_action = authorizations.get('signal_override', False)
            
            if is_critical and has_emergency_route and has_allowed_action:
                self.predicate_Approved(vehicle_id, request_category)
                return {
                    'status': 'Approved',
                    'reason': 'Integrated city service approved - critical priority with full authorization'
                }
            else:
                self.predicate_Rejected(vehicle_id, request_category)
                missing = []
                if not is_critical:
                    missing.append('Critical priority')
                if not has_emergency_route:
                    missing.append('emergency route')
                if not has_allowed_action:
                    missing.append('control action authorization')
                return {
                    'status': 'Rejected',
                    'reason': f'Integrated service rejected - missing: {", ".join(missing)}'
                }
        
        # Default: Unknown request type
        return {
            'status': 'Rejected',
            'reason': f'Unknown request category: {request_category}'
        }
    
    # =====================================================================
    # FUNCTION 10: validate_policy (Main Entry Point)
    # =====================================================================
    def validate_policy(self, traffic_request, predicted_priority=None):
        """
        Main entry point for policy validation.
        
        This function orchestrates the complete policy validation pipeline:
        1. Reset predicates for new session
        2. Assert basic predicates from request
        3. Evaluate priority rules
        4. Evaluate authorization rules
        5. Evaluate request type rules
        6. Return comprehensive validation result
        
        Args:
            traffic_request (TrafficRequest): Standardized request
            predicted_priority (str, optional): Priority from ANN module.
                                              If None, computed from rules.
        
        Returns:
            dict: Comprehensive validation result containing:
                - 'status': Approved or Rejected
                - 'priority_level': Assigned priority
                - 'authorizations': Dict of action authorizations
                - 'reason': Explanation of decision
                - 'predicates': All asserted predicates
                - 'inference_log': Step-by-step reasoning
        
        Raises:
            InvalidRequestError: If request is invalid
        
        Example:
            >>> from src.modules.input_preprocessing import InputPreprocessingModule
            >>> input_mod = InputPreprocessingModule()
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
            >>> request = input_mod.process_request(raw)
            >>> kb = LogicKnowledgeBase()
            >>> result = kb.validate_policy(request)
            >>> print(result['status'])
            'Approved'
        """
        if not isinstance(traffic_request, TrafficRequest):
            raise InvalidRequestError(
                'traffic_request',
                "Expected TrafficRequest object"
            )
        
        # Reset for new reasoning session
        self.reset_predicates()
        
        vehicle_id = traffic_request.request_id
        vehicle_type = traffic_request.vehicle_type
        request_category = traffic_request.request_category
        current_location = traffic_request.current_location
        destination = traffic_request.destination
        incident_severity = traffic_request.incident_severity
        time_sensitivity = traffic_request.time_sensitivity
        control_zone = traffic_request.control_zone
        
        # Step 1: Assert basic predicates
        self.predicate_Vehicle(vehicle_id)
        self.predicate_Request(vehicle_id)
        self.predicate_CurrentLocation(vehicle_id, current_location)
        self.predicate_Destination(vehicle_id, destination)
        self.predicate_Location(current_location)
        self.predicate_Location(destination)
        
        # Step 2: Evaluate priority rules
        if predicted_priority is not None:
            # Use ANN-predicted priority
            self.predicate_Priority(vehicle_id, predicted_priority)
            priority_level = predicted_priority
        else:
            # Compute priority from rules
            priority_level = self.evaluate_priority_rules(
                vehicle_id, vehicle_type,
                incident_severity, time_sensitivity
            )
        
        # Step 3: Evaluate authorization rules
        authorizations = self.evaluate_authorization_rules(
            vehicle_id, vehicle_type,
            destination, control_zone
        )
        
        # Step 4: Evaluate request type rules
        decision = self.evaluate_request_type_rules(
            vehicle_id, vehicle_type,
            request_category, priority_level,
            authorizations
        )
        
        # Step 5: Additional safety rules
        # Rule: Priority(v, Critical) ∧ ¬Authorized(v, action) → ¬AllowedAction(v, action)
        if priority_level == PriorityLevel.CRITICAL:
            if not authorizations.get('signal_override', False):
                self.predicate_NotAllowedAction(vehicle_id, "SignalOverride")
        
        # Rule: Priority(v, Critical) ∧ Authorized(v, EmergencyRoute) → AllowedAction(v, SignalOverride)
        if (priority_level == PriorityLevel.CRITICAL and 
            authorizations.get('emergency_route', False)):
            self.predicate_AllowedAction(vehicle_id, "SignalOverride")
            authorizations['signal_override'] = True
        
        # Rule: AllowedAction(v, action) → Approved(v, req)
        # Rule: ¬AllowedAction(v, action) → Rejected(v, req)
        # (Handled in evaluate_request_type_rules)
        
        # Compile final result
        return {
            'status': decision['status'],
            'priority_level': priority_level,
            'authorizations': authorizations,
            'reason': decision['reason'],
            'vehicle_type': vehicle_type,
            'is_emergency': traffic_request.is_emergency,
            'request_category': request_category,
            'predicates': self.predicates.copy(),
            'inference_log': self.inference_log.copy()
        }
    
    # =====================================================================
    # FUNCTION 11: check_action_allowed
    # =====================================================================
    def check_action_allowed(self, vehicle_type, action):
        """
        Quick check if a vehicle type is allowed a specific action.
        
        Args:
            vehicle_type (str): Type of vehicle
            action (str): Action to check
        
        Returns:
            dict: Result with allowed status and reason
        """
        vehicle_rules = self.rules.get('vehicle_types', {}).get(vehicle_type, {})
        allowed_actions = vehicle_rules.get('allowed_actions', [])
        forbidden_actions = vehicle_rules.get('forbidden_actions', [])
        
        is_allowed = action in allowed_actions and action not in forbidden_actions
        
        return {
            'allowed': is_allowed,
            'vehicle_type': vehicle_type,
            'action': action,
            'reason': (f"{action} is {'allowed' if is_allowed else 'forbidden'} "
                      f"for {vehicle_type}")
        }
    
    # =====================================================================
    # FUNCTION 12: get_inference_log
    # =====================================================================
    def get_inference_log(self):
        """
        Get the complete inference log.
        
        Returns:
            list: All inference steps with predicates and reasons
        """
        return self.inference_log.copy()
    
    # =====================================================================
    # FUNCTION 13: display_reasoning
    # =====================================================================
    def display_reasoning(self, validation_result):
        """
        Display formatted reasoning chain.
        
        Args:
            validation_result (dict): Result from validate_policy()
        
        Returns:
            str: Formatted reasoning display
        """
        lines = [
            "=" * 70,
            "LOGIC/KNOWLEDGE BASE - REASONING CHAIN",
            "=" * 70,
            f"Request ID:       {validation_result.get('vehicle_type', 'N/A')}",
            f"Vehicle Type:     {validation_result.get('vehicle_type', 'N/A')}",
            f"Emergency:        {validation_result.get('is_emergency', False)}",
            f"Request Category: {validation_result.get('request_category', 'N/A')}",
            f"Priority Level:   {validation_result.get('priority_level', 'N/A')}",
            f"Final Decision:   {validation_result.get('status', 'N/A')}",
            "",
            "INFERENCE STEPS:",
            "-" * 70
        ]
        
        for i, step in enumerate(validation_result.get('inference_log', []), 1):
            status = "NEW" if step['is_new'] else "UPD"
            lines.append(
                f"  {i:3d}. [{status}] {step['predicate']:<40} = {str(step['value']):<10}"
                f"  ({step['reason']})"
            )
        
        lines.extend([
            "",
            "AUTHORIZATION SUMMARY:",
            "-" * 70
        ])
        
        auths = validation_result.get('authorizations', {})
        for action, allowed in auths.items():
            symbol = "✓" if allowed else "✗"
            lines.append(f"  {symbol} {action}: {'YES' if allowed else 'NO'}")
        
        lines.extend([
            "",
            f"DECISION REASON: {validation_result.get('reason', 'N/A')}",
            "=" * 70
        ])
        
        return '\n'.join(lines)


# =============================================================================
# Standalone testing functionality
# =============================================================================
if __name__ == "__main__":
    """
    Standalone test for the Logic/Knowledge Base Module.
    Run this file directly to test module functionality.
    """
    import sys
    import os
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.modules.input_preprocessing import InputPreprocessingModule
    
    print("=" * 70)
    print("LOGIC/KNOWLEDGE BASE MODULE - STANDALONE TEST")
    print("=" * 70)
    
    # Create modules
    input_module = InputPreprocessingModule()
    kb = LogicKnowledgeBase()
    
    # Test Case 1: Emergency Vehicle (Ambulance) to Hospital
    print("\n" + "=" * 70)
    print("TEST CASE 1: Emergency Vehicle (Ambulance) to Hospital")
    print("Expected: APPROVED with Critical priority")
    print("=" * 70)
    
    raw_1 = {
        'request_id': 'REQ-001',
        'vehicle_type': 'Ambulance',
        'request_category': 'Emergency_Response_Request',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital',
        'incident_severity': 'High',
        'time_sensitivity': 'High',
        'traffic_density': 9,
        'priority_claim': 3,
        'control_zone': 'Central_Junction'
    }
    
    try:
        request_1 = input_module.process_request(raw_1)
        result_1 = kb.validate_policy(request_1)
        print(kb.display_reasoning(result_1))
        assert result_1['status'] == 'Approved', "Should be approved"
        assert result_1['priority_level'] == 'Critical', "Should be Critical"
        print("\n✓ Test Case 1 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 2: Civilian Route Request
    print("\n" + "=" * 70)
    print("TEST CASE 2: Civilian Route Request")
    print("Expected: APPROVED with Normal priority")
    print("=" * 70)
    
    raw_2 = {
        'request_id': 'REQ-002',
        'vehicle_type': 'Civilian',
        'request_category': 'Route_Request',
        'current_location': 'North_Station',
        'destination': 'Airport_Road',
        'traffic_density': 4,
        'priority_claim': 0
    }
    
    try:
        request_2 = input_module.process_request(raw_2)
        result_2 = kb.validate_policy(request_2)
        print(kb.display_reasoning(result_2))
        assert result_2['status'] == 'Approved', "Should be approved"
        assert result_2['priority_level'] == 'Normal', "Should be Normal"
        print("\n✓ Test Case 2 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 3: Civilian Policy Check (Signal Override)
    print("\n" + "=" * 70)
    print("TEST CASE 3: Civilian Policy Check (Signal Override)")
    print("Expected: REJECTED - civilian cannot override signals")
    print("=" * 70)
    
    raw_3 = {
        'request_id': 'REQ-003',
        'vehicle_type': 'Civilian',
        'request_category': 'Policy_Check',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital',
        'control_zone': 'Central_Junction'
    }
    
    try:
        request_3 = input_module.process_request(raw_3)
        result_3 = kb.validate_policy(request_3)
        print(kb.display_reasoning(result_3))
        assert result_3['status'] == 'Rejected', "Should be rejected"
        assert result_3['authorizations']['signal_override'] == False
        print("\n✓ Test Case 3 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 4: Police Control Allocation
    print("\n" + "=" * 70)
    print("TEST CASE 4: Police Control Allocation")
    print("Expected: APPROVED - police authorized for control")
    print("=" * 70)
    
    raw_4 = {
        'request_id': 'REQ-004',
        'vehicle_type': 'Police',
        'request_category': 'Control_Allocation_Request',
        'current_location': 'Police_HQ',
        'destination': 'Traffic_Control_Center',
        'incident_severity': 'Medium',
        'time_sensitivity': 'High',
        'traffic_density': 7,
        'priority_claim': 2,
        'control_zone': 'S1_Central_Junction'
    }
    
    try:
        request_4 = input_module.process_request(raw_4)
        result_4 = kb.validate_policy(request_4)
        print(kb.display_reasoning(result_4))
        assert result_4['status'] == 'Approved', "Should be approved"
        print("\n✓ Test Case 4 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 5: Fire Truck Integrated Service
    print("\n" + "=" * 70)
    print("TEST CASE 5: Fire Truck Integrated City Service")
    print("Expected: APPROVED with Critical priority")
    print("=" * 70)
    
    raw_5 = {
        'request_id': 'REQ-005',
        'vehicle_type': 'Fire_Truck',
        'request_category': 'Integrated_City_Service_Request',
        'current_location': 'Fire_Station',
        'destination': 'City_Hospital',
        'incident_severity': 'High',
        'time_sensitivity': 'High',
        'traffic_density': 8,
        'priority_claim': 3,
        'control_zone': 'S5_City_Hospital'
    }
    
    try:
        request_5 = input_module.process_request(raw_5)
        result_5 = kb.validate_policy(request_5)
        print(kb.display_reasoning(result_5))
        assert result_5['status'] == 'Approved', "Should be approved"
        assert result_5['priority_level'] == 'Critical', "Should be Critical"
        print("\n✓ Test Case 5 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 6: Emergency without Hospital Destination
    print("\n" + "=" * 70)
    print("TEST CASE 6: Ambulance NOT going to Hospital")
    print("Expected: REJECTED - no emergency corridor without hospital")
    print("=" * 70)
    
    raw_6 = {
        'request_id': 'REQ-006',
        'vehicle_type': 'Ambulance',
        'request_category': 'Emergency_Response_Request',
        'current_location': 'Central_Junction',
        'destination': 'Industrial_Zone',  # Not a hospital
        'incident_severity': 'High',
        'time_sensitivity': 'High',
        'traffic_density': 9,
        'priority_claim': 3
    }
    
    try:
        request_6 = input_module.process_request(raw_6)
        result_6 = kb.validate_policy(request_6)
        print(kb.display_reasoning(result_6))
        # Should be rejected because no emergency route to non-hospital
        assert result_6['status'] == 'Rejected', "Should be rejected"
        print("\n✓ Test Case 6 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 7: Check Action Allowed Helper
    print("\n" + "=" * 70)
    print("TEST CASE 7: check_action_allowed Helper")
    print("=" * 70)
    
    checks = [
        ('Civilian', 'Route_Request'),
        ('Civilian', 'SignalOverride'),
        ('Ambulance', 'SignalOverride'),
        ('Police', 'EmergencyRoute'),
        ('Fire_Truck', 'Route_Request')
    ]
    
    for vtype, action in checks:
        result = kb.check_action_allowed(vtype, action)
        status = "✓" if result['allowed'] else "✗"
        print(f"  {status} {vtype} -> {action}: {'ALLOWED' if result['allowed'] else 'FORBIDDEN'}")
    
    print("\n✓ Test Case 7 PASSED")
    
    # Test Case 8: With ANN Predicted Priority
    print("\n" + "=" * 70)
    print("TEST CASE 8: Using ANN Predicted Priority")
    print("=" * 70)
    
    raw_8 = {
        'request_id': 'REQ-008',
        'vehicle_type': 'Ambulance',
        'request_category': 'Emergency_Response_Request',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital',
        'incident_severity': 'Medium',
        'time_sensitivity': 'Normal',
        'traffic_density': 5,
        'priority_claim': 2
    }
    
    try:
        request_8 = input_module.process_request(raw_8)
        # Force use ANN priority (Critical) instead of rule-based
        result_8 = kb.validate_policy(request_8, predicted_priority='Critical')
        print(kb.display_reasoning(result_8))
        assert result_8['priority_level'] == 'Critical', "Should use ANN priority"
        print("\n✓ Test Case 8 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 8 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)