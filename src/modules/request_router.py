"""
request_router.py
Module 2: Request Router for the Smart City Traffic AI System.

This module serves as the control-flow manager of the system. Its role is to:
1. Receive the standardized request from Input Preprocessing Module
2. Determine which processing pipeline should be followed
3. Select and sequence the appropriate AI modules
4. Prevent inappropriate module calls
5. Preserve sequencing rules
6. Ensure each request is handled according to operational need

The router is critical because it keeps the system disciplined. It prevents
inappropriate module calls, preserves sequencing rules, and ensures that each
request is handled according to operational need rather than by a one-size-fits-all approach.

Request Category Routing:
- Route_Request: Search module only
- Policy_Check: Logic/Knowledge Base module only
- Control_Allocation_Request: ANN -> Logic/KB -> CSP -> Search
- Emergency_Response_Request: ANN -> Logic/KB -> CSP -> Search
- Integrated_City_Service_Request: ANN -> Logic/KB -> CSP -> Search -> Final Response

Author: [Your Name]
Group: [Your Group]
Course: AL-2002 Artificial Intelligence Lab
"""


from src.config import RequestCategory
from src.models.traffic_request import TrafficRequest
from src.utils.exceptions import InvalidRequestError, InvalidValueError


class RequestRouter:
    """
    Request Router for the Smart City Traffic AI System.
    
    This class determines the processing pipeline for each traffic request
    based on its category. It ensures proper module sequencing and prevents
    unauthorized or inappropriate module activation.
    
    Attributes:
        routing_table (dict): Maps request categories to module sequences
        module_registry (dict): Registry of available module names
    """
    
    # Module name constants for consistency
    MODULE_INPUT = "Input_Preprocessing"
    MODULE_ANN = "ANN_Priority"
    MODULE_LOGIC = "Logic_KnowledgeBase"
    MODULE_CSP = "CSP_Scheduler"
    MODULE_SEARCH = "Search_Navigation"
    MODULE_RESPONSE = "Final_Response"
    
    def __init__(self):
        """
        Initialize the Request Router.
        
        Sets up the routing table that maps each request category to its
        required module sequence. Also initializes the module registry.
        """
        self.routing_table = self._build_routing_table()
        self.module_registry = self._build_module_registry()
    
    # =====================================================================
    # FUNCTION 1: _build_routing_table
    # =====================================================================
    def _build_routing_table(self):
        """
        Build the routing table mapping categories to module sequences.
        
        This internal function defines which modules are activated for each
        request category. The sequences are designed to respect operational
        dependencies (e.g., policy validation must occur before control allocation).
        
        Returns:
            dict: Mapping of request_category -> list of module names
        
        Routing Logic:
        - Route_Request: Simple routing, only needs pathfinding
        - Policy_Check: Rule validation only, no routing needed
        - Control_Allocation_Request: Full pipeline with constraint solving
        - Emergency_Response_Request: Full pipeline with priority prediction
        - Integrated_City_Service_Request: Complete pipeline with all modules
        """
        return {
            RequestCategory.ROUTE_REQUEST: [
                self.MODULE_SEARCH
            ],
            
            RequestCategory.POLICY_CHECK: [
                self.MODULE_LOGIC
            ],
            
            RequestCategory.CONTROL_ALLOCATION_REQUEST: [
                self.MODULE_ANN,
                self.MODULE_LOGIC,
                self.MODULE_CSP,
                self.MODULE_SEARCH
            ],
            
            RequestCategory.EMERGENCY_RESPONSE_REQUEST: [
                self.MODULE_ANN,
                self.MODULE_LOGIC,
                self.MODULE_CSP,
                self.MODULE_SEARCH
            ],
            
            RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST: [
                self.MODULE_ANN,
                self.MODULE_LOGIC,
                self.MODULE_CSP,
                self.MODULE_SEARCH,
                self.MODULE_RESPONSE
            ]
        }
    
    # =====================================================================
    # FUNCTION 2: _build_module_registry
    # =====================================================================
    def _build_module_registry(self):
        """
        Build the module registry with descriptions.
        
        This internal function creates a registry of all available modules
        with their descriptions and dependencies. Used for validation and
        reporting.
        
        Returns:
            dict: Module name -> {description, dependencies, inputs, outputs}
        """
        return {
            self.MODULE_ANN: {
                "description": "ANN Priority Prediction Module - Estimates urgency/priority using neural network",
                "dependencies": [],
                "inputs": ["feature_vector"],
                "outputs": ["predicted_priority"],
                "required_for": [
                    RequestCategory.CONTROL_ALLOCATION_REQUEST,
                    RequestCategory.EMERGENCY_RESPONSE_REQUEST,
                    RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST
                ]
            },
            
            self.MODULE_LOGIC: {
                "description": "Logic/Knowledge Base Module - Validates policy and authorization rules",
                "dependencies": [],
                "inputs": ["vehicle_type", "request_category", "priority", "action"],
                "outputs": ["policy_status", "authorization", "approval"],
                "required_for": [
                    RequestCategory.POLICY_CHECK,
                    RequestCategory.CONTROL_ALLOCATION_REQUEST,
                    RequestCategory.EMERGENCY_RESPONSE_REQUEST,
                    RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST
                ]
            },
            
            self.MODULE_CSP: {
                "description": "CSP Scheduler Module - Assigns signal/control actions under constraints",
                "dependencies": [self.MODULE_LOGIC],
                "inputs": ["approved_request", "control_zones", "constraints"],
                "outputs": ["control_plan", "signal_assignments"],
                "required_for": [
                    RequestCategory.CONTROL_ALLOCATION_REQUEST,
                    RequestCategory.EMERGENCY_RESPONSE_REQUEST,
                    RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST
                ]
            },
            
            self.MODULE_SEARCH: {
                "description": "Search & Navigation Module - Computes optimal routes",
                "dependencies": [],
                "inputs": ["graph", "start", "goal", "algorithm"],
                "outputs": ["route", "cost", "travel_time"],
                "required_for": [
                    RequestCategory.ROUTE_REQUEST,
                    RequestCategory.CONTROL_ALLOCATION_REQUEST,
                    RequestCategory.EMERGENCY_RESPONSE_REQUEST,
                    RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST
                ]
            },
            
            self.MODULE_RESPONSE: {
                "description": "Final Response Module - Aggregates and formats all outputs",
                "dependencies": [self.MODULE_ANN, self.MODULE_LOGIC, self.MODULE_CSP, self.MODULE_SEARCH],
                "inputs": ["all_module_outputs"],
                "outputs": ["formatted_response"],
                "required_for": [
                    RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST
                ]
            }
        }
    
    # =====================================================================
    # FUNCTION 3: route_request
    # =====================================================================
    def route_request(self, traffic_request):
        """
        Determine the processing pipeline for a traffic request.
        
        This is the main routing function that selects which modules should
        be activated based on the request's category. It validates that the
        request category is recognized and returns the appropriate sequence.
        
        Args:
            traffic_request (TrafficRequest): Standardized request object
                from Input Preprocessing Module
        
        Returns:
            dict: Routing decision containing:
                - 'request_id': ID of the request
                - 'request_category': Category of the request
                - 'modules_sequence': Ordered list of modules to execute
                - 'pipeline_description': Human-readable description
                - 'estimated_complexity': Complexity indicator
        
        Raises:
            InvalidRequestError: If request category is not recognized
            InvalidValueError: If traffic_request is not a TrafficRequest object
        
        Example:
            >>> from src.modules.input_preprocessing import InputPreprocessingModule
            >>> input_module = InputPreprocessingModule()
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
            >>> request = input_module.process_request(raw)
            >>> router = RequestRouter()
            >>> decision = router.route_request(request)
            >>> print(decision['modules_sequence'])
            ['ANN_Priority', 'Logic_KnowledgeBase', 'CSP_Scheduler', 'Search_Navigation']
        """
        # Validate input type
        if not isinstance(traffic_request, TrafficRequest):
            raise InvalidValueError(
                'traffic_request',
                type(traffic_request).__name__,
                "TrafficRequest object from Input Preprocessing Module"
            )
        
        request_category = traffic_request.request_category
        
        # Validate that category exists in routing table
        if request_category not in self.routing_table:
            raise InvalidRequestError(
                'request_category',
                f"Unrecognized request category '{request_category}'. "
                f"Valid categories: {list(self.routing_table.keys())}"
            )
        
        # Get module sequence for this category
        modules_sequence = self.routing_table[request_category]
        
        # Build routing decision
        routing_decision = {
            'request_id': traffic_request.request_id,
            'request_category': request_category,
            'modules_sequence': modules_sequence.copy(),
            'pipeline_description': self._get_pipeline_description(
                request_category, modules_sequence
            ),
            'estimated_complexity': self._estimate_complexity(modules_sequence),
            'requires_emergency_handling': traffic_request.is_emergency,
            'validation_notes': []
        }
        
        # Add validation notes for emergency requests
        if traffic_request.is_emergency:
            routing_decision['validation_notes'].append(
                "Emergency vehicle detected - priority processing enabled"
            )
        
        # Add validation notes for control zone requests
        if traffic_request.control_zone is not None:
            routing_decision['validation_notes'].append(
                f"Control zone specified: {traffic_request.control_zone}"
            )
        
        return routing_decision
    
    # =====================================================================
    # FUNCTION 4: _get_pipeline_description
    # =====================================================================
    def _get_pipeline_description(self, category, modules):
        """
        Generate a human-readable description of the processing pipeline.
        
        This internal function creates an explanatory text describing what
        the system will do for this request category.
        
        Args:
            category (str): Request category
            modules (list): Sequence of modules to execute
        
        Returns:
            str: Human-readable pipeline description
        """
        descriptions = {
            RequestCategory.ROUTE_REQUEST: (
                "Standard route guidance: Compute optimal path from current "
                "location to destination using search algorithms."
            ),
            
            RequestCategory.POLICY_CHECK: (
                "Policy validation: Check if requested action is permitted "
                "under current traffic rules and vehicle authorization."
            ),
            
            RequestCategory.CONTROL_ALLOCATION_REQUEST: (
                "Control allocation with constraints: Predict priority, "
                "validate authorization, assign signal controls under safety "
                "constraints, and compute route."
            ),
            
            RequestCategory.EMERGENCY_RESPONSE_REQUEST: (
                "Emergency response support: Predict urgency level, validate "
                "emergency authorization, allocate signal corridor priority, "
                "and compute fastest route."
            ),
            
            RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST: (
                "Integrated city service: Full pipeline with priority prediction, "
                "policy validation, constraint-based control allocation, route "
                "optimization, and comprehensive response generation."
            )
        }
        
        base_desc = descriptions.get(category, "Unknown pipeline")
        
        module_list = " -> ".join(modules)
        
        return f"{base_desc}\nProcessing sequence: {module_list}"
    
    # =====================================================================
    # FUNCTION 5: _estimate_complexity
    # =====================================================================
    def _estimate_complexity(self, modules):
        """
        Estimate computational complexity of the pipeline.
        
        This internal function provides a rough complexity indicator based
        on the number and type of modules in the sequence.
        
        Args:
            modules (list): Sequence of modules
        
        Returns:
            str: Complexity level (Low, Medium, High)
        """
        complexity_scores = {
            self.MODULE_ANN: 2,
            self.MODULE_LOGIC: 1,
            self.MODULE_CSP: 3,  # CSP is most computationally expensive
            self.MODULE_SEARCH: 2,
            self.MODULE_RESPONSE: 1
        }
        
        total_score = sum(
            complexity_scores.get(m, 1) for m in modules
        )
        
        if total_score <= 1:
            return "Low"
        elif total_score <= 4:
            return "Medium"
        else:
            return "High"
    
    # =====================================================================
    # FUNCTION 6: validate_pipeline
    # =====================================================================
    def validate_pipeline(self, routing_decision):
        """
        Validate that a routing decision is internally consistent.
        
        This function checks that the proposed pipeline satisfies all
        module dependencies and sequencing constraints. It ensures that
        no module is called before its dependencies are satisfied.
        
        Args:
            routing_decision (dict): Routing decision from route_request()
        
        Returns:
            dict: Validation result with 'valid' (bool) and 'issues' (list)
        
        Example:
            >>> decision = router.route_request(request)
            >>> validation = router.validate_pipeline(decision)
            >>> print(validation['valid'])
            True
        """
        modules = routing_decision['modules_sequence']
        issues = []
        
        # Check for duplicate modules
        seen = set()
        for module in modules:
            if module in seen:
                issues.append(f"Duplicate module in pipeline: {module}")
            seen.add(module)
        
        # Check dependency satisfaction
        for i, module in enumerate(modules):
            module_info = self.module_registry.get(module, {})
            dependencies = module_info.get('dependencies', [])
            
            for dep in dependencies:
                if dep not in modules:
                    issues.append(
                        f"Module '{module}' requires '{dep}' which is not "
                        f"in the pipeline"
                    )
                else:
                    dep_index = modules.index(dep)
                    if dep_index > i:
                        issues.append(
                            f"Module '{module}' requires '{dep}' but '{dep}' "
                            f"appears later in the sequence"
                        )
        
        # Check category requirements
        category = routing_decision['request_category']
        for module_name, module_info in self.module_registry.items():
            required_for = module_info.get('required_for', [])
            if category in required_for and module_name not in modules:
                issues.append(
                    f"Category '{category}' requires module '{module_name}' "
                    f"which is missing from pipeline"
                )
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'pipeline': modules
        }
    
    # =====================================================================
    # FUNCTION 7: get_module_info
    # =====================================================================
    def get_module_info(self, module_name):
        """
        Get detailed information about a specific module.
        
        Args:
            module_name (str): Name of the module
        
        Returns:
            dict or None: Module information if found, None otherwise
        """
        return self.module_registry.get(module_name)
    
    # =====================================================================
    # FUNCTION 8: get_all_routing_options
    # =====================================================================
    def get_all_routing_options(self):
        """
        Get all available routing configurations.
        
        Returns:
            dict: All category -> module sequence mappings
        """
        return {
            category: {
                'modules': sequence,
                'description': self._get_pipeline_description(category, sequence),
                'complexity': self._estimate_complexity(sequence)
            }
            for category, sequence in self.routing_table.items()
        }
    
    # =====================================================================
    # FUNCTION 9: can_handle_request
    # =====================================================================
    def can_handle_request(self, traffic_request):
        """
        Check if the system can handle a given request.
        
        This function performs a quick check to determine if the request
        category is supported without fully routing it.
        
        Args:
            traffic_request (TrafficRequest): Request to check
        
        Returns:
            dict: Result with 'can_handle' (bool) and 'reason' (str)
        """
        if not isinstance(traffic_request, TrafficRequest):
            return {
                'can_handle': False,
                'reason': "Invalid request object type"
            }
        
        category = traffic_request.request_category
        
        if category not in self.routing_table:
            return {
                'can_handle': False,
                'reason': f"Unsupported request category: {category}"
            }
        
        return {
            'can_handle': True,
            'reason': f"Category '{category}' is supported",
            'modules_required': len(self.routing_table[category])
        }
    
    # =====================================================================
    # FUNCTION 10: display_routing_decision
    # =====================================================================
    def display_routing_decision(self, routing_decision):
        """
        Display a formatted routing decision for demonstration.
        
        Args:
            routing_decision (dict): Routing decision to display
        
        Returns:
            str: Formatted display string
        """
        lines = [
            "=" * 60,
            "REQUEST ROUTING DECISION",
            "=" * 60,
            f"Request ID:           {routing_decision['request_id']}",
            f"Request Category:     {routing_decision['request_category']}",
            f"Complexity Level:     {routing_decision['estimated_complexity']}",
            f"Emergency Handling:   {routing_decision['requires_emergency_handling']}",
            "",
            "PROCESSING PIPELINE:",
            "-" * 60
        ]
        
        for i, module in enumerate(routing_decision['modules_sequence'], 1):
            module_info = self.module_registry.get(module, {})
            desc = module_info.get('description', 'No description')
            lines.append(f"  {i}. {module}")
            lines.append(f"     {desc[:50]}...")
        
        if routing_decision['validation_notes']:
            lines.extend([
                "",
                "VALIDATION NOTES:",
                "-" * 60
            ])
            for note in routing_decision['validation_notes']:
                lines.append(f"  • {note}")
        
        lines.extend([
            "",
            "PIPELINE DESCRIPTION:",
            "-" * 60,
            routing_decision['pipeline_description'],
            "=" * 60
        ])
        
        return '\n'.join(lines)


# =====================================================================
# Standalone testing functionality
# =====================================================================
if __name__ == "__main__":
    """
    Standalone test for the Request Router Module.
    Run this file directly to test module functionality.
    """
    import sys
    import os
    
    # Add parent to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.modules.input_preprocessing import InputPreprocessingModule
    
    print("=" * 70)
    print("REQUEST ROUTER MODULE - STANDALONE TEST")
    print("=" * 70)
    
    # Create modules
    input_module = InputPreprocessingModule()
    router = RequestRouter()
    
    print("\n" + "=" * 70)
    print("AVAILABLE ROUTING OPTIONS")
    print("=" * 70)
    options = router.get_all_routing_options()
    for category, info in options.items():
        print(f"\n{category}:")
        print(f"  Modules: {info['modules']}")
        print(f"  Complexity: {info['complexity']}")
    
    # Test Case 1: Route_Request
    print("\n" + "=" * 70)
    print("TEST CASE 1: Route_Request (Civilian)")
    print("=" * 70)
    raw_1 = {
        'request_id': 'REQ-001',
        'vehicle_type': 'Civilian',
        'request_category': 'Route_Request',
        'current_location': 'North_Station',
        'destination': 'Airport_Road',
        'traffic_density': 4
    }
    
    try:
        request_1 = input_module.process_request(raw_1)
        decision_1 = router.route_request(request_1)
        print(router.display_routing_decision(decision_1))
        
        validation_1 = router.validate_pipeline(decision_1)
        print(f"\nPipeline Valid: {validation_1['valid']}")
        if validation_1['issues']:
            print(f"Issues: {validation_1['issues']}")
        print("\n✓ Test Case 1 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 1 FAILED: {e}")
    
    # Test Case 2: Policy_Check
    print("\n" + "=" * 70)
    print("TEST CASE 2: Policy_Check (Emergency Vehicle)")
    print("=" * 70)
    raw_2 = {
        'request_id': 'REQ-002',
        'vehicle_type': 'Ambulance',
        'request_category': 'Policy_Check',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital',
        'incident_severity': 'High',
        'control_zone': 'Central_Junction'
    }
    
    try:
        request_2 = input_module.process_request(raw_2)
        decision_2 = router.route_request(request_2)
        print(router.display_routing_decision(decision_2))
        
        validation_2 = router.validate_pipeline(decision_2)
        print(f"\nPipeline Valid: {validation_2['valid']}")
        print("\n✓ Test Case 2 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 2 FAILED: {e}")
    
    # Test Case 3: Control_Allocation_Request
    print("\n" + "=" * 70)
    print("TEST CASE 3: Control_Allocation_Request")
    print("=" * 70)
    raw_3 = {
        'request_id': 'REQ-003',
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
        request_3 = input_module.process_request(raw_3)
        decision_3 = router.route_request(request_3)
        print(router.display_routing_decision(decision_3))
        
        validation_3 = router.validate_pipeline(decision_3)
        print(f"\nPipeline Valid: {validation_3['valid']}")
        print("\n✓ Test Case 3 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 3 FAILED: {e}")
    
    # Test Case 4: Emergency_Response_Request
    print("\n" + "=" * 70)
    print("TEST CASE 4: Emergency_Response_Request (Ambulance)")
    print("=" * 70)
    raw_4 = {
        'request_id': 'REQ-004',
        'vehicle_type': 'Ambulance',
        'request_category': 'Emergency_Response_Request',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital',
        'incident_severity': 'High',
        'time_sensitivity': 'High',
        'traffic_density': 9,
        'priority_claim': 3,
        'control_zone': 'S1_Central_Junction',
        'description_note': 'Cardiac emergency - patient critical'
    }
    
    try:
        request_4 = input_module.process_request(raw_4)
        decision_4 = router.route_request(request_4)
        print(router.display_routing_decision(decision_4))
        
        validation_4 = router.validate_pipeline(decision_4)
        print(f"\nPipeline Valid: {validation_4['valid']}")
        print("\n✓ Test Case 4 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 4 FAILED: {e}")
    
    # Test Case 5: Integrated_City_Service_Request
    print("\n" + "=" * 70)
    print("TEST CASE 5: Integrated_City_Service_Request")
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
        decision_5 = router.route_request(request_5)
        print(router.display_routing_decision(decision_5))
        
        validation_5 = router.validate_pipeline(decision_5)
        print(f"\nPipeline Valid: {validation_5['valid']}")
        print("\n✓ Test Case 5 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 5 FAILED: {e}")
    
    # Test Case 6: Invalid Request Category
    print("\n" + "=" * 70)
    print("TEST CASE 6: Invalid Request Category")
    print("=" * 70)
    raw_6 = {
        'request_id': 'REQ-006',
        'vehicle_type': 'Civilian',
        'request_category': 'Invalid_Category',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital'
    }
    
    try:
        request_6 = input_module.process_request(raw_6)
        decision_6 = router.route_request(request_6)
        print("✗ Test Case 6 FAILED: Should have raised exception")
    except InvalidRequestError as e:
        print(f"✓ Test Case 6 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 6 FAILED: Unexpected error: {e}")
    
    # Test Case 7: can_handle_request check
    print("\n" + "=" * 70)
    print("TEST CASE 7: can_handle_request Check")
    print("=" * 70)
    raw_7 = {
        'request_id': 'REQ-007',
        'vehicle_type': 'Ambulance',
        'request_category': 'Emergency_Response_Request',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital',
        'incident_severity': 'High',
        'time_sensitivity': 'High',
        'traffic_density': 8,
        'priority_claim': 3
    }
    
    try:
        request_7 = input_module.process_request(raw_7)
        check = router.can_handle_request(request_7)
        print(f"Can Handle: {check['can_handle']}")
        print(f"Reason: {check['reason']}")
        print(f"Modules Required: {check['modules_required']}")
        print("\n✓ Test Case 7 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 7 FAILED: {e}")
    
    # Test Case 8: get_module_info
    print("\n" + "=" * 70)
    print("TEST CASE 8: get_module_info")
    print("=" * 70)
    info = router.get_module_info('ANN_Priority')
    if info:
        print(f"Module: ANN_Priority")
        print(f"Description: {info['description']}")
        print(f"Dependencies: {info['dependencies']}")
        print(f"Inputs: {info['inputs']}")
        print(f"Outputs: {info['outputs']}")
        print("\n✓ Test Case 8 PASSED")
    else:
        print("✗ Test Case 8 FAILED: Module info not found")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)