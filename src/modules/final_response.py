"""
final_response.py
Module 7: Final Response Layer for the Smart City Traffic AI System.

This module aggregates the outputs of the modules that were actually used for
the request. It may present a selected route, a validated status, a predicted
priority level, an assigned control plan, or an integrated response message.
This layer is essential because operational systems must not only make decisions;
they must also communicate them clearly.

The final response should be selective, operationally meaningful, and explainable.
It should not include information from modules that were not used for the current
request. For example, a standard route request should not display an ANN priority
field if no urgency prediction was performed.

Response Design Principles:
- Selective: Only include fields from modules that were used
- Operationally meaningful: Relevant to the user's operational role
- Explainable: Clear reasoning for the decision
- Actionable: Provides concrete next steps

Depending on the request, the final response may include:
- Recommended route
- Estimated travel or delay information
- Predicted priority level
- Policy validation result
- Assigned signal-control or lane-control action
- Decision message and explanatory text

Author: [Your Name]
Group: [Your Group]
Course: AL-2002 Artificial Intelligence Lab
"""


from src.config import RequestCategory, PriorityLevel
from src.models.traffic_request import TrafficRequest
from src.models.response import SystemResponse
from src.utils.exceptions import InvalidRequestError


class FinalResponseModule:
    """
    Final Response Layer for the Smart City Traffic AI System.
    
    This class aggregates outputs from various AI modules and formats
    them into coherent, operationally meaningful responses. It ensures
    that responses are selective, explainable, and actionable.
    
    Attributes:
        response_templates (dict): Templates for different request categories
    """
    
    def __init__(self):
        """
        Initialize the Final Response Module.
        
        Sets up response templates for different request categories
        and operational roles.
        """
        self.response_templates = self._build_templates()
    
    # =====================================================================
    # FUNCTION 1: _build_templates
    # =====================================================================
    def _build_templates(self):
        """
        Build response templates for each request category.
        
        These templates define the structure and messaging for different
        types of responses, ensuring consistency across the system.
        
        Returns:
            dict: Templates for each request category
        """
        return {
            RequestCategory.ROUTE_REQUEST: {
                'title': 'Route Recommendation',
                'message_pattern': 'Optimal route from {start} to {destination} computed.',
                'includes': ['route', 'time']
            },
            
            RequestCategory.POLICY_CHECK: {
                'title': 'Policy Validation Result',
                'message_pattern': 'Request validation: {status}. {reason}',
                'includes': ['policy', 'authorization']
            },
            
            RequestCategory.CONTROL_ALLOCATION_REQUEST: {
                'title': 'Control Allocation Decision',
                'message_pattern': 'Signal control plan assigned. {status}: {reason}',
                'includes': ['priority', 'policy', 'control', 'route']
            },
            
            RequestCategory.EMERGENCY_RESPONSE_REQUEST: {
                'title': 'Emergency Response Coordination',
                'message_pattern': 'Emergency priority {priority} validated. {status}: {reason}',
                'includes': ['priority', 'policy', 'control', 'route']
            },
            
            RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST: {
                'title': 'Integrated City Service Response',
                'message_pattern': 'Full service coordination complete. {status}: {reason}',
                'includes': ['priority', 'policy', 'control', 'route', 'summary']
            }
        }
    
    # =====================================================================
    # FUNCTION 2: generate_response
    # =====================================================================
    def generate_response(self, traffic_request, routing_decision,
                          module_outputs):
        """
        Generate the final system response by aggregating module outputs.
        
        This is the main entry point for response generation. It collects
        outputs from all modules that were actually used and formats them
        into a coherent response tailored to the request category.
        
        Args:
            traffic_request (TrafficRequest): The original processed request
            routing_decision (dict): Routing decision from Request Router
            module_outputs (dict): Dictionary of outputs from each module:
                - 'ann_priority': Output from ANN Priority Module
                - 'policy_validation': Output from Logic/KB Module
                - 'control_allocation': Output from CSP Scheduler
                - 'route': Output from Search & Navigation Module
        
        Returns:
            SystemResponse: Formatted final response
        
        Raises:
            InvalidRequestError: If inputs are invalid
        
        Example:
            >>> from src.modules.input_preprocessing import InputPreprocessingModule
            >>> from src.modules.request_router import RequestRouter
            >>> input_mod = InputPreprocessingModule()
            >>> router = RequestRouter()
            >>> raw = {
            ...     'request_id': 'REQ-001',
            ...     'vehicle_type': 'Ambulance',
            ...     'request_category': 'Emergency_Response_Request',
            ...     'current_location': 'Central_Junction',
            ...     'destination': 'City_Hospital',
            ...     'incident_severity': 'High',
            ...     'time_sensitivity': 'High',
            ...     'traffic_density': 9,
            ...     'priority_claim': 3
            ... }
            >>> request = input_mod.process_request(raw)
            >>> decision = router.route_request(request)
            >>> 
            >>> # (Assume other modules have run and produced outputs)
            >>> module_outputs = {
            ...     'ann_priority': {'priority_level': 'Critical', 'confidence': 0.95},
            ...     'policy_validation': {'status': 'Approved', 'reason': 'Emergency authorized'},
            ...     'control_allocation': {'plan_type': 'emergency', 'signals': [...]},
            ...     'route': {'path': ['Central_Junction', 'East_Market', 'City_Hospital'], 'cost': 6.0}
            ... }
            >>> 
            >>> response_module = FinalResponseModule()
            >>> final = response_module.generate_response(request, decision, module_outputs)
            >>> print(final.status)
            'Approved'
        """
        if not isinstance(traffic_request, TrafficRequest):
            raise InvalidRequestError(
                'traffic_request',
                "Expected TrafficRequest object"
            )
        
        # Create base response
        response = SystemResponse(
            traffic_request.request_id,
            traffic_request.request_category
        )
        
        request_category = traffic_request.request_category
        modules_used = routing_decision.get('modules_sequence', [])
        
        # Track which modules were used
        for module in modules_used:
            response.add_module_usage(module)
        
        # Extract outputs from each module (only if used)
        ann_output = module_outputs.get('ann_priority') if 'ANN_Priority' in modules_used else None
        policy_output = module_outputs.get('policy_validation') if 'Logic_KnowledgeBase' in modules_used else None
        csp_output = module_outputs.get('control_allocation') if 'CSP_Scheduler' in modules_used else None
        route_output = module_outputs.get('route') if 'Search_Navigation' in modules_used else None
        
        # Build response based on request category
        if request_category == RequestCategory.ROUTE_REQUEST:
            self._build_route_response(response, traffic_request, route_output)
        
        elif request_category == RequestCategory.POLICY_CHECK:
            self._build_policy_response(response, traffic_request, policy_output)
        
        elif request_category == RequestCategory.CONTROL_ALLOCATION_REQUEST:
            self._build_control_response(response, traffic_request,
                                          ann_output, policy_output, csp_output, route_output)
        
        elif request_category == RequestCategory.EMERGENCY_RESPONSE_REQUEST:
            self._build_emergency_response(response, traffic_request,
                                            ann_output, policy_output, csp_output, route_output)
        
        elif request_category == RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST:
            self._build_integrated_response(response, traffic_request,
                                             ann_output, policy_output, csp_output, route_output)
        
        return response
    
    # =====================================================================
    # FUNCTION 3: _build_route_response
    # =====================================================================
    def _build_route_response(self, response, request, route_output):
        """
        Build response for standard route requests.
        
        Includes only route information - no priority, policy, or control data.
        
        Args:
            response (SystemResponse): Response object to populate
            request (TrafficRequest): Original request
            route_output (dict): Output from Search module
        """
        if route_output and 'path' in route_output:
            response.set_route_info(
                route=route_output['path'],
                travel_time=route_output.get('cost', 0),
                cost=route_output.get('cost', 0)
            )
            
            template = self.response_templates[RequestCategory.ROUTE_REQUEST]
            message = template['message_pattern'].format(
                start=request.current_location,
                destination=request.destination
            )
            
            response.set_final_decision('Approved', message)
            
            # Add explanatory text
            response.explanatory_text = (
                f"Route from {request.current_location} to {request.destination} "
                f"via {' -> '.join(route_output['path'])}. "
                f"Estimated travel time: {route_output.get('cost', 0)} minutes."
            )
        else:
            response.set_final_decision('Error', 'Route computation failed')
    
    # =====================================================================
    # FUNCTION 4: _build_policy_response
    # =====================================================================
    def _build_policy_response(self, response, request, policy_output):
        """
        Build response for policy check requests.
        
        Includes only policy validation result - no route or control data.
        
        Args:
            response (SystemResponse): Response object to populate
            request (TrafficRequest): Original request
            policy_output (dict): Output from Logic/KB module
        """
        if policy_output:
            response.set_policy_result(
                status=policy_output.get('status', 'Unknown'),
                reason=policy_output.get('reason', '')
            )
            
            template = self.response_templates[RequestCategory.POLICY_CHECK]
            message = template['message_pattern'].format(
                status=policy_output.get('status', 'Unknown'),
                reason=policy_output.get('reason', 'No reason provided')
            )
            
            response.set_final_decision(
                policy_output.get('status', 'Unknown'),
                message
            )
            
            response.explanatory_text = (
                f"Policy check for {request.vehicle_type}: "
                f"{policy_output.get('status', 'Unknown')}. "
                f"{policy_output.get('reason', '')}"
            )
        else:
            response.set_final_decision('Error', 'Policy validation failed')
    
    # =====================================================================
    # FUNCTION 5: _build_control_response
    # =====================================================================
    def _build_control_response(self, response, request,
                                 ann_output, policy_output, csp_output, route_output):
        """
        Build response for control allocation requests.
        
        Includes priority, policy, control plan, and route.
        
        Args:
            response (SystemResponse): Response object to populate
            request (TrafficRequest): Original request
            ann_output (dict): ANN priority output
            policy_output (dict): Policy validation output
            csp_output (dict): CSP control plan output
            route_output (dict): Route output
        """
        # Priority (from ANN)
        if ann_output:
            response.set_priority(ann_output.get('priority_level', 'Unknown'))
        
        # Policy
        if policy_output:
            response.set_policy_result(
                status=policy_output.get('status', 'Unknown'),
                reason=policy_output.get('reason', '')
            )
        
        # Control plan (from CSP)
        if csp_output and csp_output.get('plan_type') != 'rejected':
            response.set_control_action({
                'plan_type': csp_output.get('plan_type'),
                'signals': csp_output.get('signals', []),
                'total_signals': csp_output.get('total_signals', 0)
            })
        elif csp_output and csp_output.get('plan_type') == 'rejected':
            response.set_control_action({
                'status': 'rejected',
                'reason': csp_output.get('reason', 'Control allocation rejected')
            })
        
        # Route
        if route_output and 'path' in route_output:
            response.set_route_info(
                route=route_output['path'],
                travel_time=route_output.get('cost', 0),
                cost=route_output.get('cost', 0)
            )
        
        # Final decision based on policy
        if policy_output:
            status = policy_output.get('status', 'Rejected')
            template = self.response_templates[RequestCategory.CONTROL_ALLOCATION_REQUEST]
            message = template['message_pattern'].format(
                status=status,
                reason=policy_output.get('reason', 'No reason provided')
            )
            response.set_final_decision(status, message)
            
            # Build explanatory text
            parts = []
            if ann_output:
                parts.append(f"Priority level: {ann_output.get('priority_level', 'Unknown')}")
            if csp_output and csp_output.get('plan_type') != 'rejected':
                parts.append(f"Control plan: {csp_output.get('total_signals', 0)} signals assigned")
            if route_output:
                parts.append(f"Route: {' -> '.join(route_output['path'])}")
            
            response.explanatory_text = ". ".join(parts) + "."
    
    # =====================================================================
    # FUNCTION 6: _build_emergency_response
    # =====================================================================
    def _build_emergency_response(self, response, request,
                                   ann_output, policy_output, csp_output, route_output):
        """
        Build response for emergency response requests.
        
        Comprehensive response with priority, validation, corridor, and route.
        
        Args:
            response (SystemResponse): Response object to populate
            request (TrafficRequest): Original request
            ann_output (dict): ANN priority output
            policy_output (dict): Policy validation output
            csp_output (dict): CSP control plan output
            route_output (dict): Route output
        """
        # Priority (critical for emergency)
        if ann_output:
            response.set_priority(ann_output.get('priority_level', 'Critical'))
        else:
            response.set_priority('Critical')  # Default for emergency
        
        # Policy validation
        if policy_output:
            response.set_policy_result(
                status=policy_output.get('status', 'Unknown'),
                reason=policy_output.get('reason', '')
            )
        
        # Emergency control plan
        if csp_output and csp_output.get('plan_type') == 'emergency':
            signals = csp_output.get('signals', [])
            emergency_signals = [s for s in signals 
                                if s.get('phase') in ['PhaseA', 'PhaseB']]
            
            response.set_control_action({
                'plan_type': 'emergency',
                'emergency_corridor': True,
                'signals': signals,
                'priority_signals': [s['signal_id'] for s in emergency_signals],
                'total_signals': len(signals)
            })
        elif csp_output:
            response.set_control_action(csp_output)
        
        # Emergency route
        if route_output and 'path' in route_output:
            response.set_route_info(
                route=route_output['path'],
                travel_time=route_output.get('cost', 0),
                cost=route_output.get('cost', 0)
            )
        
        # Final decision
        if policy_output and policy_output.get('status') == 'Approved':
            template = self.response_templates[RequestCategory.EMERGENCY_RESPONSE_REQUEST]
            message = template['message_pattern'].format(
                priority=response.predicted_priority or 'Critical',
                status='APPROVED',
                reason='Emergency corridor allocated with signal priority'
            )
            response.set_final_decision('Approved', message)
            
            # Detailed explanatory text for emergency
            parts = [
                f"EMERGENCY RESPONSE COORDINATED",
                f"Vehicle: {request.vehicle_type}",
                f"Priority: {response.predicted_priority or 'Critical'}",
                f"Route: {' -> '.join(route_output['path'])}",
                f"Estimated arrival: {route_output.get('cost', 0)} minutes",
                f"Signal corridor: ACTIVE"
            ]
            response.explanatory_text = "\n".join(parts)
            
        else:
            status = policy_output.get('status', 'Rejected') if policy_output else 'Rejected'
            response.set_final_decision(status, 
                f"Emergency response {status}: {policy_output.get('reason', 'Authorization failed')}")
    
    # =====================================================================
    # FUNCTION 7: _build_integrated_response
    # =====================================================================
    def _build_integrated_response(self, response, request,
                                    ann_output, policy_output, csp_output, route_output):
        """
        Build response for integrated city service requests.
        
        Most comprehensive response with all available information.
        
        Args:
            response (SystemResponse): Response object to populate
            request (TrafficRequest): Original request
            ann_output (dict): ANN priority output
            policy_output (dict): Policy validation output
            csp_output (dict): CSP control plan output
            route_output (dict): Route output
        """
        # All available information
        if ann_output:
            response.set_priority(ann_output.get('priority_level', 'Unknown'))
        
        if policy_output:
            response.set_policy_result(
                status=policy_output.get('status', 'Unknown'),
                reason=policy_output.get('reason', '')
            )
        
        if csp_output:
            response.set_control_action({
                'plan_type': csp_output.get('plan_type'),
                'signals': csp_output.get('signals', []),
                'total_signals': csp_output.get('total_signals', 0),
                'nodes_explored': csp_output.get('nodes_explored', 0)
            })
        
        if route_output and 'path' in route_output:
            response.set_route_info(
                route=route_output['path'],
                travel_time=route_output.get('cost', 0),
                cost=route_output.get('cost', 0)
            )
        
        # Final decision
        if policy_output:
            status = policy_output.get('status', 'Rejected')
            template = self.response_templates[RequestCategory.INTEGRATED_CITY_SERVICE_REQUEST]
            message = template['message_pattern'].format(
                status=status,
                reason=policy_output.get('reason', 'No reason provided')
            )
            response.set_final_decision(status, message)
            
            # Comprehensive summary
            summary_parts = [
                f"INTEGRATED CITY SERVICE RESPONSE",
                f"{'=' * 50}",
                f"Request ID: {request.request_id}",
                f"Vehicle: {request.vehicle_type} ({'EMERGENCY' if request.is_emergency else 'Civilian'})",
                f"Priority: {response.predicted_priority or 'N/A'}",
                f"Policy Status: {response.policy_status or 'N/A'}",
                f"Control Plan: {csp_output.get('plan_type', 'N/A') if csp_output else 'N/A'}",
                f"Route: {' -> '.join(route_output['path']) if route_output else 'N/A'}",
                f"Travel Time: {route_output.get('cost', 'N/A') if route_output else 'N/A'} minutes",
                f"{'=' * 50}",
                f"Decision: {status}",
                f"Modules Used: {', '.join(response.modules_used)}"
            ]
            response.explanatory_text = "\n".join(summary_parts)
    
    # =====================================================================
    # FUNCTION 8: format_for_display
    # =====================================================================
    def format_for_display(self, response):
        """
        Format a SystemResponse for console display.
        
        Creates a human-readable, visually structured output suitable
        for operators and system users.
        
        Args:
            response (SystemResponse): Response to format
        
        Returns:
            str: Formatted display string
        
        Example:
            >>> display_text = response_module.format_for_display(final_response)
            >>> print(display_text)
        """
        lines = [
            "=" * 70,
            "SMART CITY TRAFFIC AI SYSTEM - FINAL RESPONSE",
            "=" * 70,
            f"Request ID:       {response.request_id}",
            f"Request Category: {response.request_category}",
            f"System Status:    {response.status}",
            "",
            "MODULES ACTIVATED:",
            "-" * 70
        ]
        
        for i, module in enumerate(response.modules_used, 1):
            lines.append(f"  {i}. {module}")
        
        # Priority (only if present)
        if response.predicted_priority:
            lines.extend([
                "",
                "PRIORITY ASSESSMENT:",
                "-" * 70,
                f"  Predicted Priority: {response.predicted_priority}"
            ])
        
        # Policy (only if present)
        if response.policy_status:
            lines.extend([
                "",
                "POLICY VALIDATION:",
                "-" * 70,
                f"  Status: {response.policy_status}",
                f"  Reason: {response.policy_reason or 'N/A'}"
            ])
        
        # Route (only if present)
        if response.recommended_route:
            lines.extend([
                "",
                "RECOMMENDED ROUTE:",
                "-" * 70
            ])
            for i, node in enumerate(response.recommended_route):
                arrow = " → " if i < len(response.recommended_route) - 1 else " (DESTINATION)"
                lines.append(f"  [{i}] {node}{arrow}")
            
            lines.append(f"")
            lines.append(f"  Estimated Travel Time: {response.estimated_travel_time} minutes")
            if response.estimated_cost is not None:
                lines.append(f"  Total Route Cost: {response.estimated_cost}")
        
        # Control Action (only if present)
        if response.control_action:
            lines.extend([
                "",
                "CONTROL ALLOCATION:",
                "-" * 70
            ])
            
            if isinstance(response.control_action, dict):
                for key, value in response.control_action.items():
                    if key == 'signals' and isinstance(value, list):
                        lines.append(f"  Signals Assigned: {len(value)}")
                        for sig in value:
                            if isinstance(sig, dict):
                                lines.append(f"    - {sig.get('signal_id', 'Unknown')}: {sig.get('phase', 'Unknown')}")
                    else:
                        lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {response.control_action}")
        
        # Decision Message
        lines.extend([
            "",
            "FINAL DECISION:",
            "-" * 70,
            f"  {response.decision_message or 'No decision message'}",
            ""
        ])
        
        # Explanatory Text
        if response.explanatory_text:
            lines.extend([
                "DETAILED EXPLANATION:",
                "-" * 70,
                response.explanatory_text,
                ""
            ])
        
        lines.append("=" * 70)
        
        return '\n'.join(lines)
    
    # =====================================================================
    # FUNCTION 9: format_for_operator
    # =====================================================================
    def format_for_operator(self, response, role='Traffic Control Operator'):
        """
        Format response tailored to specific operational roles.
        
        Different roles need different information emphasis:
        - Civilian Driver: Route and time
        - Emergency Vehicle Unit: Priority and corridor status
        - Traffic Control Operator: Signal assignments and coordination
        - Emergency Coordination Center: Full picture with validation
        
        Args:
            response (SystemResponse): Response to format
            role (str): Operational role
        
        Returns:
            str: Role-tailored display string
        """
        role_focus = {
            'Civilian Driver': ['route', 'time'],
            'Emergency Vehicle Unit': ['priority', 'route', 'corridor'],
            'Traffic Control Operator': ['control', 'coordination'],
            'Emergency Coordination Center': ['all']
        }
        
        focus = role_focus.get(role, ['all'])
        
        lines = [
            "=" * 70,
            f"RESPONSE FOR: {role}",
            "=" * 70,
            f"Request: {response.request_id}",
            f"Status: {response.status}",
            ""
        ]
        
        # Include relevant sections based on role
        if 'all' in focus or 'priority' in focus:
            if response.predicted_priority:
                lines.extend([
                    f"PRIORITY: {response.predicted_priority}",
                    ""
                ])
        
        if 'all' in focus or 'route' in focus or 'time' in focus:
            if response.recommended_route:
                lines.append(f"ROUTE: {' → '.join(response.recommended_route)}")
                lines.append(f"TIME: {response.estimated_travel_time} minutes")
                lines.append("")
        
        if 'all' in focus or 'control' in focus or 'coordination' in focus:
            if response.control_action:
                lines.append("CONTROL ACTIONS:")
                if isinstance(response.control_action, dict):
                    signals = response.control_action.get('signals', [])
                    for sig in signals:
                        if isinstance(sig, dict):
                            lines.append(f"  {sig.get('signal_id')}: {sig.get('phase')}")
                lines.append("")
        
        if 'all' in focus or 'corridor' in focus:
            if response.control_action and isinstance(response.control_action, dict):
                if response.control_action.get('emergency_corridor'):
                    lines.append("EMERGENCY CORRIDOR: ACTIVE")
                    priority_signals = response.control_action.get('priority_signals', [])
                    lines.append(f"PRIORITY SIGNALS: {', '.join(priority_signals)}")
                    lines.append("")
        
        lines.extend([
            f"DECISION: {response.decision_message}",
            "=" * 70
        ])
        
        return '\n'.join(lines)
    
    # =====================================================================
    # FUNCTION 10: generate_minimal_response
    # =====================================================================
    def generate_minimal_response(self, request_id, status, message):
        """
        Generate a minimal response for simple cases or errors.
        
        Args:
            request_id (str): Request identifier
            status (str): Status code
            message (str): Brief message
        
        Returns:
            dict: Minimal response dictionary
        """
        return {
            'request_id': request_id,
            'status': status,
            'message': message,
            'timestamp': None,  # Could add actual timestamp
            'modules_used': []
        }


# =============================================================================
# Standalone testing functionality
# =============================================================================
if __name__ == "__main__":
    """
    Standalone test for the Final Response Module.
    Run this file directly to test module functionality.
    """
    import sys
    import os
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.modules.input_preprocessing import InputPreprocessingModule
    from src.modules.request_router import RequestRouter
    from src.modules.ann_priority import ANNPriorityModule
    from src.modules.logic_knowledge_base import LogicKnowledgeBase
    from src.modules.csp_scheduler import CSPScheduler
    from src.modules.search_navigation import SearchNavigationModule
    
    print("=" * 70)
    print("FINAL RESPONSE MODULE - STANDALONE TEST")
    print("=" * 70)
    
    # Create all modules
    input_module = InputPreprocessingModule()
    router = RequestRouter()
    ann = ANNPriorityModule()
    kb = LogicKnowledgeBase()
    csp = CSPScheduler()
    search = SearchNavigationModule()
    response_module = FinalResponseModule()
    
    # Train ANN
    print("\n" + "=" * 70)
    print("TRAINING ANN MODELS")
    print("=" * 70)
    ann.load_training_data()
    ann.train_models(binary_epochs=50, mlp_epochs=100, verbose=False)
    print("✓ ANN trained")
    
    # Test Case 1: Route Request (Civilian)
    print("\n" + "=" * 70)
    print("TEST CASE 1: Route Request (Civilian)")
    print("Expected: Simple route response, no priority/control fields")
    print("=" * 70)
    
    try:
        raw_1 = {
            'request_id': 'REQ-001',
            'vehicle_type': 'Civilian',
            'request_category': 'Route_Request',
            'current_location': 'North_Station',
            'destination': 'Airport_Road',
            'traffic_density': 4
        }
        
        request_1 = input_module.process_request(raw_1)
        decision_1 = router.route_request(request_1)
        route_1 = search.find_route(request_1)
        
        module_outputs_1 = {
            'route': route_1
        }
        
        final_1 = response_module.generate_response(
            request_1, decision_1, module_outputs_1
        )
        
        print(response_module.format_for_display(final_1))
        
        # Verify selective inclusion
        assert final_1.predicted_priority is None, "Route request should not have priority"
        assert final_1.control_action is None, "Route request should not have control"
        assert final_1.recommended_route is not None, "Route request should have route"
        assert final_1.status == 'Approved'
        print("\n✓ Test Case 1 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 2: Policy Check (Emergency)
    print("\n" + "=" * 70)
    print("TEST CASE 2: Policy Check (Emergency Vehicle)")
    print("Expected: Policy validation only, no route")
    print("=" * 70)
    
    try:
        raw_2 = {
            'request_id': 'REQ-002',
            'vehicle_type': 'Ambulance',
            'request_category': 'Policy_Check',
            'current_location': 'Central_Junction',
            'destination': 'City_Hospital',
            'control_zone': 'Central_Junction'
        }
        
        request_2 = input_module.process_request(raw_2)
        decision_2 = router.route_request(request_2)
        policy_2 = kb.validate_policy(request_2)
        
        module_outputs_2 = {
            'policy_validation': policy_2
        }
        
        final_2 = response_module.generate_response(
            request_2, decision_2, module_outputs_2
        )
        
        print(response_module.format_for_display(final_2))
        
        assert final_2.policy_status is not None
        assert final_2.recommended_route is None, "Policy check should not have route"
        print("\n✓ Test Case 2 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 3: Emergency Response (Full Pipeline)
    print("\n" + "=" * 70)
    print("TEST CASE 3: Emergency Response (Full Pipeline)")
    print("Expected: Complete response with all fields")
    print("=" * 70)
    
    try:
        raw_3 = {
            'request_id': 'REQ-003',
            'vehicle_type': 'Ambulance',
            'request_category': 'Emergency_Response_Request',
            'current_location': 'Central_Junction',
            'destination': 'City_Hospital',
            'incident_severity': 'High',
            'time_sensitivity': 'High',
            'traffic_density': 9,
            'priority_claim': 3,
            'control_zone': 'S1_Central_Junction'
        }
        
        request_3 = input_module.process_request(raw_3)
        decision_3 = router.route_request(request_3)
        
        # Run full pipeline
        priority_3 = ann.predict_priority(request_3.normalized_features, 'multiclass')
        policy_3 = kb.validate_policy(request_3, priority_3['priority_level'])
        plan_3 = csp.allocate_control(request_3, policy_3)
        route_3 = search.find_route(request_3, plan_3)
        
        module_outputs_3 = {
            'ann_priority': priority_3,
            'policy_validation': policy_3,
            'control_allocation': plan_3,
            'route': route_3
        }
        
        final_3 = response_module.generate_response(
            request_3, decision_3, module_outputs_3
        )
        
        print(response_module.format_for_display(final_3))
        
        assert final_3.predicted_priority is not None
        assert final_3.policy_status is not None
        assert final_3.control_action is not None
        assert final_3.recommended_route is not None
        assert final_3.status == 'Approved'
        print("\n✓ Test Case 3 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 4: Integrated City Service
    print("\n" + "=" * 70)
    print("TEST CASE 4: Integrated City Service")
    print("Expected: Most comprehensive response")
    print("=" * 70)
    
    try:
        raw_4 = {
            'request_id': 'REQ-004',
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
        
        request_4 = input_module.process_request(raw_4)
        decision_4 = router.route_request(request_4)
        
        # Run full pipeline
        priority_4 = ann.predict_priority(request_4.normalized_features, 'multiclass')
        policy_4 = kb.validate_policy(request_4, priority_4['priority_level'])
        plan_4 = csp.allocate_control(request_4, policy_4)
        route_4 = search.find_route(request_4, plan_4)
        
        module_outputs_4 = {
            'ann_priority': priority_4,
            'policy_validation': policy_4,
            'control_allocation': plan_4,
            'route': route_4
        }
        
        final_4 = response_module.generate_response(
            request_4, decision_4, module_outputs_4
        )
        
        print(response_module.format_for_display(final_4))
        
        assert len(final_4.modules_used) == 5, "Should use all 5 modules"
        assert 'Final_Response' in final_4.modules_used
        print("\n✓ Test Case 4 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 5: Role-Specific Formatting
    print("\n" + "=" * 70)
    print("TEST CASE 5: Role-Specific Formatting")
    print("=" * 70)
    
    try:
        roles = [
            'Civilian Driver',
            'Emergency Vehicle Unit',
            'Traffic Control Operator',
            'Emergency Coordination Center'
        ]
        
        for role in roles:
            print(f"\n--- Format for {role} ---")
            formatted = response_module.format_for_operator(final_3, role)
            print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
        
        print("\n✓ Test Case 5 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 6: Rejected Request Response
    print("\n" + "=" * 70)
    print("TEST CASE 6: Rejected Request Response")
    print("Expected: Clear rejection with reason")
    print("=" * 70)
    
    try:
        raw_6 = {
            'request_id': 'REQ-006',
            'vehicle_type': 'Civilian',
            'request_category': 'Control_Allocation_Request',
            'current_location': 'Central_Junction',
            'destination': 'City_Hospital',
            'control_zone': 'S1_Central_Junction'
        }
        
        request_6 = input_module.process_request(raw_6)
        decision_6 = router.route_request(request_6)
        policy_6 = kb.validate_policy(request_6)
        
        module_outputs_6 = {
            'policy_validation': policy_6
        }
        
        final_6 = response_module.generate_response(
            request_6, decision_6, module_outputs_6
        )
        
        print(response_module.format_for_display(final_6))
        
        assert final_6.status == 'Rejected'
        assert 'civilian' in final_6.decision_message.lower() or 'not authorized' in final_6.decision_message.lower()
        print("\n✓ Test Case 6 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 7: Minimal Response
    print("\n" + "=" * 70)
    print("TEST CASE 7: Minimal Response")
    print("=" * 70)
    
    try:
        minimal = response_module.generate_minimal_response(
            'REQ-007', 'Error', 'System temporarily unavailable'
        )
        print(f"Minimal Response: {minimal}")
        assert minimal['status'] == 'Error'
        assert minimal['message'] == 'System temporarily unavailable'
        print("\n✓ Test Case 7 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 7 FAILED: {e}")
    
    # Test Case 8: to_dict() Serialization
    print("\n" + "=" * 70)
    print("TEST CASE 8: Response Serialization")
    print("=" * 70)
    
    try:
        response_dict = final_3.to_dict()
        print(f"Serialized keys: {list(response_dict.keys())}")
        
        # Verify selective serialization
        assert 'recommended_route' in response_dict
        assert 'predicted_priority' in response_dict
        assert 'policy_status' in response_dict
        assert 'control_action' in response_dict
        assert 'decision_message' in response_dict
        print("\n✓ Test Case 8 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 8 FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)