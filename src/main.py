"""
main.py
Main Entry Point for the Smart City Traffic & Emergency Response AI System.

This module provides the unified interface for the entire system. It:
1. Initializes all 7 modules
2. Provides an interactive menu for submitting traffic requests
3. Orchestrates the complete processing pipeline
4. Displays final responses

Usage:
    python -m src.main              # Interactive mode
    python -m src.main --demo       # Run all demo scenarios
    python -m src.main --test       # Run system tests

Author: [Your Name]
Group: [Your Group]
Course: AL-2002 Artificial Intelligence Lab
"""


import sys
import argparse

from src.modules.input_preprocessing import InputPreprocessingModule
from src.modules.request_router import RequestRouter
from src.modules.ann_priority import ANNPriorityModule
from src.modules.logic_knowledge_base import LogicKnowledgeBase
from src.modules.csp_scheduler import CSPScheduler
from src.modules.search_navigation import SearchNavigationModule
from src.modules.final_response import FinalResponseModule

from src.models.traffic_request import TrafficRequest
from src.utils.exceptions import SmartCityTrafficError


class SmartCityTrafficSystem:
    """
    Main System Controller for the Smart City Traffic AI System.
    
    This class initializes and coordinates all 7 modules, providing
    a unified interface for processing traffic requests.
    
    Attributes:
        input_module (InputPreprocessingModule): Module 1
        router (RequestRouter): Module 2
        ann_module (ANNPriorityModule): Module 3
        kb (LogicKnowledgeBase): Module 4
        csp (CSPScheduler): Module 5
        search (SearchNavigationModule): Module 6
        response_module (FinalResponseModule): Module 7
        is_initialized (bool): Whether system is ready
    """
    
    def __init__(self):
        """
        Initialize the Smart City Traffic AI System.
        
        Creates instances of all 7 modules and prepares the system
        for processing requests.
        """
        self.input_module = None
        self.router = None
        self.ann_module = None
        self.kb = None
        self.csp = None
        self.search = None
        self.response_module = None
        self.is_initialized = False
    
    # =====================================================================
    # FUNCTION 1: initialize
    # =====================================================================
    def initialize(self, train_ann=True, verbose=True):
        """
        Initialize all system modules.
        
        This function creates instances of all modules, loads data,
        and trains the ANN models. Must be called before processing
        any requests.
        
        Args:
            train_ann (bool): Whether to train ANN models. Default: True
            verbose (bool): Print initialization progress. Default: True
        
        Returns:
            bool: True if initialization successful, False otherwise
        
        Example:
            >>> system = SmartCityTrafficSystem()
            >>> system.initialize()
            Initializing Smart City Traffic AI System...
            [OK] Input & Preprocessing Module
            [OK] Request Router
            ...
            System ready!
        """
        if verbose:
            print("=" * 70)
            print("INITIALIZING SMART CITY TRAFFIC AI SYSTEM")
            print("=" * 70)
        
        try:
            # Module 1: Input & Preprocessing
            if verbose:
                print("\n[1/7] Input & Preprocessing Module...")
            self.input_module = InputPreprocessingModule()
            if verbose:
                print("      ✓ Ready")
            
            # Module 2: Request Router
            if verbose:
                print("\n[2/7] Request Router...")
            self.router = RequestRouter()
            if verbose:
                print("      ✓ Ready")
            
            # Module 3: ANN Priority
            if verbose:
                print("\n[3/7] ANN Priority Prediction Module...")
            self.ann_module = ANNPriorityModule()
            self.ann_module.load_training_data()
            if train_ann:
                if verbose:
                    print("      Training models...")
                self.ann_module.train_models(binary_epochs=100, mlp_epochs=200, verbose=False)
            if verbose:
                print("      ✓ Ready")
            
            # Module 4: Logic/Knowledge Base
            if verbose:
                print("\n[4/7] Logic/Knowledge Base Module...")
            self.kb = LogicKnowledgeBase()
            if verbose:
                print("      ✓ Ready")
            
            # Module 5: CSP Scheduler
            if verbose:
                print("\n[5/7] CSP Scheduler Module...")
            self.csp = CSPScheduler()
            if verbose:
                print("      ✓ Ready")
            
            # Module 6: Search & Navigation
            if verbose:
                print("\n[6/7] Search & Navigation Module...")
            self.search = SearchNavigationModule()
            if verbose:
                print("      ✓ Ready")
            
            # Module 7: Final Response
            if verbose:
                print("\n[7/7] Final Response Module...")
            self.response_module = FinalResponseModule()
            if verbose:
                print("      ✓ Ready")
            
            self.is_initialized = True
            
            if verbose:
                print("\n" + "=" * 70)
                print("SYSTEM INITIALIZATION COMPLETE")
                print("All 7 modules loaded and ready.")
                print("=" * 70)
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"\n✗ INITIALIZATION FAILED: {e}")
            return False
    
    # =====================================================================
    # FUNCTION 2: process_request
    # =====================================================================
    def process_request(self, raw_input, verbose=True):
        """
        Process a complete traffic request through all modules.
        
        This is the main system pipeline that orchestrates the flow:
        1. Input preprocessing (validation, normalization)
        2. Request routing (determine module sequence)
        3. Execute required modules in sequence
        4. Generate final response
        
        Args:
            raw_input (dict): Raw traffic request dictionary
            verbose (bool): Print processing details
        
        Returns:
            dict: Final system response
        
        Example:
            >>> system = SmartCityTrafficSystem()
            >>> system.initialize()
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
            >>> result = system.process_request(raw)
            >>> print(result['status'])
            'Approved'
        """
        if not self.is_initialized:
            print("ERROR: System not initialized. Call initialize() first.")
            return None
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"PROCESSING REQUEST: {raw_input.get('request_id', 'UNKNOWN')}")
            print("=" * 70)
        
        try:
            # =====================================================================
            # STEP 1: Input & Preprocessing (Module 1)
            # =====================================================================
            if verbose:
                print("\n[STEP 1] Input & Preprocessing")
                print("-" * 70)
            
            traffic_request = self.input_module.process_request(raw_input)
            
            if verbose:
                print(f"  ✓ Request validated and normalized")
                print(f"  ✓ Feature vector prepared: {traffic_request.normalized_features}")
            
            # =====================================================================
            # STEP 2: Request Routing (Module 2)
            # =====================================================================
            if verbose:
                print("\n[STEP 2] Request Routing")
                print("-" * 70)
            
            routing_decision = self.router.route_request(traffic_request)
            modules_sequence = routing_decision['modules_sequence']
            
            if verbose:
                print(f"  ✓ Category: {routing_decision['request_category']}")
                print(f"  ✓ Pipeline: {' -> '.join(modules_sequence)}")
                print(f"  ✓ Complexity: {routing_decision['estimated_complexity']}")
            
            # =====================================================================
            # STEP 3: Execute AI Modules
            # =====================================================================
            module_outputs = {}
            
            for module_name in modules_sequence:
                # -----------------------------------------------------------------
                # Module 3: ANN Priority Prediction
                # -----------------------------------------------------------------
                if module_name == 'ANN_Priority':
                    if verbose:
                        print("\n[STEP 3] ANN Priority Prediction")
                        print("-" * 70)
                    
                    priority_result = self.ann_module.predict_priority(
                        traffic_request.normalized_features,
                        mode='multiclass'
                    )
                    module_outputs['ann_priority'] = priority_result
                    
                    if verbose:
                        print(f"  ✓ Priority: {priority_result['priority_level']}")
                        print(f"  ✓ Confidence: {priority_result['confidence']:.4f}")
                
                # -----------------------------------------------------------------
                # Module 4: Logic/Knowledge Base
                # -----------------------------------------------------------------
                elif module_name == 'Logic_KnowledgeBase':
                    if verbose:
                        print("\n[STEP 4] Logic/Knowledge Base Validation")
                        print("-" * 70)
                    
                    # Use ANN priority if available, otherwise None
                    predicted_priority = None
                    if 'ann_priority' in module_outputs:
                        predicted_priority = module_outputs['ann_priority']['priority_level']
                    
                    policy_result = self.kb.validate_policy(
                        traffic_request,
                        predicted_priority=predicted_priority
                    )
                    module_outputs['policy_validation'] = policy_result
                    
                    if verbose:
                        print(f"  ✓ Status: {policy_result['status']}")
                        print(f"  ✓ Priority: {policy_result['priority_level']}")
                        print(f"  ✓ Reason: {policy_result['reason']}")
                
                # -----------------------------------------------------------------
                # Module 5: CSP Scheduler
                # -----------------------------------------------------------------
                elif module_name == 'CSP_Scheduler':
                    if verbose:
                        print("\n[STEP 5] CSP Control Allocation")
                        print("-" * 70)
                    
                    policy_result = module_outputs.get('policy_validation')
                    control_plan = self.csp.allocate_control(
                        traffic_request,
                        policy_result=policy_result
                    )
                    module_outputs['control_allocation'] = control_plan
                    
                    if verbose:
                        print(f"  ✓ Plan Type: {control_plan.get('plan_type', 'N/A')}")
                        if control_plan.get('signals'):
                            print(f"  ✓ Signals: {len(control_plan['signals'])} assigned")
                        if control_plan.get('nodes_explored'):
                            print(f"  ✓ CSP Nodes Explored: {control_plan['nodes_explored']}")
                
                # -----------------------------------------------------------------
                # Module 6: Search & Navigation
                # -----------------------------------------------------------------
                elif module_name == 'Search_Navigation':
                    if verbose:
                        print("\n[STEP 6] Search & Navigation")
                        print("-" * 70)
                    
                    control_plan = module_outputs.get('control_allocation')
                    route_result = self.search.find_route(
                        traffic_request,
                        control_plan=control_plan if control_plan and control_plan.get('plan_type') != 'rejected' else None
                    )
                    module_outputs['route'] = route_result
                    
                    if verbose:
                        if route_result.get('path'):
                            print(f"  ✓ Route: {' -> '.join(route_result['path'])}")
                            print(f"  ✓ Cost: {route_result['cost']} {route_result.get('cost_type', '')}")
                            print(f"  ✓ Algorithm: {route_result['algorithm']}")
                        else:
                            print(f"  ✗ No valid route found")
                
                # -----------------------------------------------------------------
                # Module 7: Final Response
                # -----------------------------------------------------------------
                elif module_name == 'Final_Response':
                    if verbose:
                        print("\n[STEP 7] Final Response Generation")
                        print("-" * 70)
                    
                    # This is handled after the loop
                    pass
            
            # =====================================================================
            # STEP 4: Generate Final Response (Module 7)
            # =====================================================================
            if verbose:
                print("\n[FINAL] Response Aggregation")
                print("-" * 70)
            
            final_response = self.response_module.generate_response(
                traffic_request,
                routing_decision,
                module_outputs
            )
            
            if verbose:
                print(f"  ✓ Response generated")
                print(f"  ✓ Status: {final_response.status}")
                print(f"  ✓ Modules used: {len(final_response.modules_used)}")
            
            # =====================================================================
            # Display Final Output
            # =====================================================================
            if verbose:
                print("\n" + "=" * 70)
                display_text = self.response_module.format_for_display(final_response)
                print(display_text)
            
            return final_response.to_dict()
            
        except SmartCityTrafficError as e:
            if verbose:
                print(f"\n✗ SYSTEM ERROR: {e}")
            return {
                'request_id': raw_input.get('request_id', 'UNKNOWN'),
                'status': 'Error',
                'error': str(e),
                'error_type': type(e).__name__
            }
        except Exception as e:
            if verbose:
                print(f"\n✗ UNEXPECTED ERROR: {e}")
                import traceback
                traceback.print_exc()
            return {
                'request_id': raw_input.get('request_id', 'UNKNOWN'),
                'status': 'System Error',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    # =====================================================================
    # FUNCTION 3: run_demo_scenarios
    # =====================================================================
    def run_demo_scenarios(self):
        """
        Run all 5 request category scenarios for demonstration.
        
        This function executes pre-defined scenarios for each request
        category, showing the complete system capabilities.
        
        Returns:
            list: Results from all scenarios
        """
        print("\n" + "=" * 70)
        print("RUNNING ALL DEMO SCENARIOS")
        print("=" * 70)
        
        scenarios = [
            {
                'name': 'Standard Route Request (Civilian)',
                'request': {
                    'request_id': 'DEMO-001',
                    'vehicle_type': 'Civilian',
                    'request_category': 'Route_Request',
                    'current_location': 'North_Station',
                    'destination': 'Airport_Road',
                    'traffic_density': 4,
                    'description_note': 'Daily commute'
                }
            },
            {
                'name': 'Policy Check (Emergency Vehicle)',
                'request': {
                    'request_id': 'DEMO-002',
                    'vehicle_type': 'Ambulance',
                    'request_category': 'Policy_Check',
                    'current_location': 'Central_Junction',
                    'destination': 'City_Hospital',
                    'control_zone': 'Central_Junction',
                    'incident_severity': 'High'
                }
            },
            {
                'name': 'Control Allocation (Police)',
                'request': {
                    'request_id': 'DEMO-003',
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
            },
            {
                'name': 'Emergency Response (Ambulance)',
                'request': {
                    'request_id': 'DEMO-004',
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
            },
            {
                'name': 'Integrated City Service (Fire Truck)',
                'request': {
                    'request_id': 'DEMO-005',
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
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'#' * 70}")
            print(f"SCENARIO {i}: {scenario['name']}")
            print(f"{'#' * 70}")
            
            result = self.process_request(scenario['request'], verbose=True)
            results.append({
                'scenario': scenario['name'],
                'result': result
            })
            
            input("\nPress Enter to continue to next scenario...")
        
        print("\n" + "=" * 70)
        print("ALL DEMO SCENARIOS COMPLETED")
        print("=" * 70)
        
        # Summary
        print("\nSUMMARY:")
        for r in results:
            status = r['result'].get('status', 'Unknown') if r['result'] else 'Error'
            symbol = "✓" if status == 'Approved' else "✗" if status == 'Rejected' else "?"
            print(f"  {symbol} {r['scenario']}: {status}")
        
        return results
    
    # =====================================================================
    # FUNCTION 4: interactive_mode
    # =====================================================================
    def interactive_mode(self):
        """
        Run interactive command-line interface for the system.
        
        Provides a menu-driven interface for users to submit traffic
        requests and view responses.
        """
        print("\n" + "=" * 70)
        print("SMART CITY TRAFFIC AI SYSTEM - INTERACTIVE MODE")
        print("=" * 70)
        print("\nAvailable request categories:")
        print("  1. Route_Request")
        print("  2. Policy_Check")
        print("  3. Control_Allocation_Request")
        print("  4. Emergency_Response_Request")
        print("  5. Integrated_City_Service_Request")
        print("  0. Exit")
        
        while True:
            print("\n" + "-" * 70)
            choice = input("Enter request category (0-5): ").strip()
            
            if choice == '0':
                print("\nExiting system. Goodbye!")
                break
            
            categories = {
                '1': 'Route_Request',
                '2': 'Policy_Check',
                '3': 'Control_Allocation_Request',
                '4': 'Emergency_Response_Request',
                '5': 'Integrated_City_Service_Request'
            }
            
            if choice not in categories:
                print("Invalid choice. Please enter 0-5.")
                continue
            
            category = categories[choice]
            
            # Collect request details
            print(f"\n--- Enter Request Details for {category} ---")
            
            request_id = input("Request ID (e.g., REQ-001): ").strip()
            vehicle_type = input("Vehicle Type (Civilian/Ambulance/Fire_Truck/Police): ").strip()
            current = input("Current Location: ").strip()
            destination = input("Destination: ").strip()
            
            raw = {
                'request_id': request_id,
                'vehicle_type': vehicle_type,
                'request_category': category,
                'current_location': current,
                'destination': destination
            }
            
            # Optional fields
            if category in ['Control_Allocation_Request', 'Emergency_Response_Request', 
                           'Integrated_City_Service_Request']:
                severity = input("Incident Severity (None/Low/Medium/High) [None]: ").strip()
                if severity:
                    raw['incident_severity'] = severity
                
                time_sens = input("Time Sensitivity (Normal/High) [Normal]: ").strip()
                if time_sens:
                    raw['time_sensitivity'] = time_sens
                
                density = input("Traffic Density (0-10) [0]: ").strip()
                if density:
                    raw['traffic_density'] = int(density)
                
                priority = input("Priority Claim (0-3) [0]: ").strip()
                if priority:
                    raw['priority_claim'] = int(priority)
                
                zone = input("Control Zone (or leave empty): ").strip()
                if zone:
                    raw['control_zone'] = zone
            
            # Process the request
            result = self.process_request(raw, verbose=True)
            
            if result:
                print(f"\n{'=' * 70}")
                print("REQUEST PROCESSING COMPLETE")
                print(f"Final Status: {result.get('status', 'Unknown')}")
                print(f"{'=' * 70}")


# =====================================================================
# Main execution
# =====================================================================
def main():
    """
    Main function to parse arguments and run the system.
    """
    parser = argparse.ArgumentParser(
        description='Smart City Traffic & Emergency Response AI System'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run all demo scenarios'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run system tests'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimize output during initialization'
    )
    
    args = parser.parse_args()
    
    # Create and initialize system
    system = SmartCityTrafficSystem()
    
    if not system.initialize(verbose=not args.quiet):
        print("System initialization failed. Exiting.")
        sys.exit(1)
    
    # Run based on arguments
    if args.demo:
        system.run_demo_scenarios()
    elif args.test:
        # Run built-in tests for all modules
        print("\nRunning module tests...")
        print("Tests are in each module's __main__ block.")
        print("Run: python -m src.modules.MODULE_NAME")
    else:
        # Default: interactive mode
        system.interactive_mode()


if __name__ == "__main__":
    main()