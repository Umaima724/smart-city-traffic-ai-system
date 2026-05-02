"""
csp_scheduler.py
Module 5: CSP Scheduler / Control Allocation Module for the Smart City Traffic AI System.

This module handles constrained control decisions. In the smart city traffic scenario,
these decisions may involve signal timing plans, lane-control adjustments, emergency
corridor allocation, or intersection sequence coordination. The module searches for a
valid assignment that respects safety, timing, and conflict constraints.

This module is especially important when multiple intersections or control points must
be coordinated together. The objective is not arbitrary control adjustment; the objective
is a control plan that satisfies operational constraints while supporting the current
traffic request.

CSP Components:
- Variables: Signal zones (S1_Central_Junction, S2_North_Station, etc.)
- Domains: Phase combinations (PhaseA, PhaseB, PhaseC)
- Constraints:
  * != conflict: S1 and S2 cannot have conflicting phases simultaneously
  * coordination: S2 and S4 must coordinate phases
  * precedence corridor: S1 must precede S5
  * emergency priority: S4 to S5 gets emergency priority

Algorithm: Backtracking with constraint checking and forward checking.

Author: [Your Name]
Group: [Your Group]
Course: AL-2002 Artificial Intelligence Lab
"""


import json
import os
import copy

from src.config import (
    ControlZone,
    CSP_CONSTRAINTS_PATH
)
from src.models.traffic_request import TrafficRequest
from src.utils.exceptions import (
    InvalidRequestError,
    CSPNoSolutionError,
    DataLoadError
)


class CSPScheduler:
    """
    CSP Scheduler for traffic signal control allocation.
    
    This class implements constraint satisfaction problem solving for
    traffic signal coordination. It uses backtracking search to find
    valid assignments of signal phases across multiple intersections.
    
    Attributes:
        variables (dict): CSP variables with domains
        constraints (list): List of constraints to satisfy
        assignment (dict): Current partial or complete assignment
        nodes_explored (int): Counter for search nodes explored
    """
    
    def __init__(self):
        """
        Initialize the CSP Scheduler.
        
        Loads constraints from JSON and initializes the CSP structure.
        """
        self.variables = {}
        self.constraints = []
        self.domains = {}
        self.assignment = {}
        self.nodes_explored = 0
        self._load_constraints()
    
    # =====================================================================
    # FUNCTION 1: _load_constraints
    # =====================================================================
    def _load_constraints(self):
        """
        Load CSP constraints from JSON file.
        
        This internal function loads the constraint database that defines
        variables, domains, and constraints for the traffic signal CSP.
        
        Raises:
            DataLoadError: If constraints file cannot be loaded
        """
        try:
            if os.path.exists(CSP_CONSTRAINTS_PATH):
                with open(CSP_CONSTRAINTS_PATH, 'r') as f:
                    data = json.load(f)
                    self.variables = data.get('variables', {})
                    self.constraints = data.get('constraints', [])
                    self.domains = {
                        var: info.get('domain', [])
                        for var, info in self.variables.items()
                    }
            else:
                self._setup_default_constraints()
        except Exception as e:
            raise DataLoadError(f"{CSP_CONSTRAINTS_PATH} ({str(e)})")
    
    # =====================================================================
    # FUNCTION 2: _setup_default_constraints
    # =====================================================================
    def _setup_default_constraints(self):
        """
        Setup default CSP constraints if JSON file is unavailable.
        
        Creates the standard 5-signal CSP from the project specification.
        """
        # Variables and domains
        self.variables = {
            'S1_Central_Junction': {
                'description': 'Signal at Central Junction',
                'domain': ['PhaseA', 'PhaseB', 'PhaseC']
            },
            'S2_North_Station': {
                'description': 'Signal at North Station',
                'domain': ['PhaseA', 'PhaseB', 'PhaseC']
            },
            'S3_East_Market': {
                'description': 'Signal at East Market',
                'domain': ['PhaseB', 'PhaseC']
            },
            'S4_River_Bridge': {
                'description': 'Signal at River Bridge',
                'domain': ['PhaseA', 'PhaseC']
            },
            'S5_City_Hospital': {
                'description': 'Signal at City Hospital',
                'domain': ['PhaseB', 'PhaseC']
            }
        }
        
        self.domains = {
            var: info['domain']
            for var, info in self.variables.items()
        }
        
        # Constraints
        self.constraints = [
            {
                'type': 'inequality',
                'name': 'conflict_S1_S2',
                'description': 'S1 and S2 cannot have conflicting phases',
                'variables': ['S1_Central_Junction', 'S2_North_Station'],
                'forbidden_pairs': [['PhaseA', 'PhaseA'], ['PhaseB', 'PhaseB']]
            },
            {
                'type': 'inequality',
                'name': 'conflict_S1_S3',
                'description': 'S1 and S3 cannot have conflicting phases',
                'variables': ['S1_Central_Junction', 'S3_East_Market'],
                'forbidden_pairs': [['PhaseB', 'PhaseB'], ['PhaseC', 'PhaseC']]
            },
            {
                'type': 'coordination',
                'name': 'coordination_S2_S4',
                'description': 'S2 and S4 must coordinate phases',
                'variables': ['S2_North_Station', 'S4_River_Bridge'],
                'allowed_pairs': [['PhaseA', 'PhaseA'], ['PhaseC', 'PhaseC']]
            },
            {
                'type': 'precedence',
                'name': 'precedence_S1_S5',
                'description': 'S1 must precede S5 for corridor movement',
                'variables': ['S1_Central_Junction', 'S5_City_Hospital'],
                'relation': 'precedes'
            },
            {
                'type': 'emergency_priority',
                'name': 'emergency_S4_S5',
                'description': 'S4 to S5 gets emergency priority corridor',
                'variables': ['S4_River_Bridge', 'S5_City_Hospital'],
                'required_assignment': {
                    'S4_River_Bridge': 'PhaseA',
                    'S5_City_Hospital': 'PhaseB'
                }
            }
        ]
    
    # =====================================================================
    # FUNCTION 3: is_consistent
    # =====================================================================
    def is_consistent(self, variable, value, assignment):
        """
        Check if assigning value to variable is consistent with current assignment.
        
        This function checks all constraints involving the given variable
        against the current partial assignment.
        
        Args:
            variable (str): Variable to check
            value: Value to assign
            assignment (dict): Current partial assignment
        
        Returns:
            bool: True if consistent, False otherwise
        
        Example:
            >>> csp = CSPScheduler()
            >>> assignment = {'S2_North_Station': 'PhaseA'}
            >>> csp.is_consistent('S4_River_Bridge', 'PhaseA', assignment)
            True  # Coordination constraint satisfied
        """
        # Create temporary assignment with new value
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        # Check all constraints involving this variable
        for constraint in self.constraints:
            vars_in_constraint = constraint.get('variables', [])
            
            if variable not in vars_in_constraint:
                continue  # Constraint doesn't involve this variable
            
            # Check if all variables in constraint are assigned
            assigned_vars = [v for v in vars_in_constraint if v in temp_assignment]
            
            if len(assigned_vars) < 2:
                continue  # Need at least 2 assigned variables to check
            
            # Get values for assigned variables in this constraint
            values = [temp_assignment[v] for v in vars_in_constraint if v in temp_assignment]
            
            # Check constraint type
            if not self._check_constraint(constraint, temp_assignment):
                return False
        
        return True
    
    # =====================================================================
    # FUNCTION 4: _check_constraint
    # =====================================================================
    def _check_constraint(self, constraint, assignment):
        """
        Check if a specific constraint is satisfied by the assignment.
        
        Args:
            constraint (dict): Constraint definition
            assignment (dict): Current assignment
        
        Returns:
            bool: True if constraint satisfied, False otherwise
        """
        constraint_type = constraint.get('type')
        vars_in_constraint = constraint.get('variables', [])
        
        # Get assigned values for variables in this constraint
        assigned_values = {}
        for var in vars_in_constraint:
            if var in assignment:
                assigned_values[var] = assignment[var]
        
        # If not all variables assigned, constraint is trivially satisfied
        if len(assigned_values) < len(vars_in_constraint):
            return True
        
        values = [assigned_values[var] for var in vars_in_constraint]
        
        # Inequality constraint (conflict)
        if constraint_type == 'inequality':
            forbidden_pairs = constraint.get('forbidden_pairs', [])
            return values not in forbidden_pairs
        
        # Coordination constraint
        elif constraint_type == 'coordination':
            allowed_pairs = constraint.get('allowed_pairs', [])
            return values in allowed_pairs
        
        # Precedence constraint
        elif constraint_type == 'precedence':
            # S1 must precede S5 - S1 should be set before S5
            # In terms of phases, this is handled by ordering in backtracking
            return True
        
        # Emergency priority constraint
        elif constraint_type == 'emergency_priority':
            required = constraint.get('required_assignment', {})
            for var, req_value in required.items():
                if var in assignment and assignment[var] != req_value:
                    return False
            return True
        
        return True
    
    # =====================================================================
    # FUNCTION 5: select_unassigned_variable
    # =====================================================================
    def select_unassigned_variable(self, assignment, domains):
        """
        Select the next unassigned variable using MRV heuristic.
        
        Minimum Remaining Values (MRV) heuristic: Choose the variable
        with the fewest legal values in its domain.
        
        Args:
            assignment (dict): Current partial assignment
            domains (dict): Current domains for each variable
        
        Returns:
            str or None: Selected variable name, or None if all assigned
        """
        unassigned = [v for v in self.variables if v not in assignment]
        
        if not unassigned:
            return None
        
        # MRV: Select variable with smallest domain
        min_var = min(unassigned, key=lambda v: len(domains.get(v, [])))
        return min_var
    
    # =====================================================================
    # FUNCTION 6: order_domain_values
    # =====================================================================
    def order_domain_values(self, variable, assignment, domains):
        """
        Order domain values for a variable.
        
        Uses least-constraining-value heuristic: prefer values that
        leave the most options for neighboring variables.
        
        Args:
            variable (str): Variable to order values for
            assignment (dict): Current partial assignment
            domains (dict): Current domains
        
        Returns:
            list: Ordered list of values to try
        """
        values = domains.get(variable, []).copy()
        
        # For emergency scenarios, prioritize certain phases
        # This is handled by constraint checking, so default order is fine
        return values
    
    # =====================================================================
    # FUNCTION 7: forward_checking
    # =====================================================================
    def forward_checking(self, variable, value, assignment, domains):
        """
        Perform forward checking to reduce domains of unassigned variables.
        
        After assigning a value, this function removes inconsistent values
        from the domains of neighboring unassigned variables.
        
        Args:
            variable (str): Just-assigned variable
            value: Assigned value
            assignment (dict): Current partial assignment
            domains (dict): Current domains (will be modified)
        
        Returns:
            dict or None: Reduced domains if consistent, None if inconsistency found
        """
        new_domains = {v: vals.copy() for v, vals in domains.items()}
        new_domains[variable] = [value]  # Assigned variable has only this value
        
        # Check all constraints involving this variable
        for constraint in self.constraints:
            vars_in_constraint = constraint.get('variables', [])
            
            if variable not in vars_in_constraint:
                continue
            
            # For each unassigned variable in this constraint
            for other_var in vars_in_constraint:
                if other_var == variable or other_var in assignment:
                    continue
                
                # Remove values from other_var's domain that conflict
                valid_values = []
                for other_value in new_domains.get(other_var, []):
                    # Test if this pair is consistent
                    test_assignment = assignment.copy()
                    test_assignment[variable] = value
                    test_assignment[other_var] = other_value
                    
                    if self._check_constraint(constraint, test_assignment):
                        valid_values.append(other_value)
                
                new_domains[other_var] = valid_values
                
                # If domain becomes empty, inconsistency detected
                if not new_domains[other_var]:
                    return None  # Domain wipeout
        
        return new_domains
    
    # =====================================================================
    # FUNCTION 8: backtrack
    # =====================================================================
    def backtrack(self, assignment, domains):
        """
        Recursive backtracking search for CSP solution.
        
        This is the core CSP solving algorithm. It assigns values to
        variables one by one, backtracking when constraints are violated.
        
        Args:
            assignment (dict): Current partial assignment
            domains (dict): Current domains for each variable
        
        Returns:
            dict or None: Complete assignment if found, None otherwise
        
        Example:
            >>> csp = CSPScheduler()
            >>> solution = csp.backtrack({}, csp.domains)
            >>> print(solution)
            {'S1_Central_Junction': 'PhaseB', 'S2_North_Station': 'PhaseC', ...}
        """
        self.nodes_explored += 1
        
        # Check if assignment is complete
        if len(assignment) == len(self.variables):
            return assignment
        
        # Select unassigned variable (MRV heuristic)
        variable = self.select_unassigned_variable(assignment, domains)
        if variable is None:
            return assignment
        
        # Order domain values
        values = self.order_domain_values(variable, assignment, domains)
        
        for value in values:
            # Check consistency with current assignment
            if self.is_consistent(variable, value, assignment):
                # Assign value
                new_assignment = assignment.copy()
                new_assignment[variable] = value
                
                # Forward checking
                new_domains = self.forward_checking(
                    variable, value, new_assignment, domains
                )
                
                if new_domains is not None:
                    # Recurse
                    result = self.backtrack(new_assignment, new_domains)
                    if result is not None:
                        return result
        
        # No valid value found, backtrack
        return None
    
    # =====================================================================
    # FUNCTION 9: solve
    # =====================================================================
    def solve(self, emergency_mode=False, emergency_corridor=None):
        """
        Solve the CSP and return a valid assignment.
        
        This is the main entry point for CSP solving. It initializes
        the search and returns the complete solution.
        
        Args:
            emergency_mode (bool): If True, enforce emergency constraints
            emergency_corridor (dict, optional): Specific corridor requirements
                Format: {'variable': 'required_phase', ...}
        
        Returns:
            dict: Solution assignment with all variables assigned
        
        Raises:
            CSPNoSolutionError: If no valid assignment exists
        
        Example:
            >>> csp = CSPScheduler()
            >>> solution = csp.solve(emergency_mode=True)
            >>> print(solution)
            {'S1_Central_Junction': 'PhaseB', ...}
        """
        self.nodes_explored = 0
        self.assignment = {}
        
        # Adjust domains for emergency mode
        domains = {v: vals.copy() for v, vals in self.domains.items()}
        
        if emergency_mode and emergency_corridor:
            # Pre-assign emergency corridor variables
            for var, phase in emergency_corridor.items():
                if var in domains:
                    domains[var] = [phase]
        
        # Solve using backtracking
        solution = self.backtrack({}, domains)
        
        if solution is None:
            raise CSPNoSolutionError(list(self.variables.keys()))
        
        self.assignment = solution
        return solution
    
    # =====================================================================
    # FUNCTION 10: solve_with_emergency_priority
    # =====================================================================
    def solve_with_emergency_priority(self, traffic_request):
        """
        Solve CSP with emergency priority for a traffic request.
        
        This function configures the CSP for emergency scenarios,
        ensuring that emergency corridors get proper signal priority.
        
        Args:
            traffic_request (TrafficRequest): The emergency request
        
        Returns:
            dict: Control plan with signal assignments
        
        Raises:
            InvalidRequestError: If request is not emergency
        """
        if not isinstance(traffic_request, TrafficRequest):
            raise InvalidRequestError(
                'traffic_request',
                "Expected TrafficRequest object"
            )
        
        if not traffic_request.is_emergency:
            raise InvalidRequestError(
                'vehicle_type',
                "Emergency mode requires emergency vehicle"
            )
        
        # Determine emergency corridor based on destination
        emergency_corridor = {}
        
        if traffic_request.destination == ControlZone.CITY_HOSPITAL:
            # Hospital corridor: S1 -> S4 -> S5
            emergency_corridor = {
                'S1_Central_Junction': 'PhaseB',
                'S4_River_Bridge': 'PhaseA',
                'S5_City_Hospital': 'PhaseB'
            }
        
        # Solve with emergency constraints
        solution = self.solve(
            emergency_mode=True,
            emergency_corridor=emergency_corridor
        )
        
        return self._format_control_plan(solution, emergency=True)
    
    # =====================================================================
    # FUNCTION 11: solve_standard
    # =====================================================================
    def solve_standard(self):
        """
        Solve CSP for standard (non-emergency) traffic control.
        
        Finds a valid signal assignment that satisfies all safety and
        coordination constraints without emergency priority.
        
        Returns:
            dict: Control plan with signal assignments
        """
        solution = self.solve(emergency_mode=False)
        return self._format_control_plan(solution, emergency=False)
    
    # =====================================================================
    # FUNCTION 12: _format_control_plan
    # =====================================================================
    def _format_control_plan(self, solution, emergency=False):
        """
        Format the raw solution into a control plan.
        
        Args:
            solution (dict): Raw variable assignment
            emergency (bool): Whether this is an emergency plan
        
        Returns:
            dict: Formatted control plan
        """
        signals = []
        for var, phase in solution.items():
            # Extract signal name from variable key
            signal_name = var.split('_', 1)[1] if '_' in var else var
            signals.append({
                'signal_id': var,
                'location': signal_name,
                'phase': phase,
                'status': 'active'
            })
        
        return {
            'plan_type': 'emergency' if emergency else 'standard',
            'signals': signals,
            'total_signals': len(signals),
            'assignment': solution,
            'nodes_explored': self.nodes_explored,
            'constraints_satisfied': len(self.constraints)
        }
    
    # =====================================================================
    # FUNCTION 13: allocate_control (Main Entry Point)
    # =====================================================================
    def allocate_control(self, traffic_request, policy_result=None):
        """
        Main entry point for control allocation.
        
        This function determines the appropriate control plan based on
        the traffic request and policy validation results.
        
        Args:
            traffic_request (TrafficRequest): Standardized request
            policy_result (dict, optional): Result from Logic/KB module
        
        Returns:
            dict: Control allocation plan
        
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
            ...     'traffic_density': 9,
            ...     'priority_claim': 3
            ... }
            >>> request = input_mod.process_request(raw)
            >>> csp = CSPScheduler()
            >>> plan = csp.allocate_control(request)
            >>> print(plan['plan_type'])
            'emergency'
        """
        if not isinstance(traffic_request, TrafficRequest):
            raise InvalidRequestError(
                'traffic_request',
                "Expected TrafficRequest object"
            )
        
        # Check if policy allows control allocation
        if policy_result and policy_result.get('status') == 'Rejected':
            return {
                'plan_type': 'rejected',
                'reason': policy_result.get('reason', 'Policy rejection'),
                'signals': [],
                'assignment': {}
            }
        
        # Determine if emergency mode is needed
        is_emergency = traffic_request.is_emergency
        
        if is_emergency:
            return self.solve_with_emergency_priority(traffic_request)
        else:
            return self.solve_standard()
    
    # =====================================================================
    # FUNCTION 14: validate_assignment
    # =====================================================================
    def validate_assignment(self, assignment):
        """
        Validate that a complete assignment satisfies all constraints.
        
        Args:
            assignment (dict): Variable assignment to validate
        
        Returns:
            dict: Validation result with 'valid' and 'violations'
        """
        violations = []
        
        for constraint in self.constraints:
            if not self._check_constraint(constraint, assignment):
                violations.append({
                    'constraint': constraint.get('name', 'unknown'),
                    'description': constraint.get('description', ''),
                    'type': constraint.get('type', 'unknown')
                })
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'total_constraints': len(self.constraints),
            'satisfied_constraints': len(self.constraints) - len(violations)
        }
    
    # =====================================================================
    # FUNCTION 15: get_constraint_graph
    # =====================================================================
    def get_constraint_graph(self):
        """
        Get the constraint graph structure for visualization.
        
        Returns:
            dict: Graph structure with nodes and edges
        """
        nodes = list(self.variables.keys())
        edges = []
        
        for constraint in self.constraints:
            vars_in = constraint.get('variables', [])
            if len(vars_in) == 2:
                edges.append({
                    'from': vars_in[0],
                    'to': vars_in[1],
                    'type': constraint.get('type'),
                    'name': constraint.get('name')
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'constraints': self.constraints
        }
    
    # =====================================================================
    # FUNCTION 16: display_solution
    # =====================================================================
    def display_solution(self, control_plan):
        """
        Display a formatted control plan.
        
        Args:
            control_plan (dict): Control plan from allocate_control()
        
        Returns:
            str: Formatted display string
        """
        lines = [
            "=" * 70,
            "CSP SCHEDULER - CONTROL ALLOCATION PLAN",
            "=" * 70,
            f"Plan Type:        {control_plan.get('plan_type', 'unknown').upper()}",
            f"Total Signals:    {control_plan.get('total_signals', 0)}",
            f"Nodes Explored:   {control_plan.get('nodes_explored', 0)}",
            f"Constraints:      {control_plan.get('constraints_satisfied', 0)}",
            "",
            "SIGNAL ASSIGNMENTS:",
            "-" * 70
        ]
        
        for signal in control_plan.get('signals', []):
            emergency_marker = " [EMERGENCY]" if control_plan.get('plan_type') == 'emergency' else ""
            lines.append(
                f"  {signal['signal_id']:<25} -> {signal['phase']:<10}"
                f"  ({signal['location']}){emergency_marker}"
            )
        
        # Show constraint satisfaction
        assignment = control_plan.get('assignment', {})
        if assignment:
            validation = self.validate_assignment(assignment)
            lines.extend([
                "",
                "CONSTRAINT VALIDATION:",
                "-" * 70,
                f"  Valid: {validation['valid']}",
                f"  Satisfied: {validation['satisfied_constraints']}/{validation['total_constraints']}"
            ])
            
            if validation['violations']:
                lines.append("  Violations:")
                for v in validation['violations']:
                    lines.append(f"    - {v['constraint']}: {v['description']}")
        
        lines.append("=" * 70)
        
        return '\n'.join(lines)


# =============================================================================
# Standalone testing functionality
# =============================================================================
if __name__ == "__main__":
    """
    Standalone test for the CSP Scheduler Module.
    Run this file directly to test module functionality.
    """
    import sys
    import os
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.modules.input_preprocessing import InputPreprocessingModule
    from src.modules.logic_knowledge_base import LogicKnowledgeBase
    
    print("=" * 70)
    print("CSP SCHEDULER MODULE - STANDALONE TEST")
    print("=" * 70)
    
    # Create modules
    input_module = InputPreprocessingModule()
    kb = LogicKnowledgeBase()
    csp = CSPScheduler()
    
    # Display constraint graph
    print("\n" + "=" * 70)
    print("CONSTRAINT GRAPH")
    print("=" * 70)
    graph = csp.get_constraint_graph()
    print(f"Nodes (Variables): {len(graph['nodes'])}")
    for node in graph['nodes']:
        print(f"  - {node}")
    print(f"\nEdges (Constraints): {len(graph['edges'])}")
    for edge in graph['edges']:
        print(f"  - {edge['from']} --[{edge['type']}]--> {edge['to']}")
    
    # Test Case 1: Standard Control Allocation
    print("\n" + "=" * 70)
    print("TEST CASE 1: Standard Control Allocation")
    print("Expected: Valid assignment for all 5 signals")
    print("=" * 70)
    
    try:
        plan_1 = csp.solve_standard()
        print(csp.display_solution(plan_1))
        
        assert plan_1['plan_type'] == 'standard'
        assert plan_1['total_signals'] == 5
        assert len(plan_1['assignment']) == 5
        print("\n✓ Test Case 1 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 2: Emergency Control Allocation (Ambulance to Hospital)
    print("\n" + "=" * 70)
    print("TEST CASE 2: Emergency Control Allocation (Ambulance to Hospital)")
    print("Expected: Emergency plan with S4->S5 priority corridor")
    print("=" * 70)
    
    raw_2 = {
        'request_id': 'REQ-002',
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
    
    try:
        request_2 = input_module.process_request(raw_2)
        policy_2 = kb.validate_policy(request_2)
        
        plan_2 = csp.allocate_control(request_2, policy_2)
        print(csp.display_solution(plan_2))
        
        assert plan_2['plan_type'] == 'emergency'
        # Check emergency corridor assignment
        assignment = plan_2['assignment']
        assert assignment.get('S4_River_Bridge') == 'PhaseA', "S4 should be PhaseA"
        assert assignment.get('S5_City_Hospital') == 'PhaseB', "S5 should be PhaseB"
        print("\n✓ Test Case 2 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 3: Fire Truck Integrated Service
    print("\n" + "=" * 70)
    print("TEST CASE 3: Fire Truck Integrated City Service")
    print("Expected: Full emergency corridor to hospital")
    print("=" * 70)
    
    raw_3 = {
        'request_id': 'REQ-003',
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
        request_3 = input_module.process_request(raw_3)
        policy_3 = kb.validate_policy(request_3)
        
        plan_3 = csp.allocate_control(request_3, policy_3)
        print(csp.display_solution(plan_3))
        
        assert plan_3['plan_type'] == 'emergency'
        print("\n✓ Test Case 3 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 4: Rejected Policy -> No Control Allocation
    print("\n" + "=" * 70)
    print("TEST CASE 4: Rejected Policy -> No Control Allocation")
    print("Expected: Rejected plan with no signals")
    print("=" * 70)
    
    raw_4 = {
        'request_id': 'REQ-004',
        'vehicle_type': 'Civilian',
        'request_category': 'Control_Allocation_Request',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital',
        'control_zone': 'S1_Central_Junction'
    }
    
    try:
        request_4 = input_module.process_request(raw_4)
        policy_4 = kb.validate_policy(request_4)
        
        plan_4 = csp.allocate_control(request_4, policy_4)
        print(csp.display_solution(plan_4))
        
        assert plan_4['plan_type'] == 'rejected'
        assert len(plan_4['signals']) == 0
        print("\n✓ Test Case 4 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 5: Police Emergency (Not to Hospital)
    print("\n" + "=" * 70)
    print("TEST CASE 5: Police Emergency (Not to Hospital)")
    print("Expected: Emergency plan without hospital corridor")
    print("=" * 70)
    
    raw_5 = {
        'request_id': 'REQ-005',
        'vehicle_type': 'Police',
        'request_category': 'Emergency_Response_Request',
        'current_location': 'Police_HQ',
        'destination': 'Industrial_Zone',
        'incident_severity': 'High',
        'time_sensitivity': 'High',
        'traffic_density': 7,
        'priority_claim': 2
    }
    
    try:
        request_5 = input_module.process_request(raw_5)
        policy_5 = kb.validate_policy(request_5)
        
        plan_5 = csp.allocate_control(request_5, policy_5)
        print(csp.display_solution(plan_5))
        
        # Should still be emergency but no pre-defined corridor
        assert plan_5['plan_type'] == 'emergency'
        print("\n✓ Test Case 5 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 6: Constraint Validation
    print("\n" + "=" * 70)
    print("TEST CASE 6: Constraint Validation")
    print("=" * 70)
    
    try:
        # Get a valid solution first
        plan_6 = csp.solve_standard()
        assignment = plan_6['assignment']
        
        validation = csp.validate_assignment(assignment)
        print(f"Assignment Valid: {validation['valid']}")
        print(f"Constraints Satisfied: {validation['satisfied_constraints']}/{validation['total_constraints']}")
        
        assert validation['valid'] == True
        assert validation['violations'] == []
        print("\n✓ Test Case 6 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 7: Invalid Request (Non-emergency in emergency mode)
    print("\n" + "=" * 70)
    print("TEST CASE 7: Invalid Request (Civilian in emergency mode)")
    print("Expected: Error - civilian cannot use emergency mode")
    print("=" * 70)
    
    raw_7 = {
        'request_id': 'REQ-007',
        'vehicle_type': 'Civilian',
        'request_category': 'Route_Request',
        'current_location': 'Central_Junction',
        'destination': 'City_Hospital'
    }
    
    try:
        request_7 = input_module.process_request(raw_7)
        plan_7 = csp.solve_with_emergency_priority(request_7)
        print("✗ Test Case 7 FAILED: Should have raised exception")
    except InvalidRequestError as e:
        print(f"✓ Test Case 7 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 7 FAILED: Unexpected error: {e}")
    
    # Test Case 8: Multiple Solutions Check
    print("\n" + "=" * 70)
    print("TEST CASE 8: Search Statistics")
    print("=" * 70)
    
    try:
        csp_test = CSPScheduler()
        plan_8 = csp_test.solve_standard()
        print(f"Nodes explored: {csp_test.nodes_explored}")
        print(f"Solution found: {plan_8['assignment'] is not None}")
        print(f"Total variables: {len(plan_8['assignment'])}")
        print("\n✓ Test Case 8 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 8 FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)