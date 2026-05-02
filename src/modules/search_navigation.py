"""
search_navigation.py
Module 6: Search & Navigation Module for the Smart City Traffic AI System.

This module performs route generation over the city road network. Intersections
are modeled as nodes and roads as edges, with optional weights representing time,
congestion, delay, or access penalties. This allows the system to support both
standard and emergency route planning.

Operationally, the system uses:
- BFS (Breadth-First Search): For unweighted route problems - finds shortest
  path in terms of number of edges
- UCS (Uniform Cost Search): For weighted problems without a suitable heuristic
  - finds lowest cost path
- A* (A-Star Search): For weighted problems when heuristic guidance is available
  - finds optimal path efficiently using heuristic estimates

Search may operate directly on a route request, or it may run after control
allocation when the final route depends on approved traffic-control conditions.

The module supports both standard civilian routing and emergency vehicle routing
with corridor preferences.

Author: [Your Name]
Group: [Your Group]
Course: AL-2002 Artificial Intelligence Lab
"""


import heapq
import json
import math
import os

from src.config import (
    CITY_GRAPH_UNWEIGHTED_PATH,
    CITY_GRAPH_WEIGHTED_PATH,
    ControlZone
)
from src.models.traffic_request import TrafficRequest
from src.utils.exceptions import (
    InvalidRequestError,
    NoValidRouteError,
    DataLoadError
)
from src.utils.graph_loader import (
    load_unweighted_graph,
    load_weighted_graph,
    get_graph_neighbors,
    get_edge_cost
)


class SearchNavigationModule:
    """
    Search & Navigation Module for route generation.
    
    This class implements BFS, UCS, and A* search algorithms for finding
    optimal routes in the city road network. It supports both weighted
    and unweighted graphs, with special handling for emergency routes.
    
    Attributes:
        unweighted_graph (dict): Unweighted city graph (adjacency list)
        weighted_graph (dict): Weighted city graph with edge costs
        heuristic_table (dict): Precomputed heuristic estimates
    """
    
    def __init__(self):
        """
        Initialize the Search & Navigation Module.
        
        Loads both weighted and unweighted city graphs and initializes
        the heuristic table for A* search.
        """
        self.unweighted_graph = None
        self.weighted_graph = None
        self.heuristic_table = {}
        self._load_graphs()
        self._build_heuristic_table()
    
    # =====================================================================
    # FUNCTION 1: _load_graphs
    # =====================================================================
    def _load_graphs(self):
        """
        Load both unweighted and weighted city graphs.
        
        This internal function loads the road network data from JSON files.
        If files are unavailable, it creates default graph structures.
        
        Raises:
            DataLoadError: If graphs cannot be loaded
        """
        try:
            self.unweighted_graph = load_unweighted_graph()
            self.weighted_graph = load_weighted_graph()
        except DataLoadError:
            # Use default graphs if files unavailable
            self._setup_default_graphs()
    
    # =====================================================================
    # FUNCTION 2: _setup_default_graphs
    # =====================================================================
    def _setup_default_graphs(self):
        """
        Setup default city graphs if JSON files are unavailable.
        
        Creates the standard city graph from the project specification.
        """
        nodes = [
            'Central_Junction', 'North_Station', 'River_Bridge',
            'Police_HQ', 'Traffic_Control_Center', 'Stadium',
            'East_Market', 'Airport_Road', 'City_Hospital',
            'South_Residential', 'West_Terminal', 'Fire_Station',
            'Industrial_Zone'
        ]
        
        # Unweighted graph - adjacency list
        self.unweighted_graph = {
            'nodes': nodes,
            'edges': {
                'Central_Junction': ['North_Station', 'East_Market', 'South_Residential', 'West_Terminal'],
                'North_Station': ['Central_Junction', 'River_Bridge', 'Traffic_Control_Center'],
                'River_Bridge': ['North_Station', 'Police_HQ', 'Stadium'],
                'Police_HQ': ['River_Bridge', 'Traffic_Control_Center'],
                'Traffic_Control_Center': ['Police_HQ', 'North_Station'],
                'Stadium': ['River_Bridge', 'East_Market', 'Airport_Road'],
                'East_Market': ['Stadium', 'Central_Junction', 'City_Hospital'],
                'Airport_Road': ['Stadium', 'South_Residential'],
                'City_Hospital': ['East_Market', 'South_Residential'],
                'South_Residential': ['City_Hospital', 'Airport_Road', 'Central_Junction'],
                'West_Terminal': ['Central_Junction', 'Fire_Station', 'Industrial_Zone'],
                'Fire_Station': ['West_Terminal'],
                'Industrial_Zone': ['West_Terminal']
            }
        }
        
        # Weighted graph - adjacency list with costs
        self.weighted_graph = {
            'nodes': nodes,
            'edges': {
                'Central_Junction': {
                    'North_Station': 3, 'East_Market': 3,
                    'South_Residential': 4, 'West_Terminal': 4
                },
                'North_Station': {
                    'Central_Junction': 3, 'River_Bridge': 4,
                    'Traffic_Control_Center': 2
                },
                'River_Bridge': {
                    'North_Station': 4, 'Police_HQ': 2, 'Stadium': 2
                },
                'Police_HQ': {
                    'River_Bridge': 2, 'Traffic_Control_Center': 2
                },
                'Traffic_Control_Center': {
                    'Police_HQ': 2, 'North_Station': 2
                },
                'Stadium': {
                    'River_Bridge': 2, 'East_Market': 2, 'Airport_Road': 5
                },
                'East_Market': {
                    'Stadium': 2, 'Central_Junction': 3, 'City_Hospital': 3
                },
                'Airport_Road': {
                    'Stadium': 5, 'South_Residential': 2
                },
                'City_Hospital': {
                    'East_Market': 3, 'South_Residential': 6
                },
                'South_Residential': {
                    'City_Hospital': 6, 'Airport_Road': 2, 'Central_Junction': 4
                },
                'West_Terminal': {
                    'Central_Junction': 4, 'Fire_Station': 2, 'Industrial_Zone': 4
                },
                'Fire_Station': {'West_Terminal': 2},
                'Industrial_Zone': {'West_Terminal': 4}
            }
        }
    
    # =====================================================================
    # FUNCTION 3: _build_heuristic_table
    # =====================================================================
    def _build_heuristic_table(self):
        """
        Build heuristic estimates for A* search.
        
        Creates a table of straight-line distance estimates between
        all pairs of nodes. These are admissible heuristics (never
        overestimate actual cost).
        """
        # Predefined heuristic estimates (straight-line distance approximations)
        # These are admissible - they underestimate or equal actual cost
        self.heuristic_table = {
            # From Central_Junction
            ('Central_Junction', 'City_Hospital'): 4.0,
            ('Central_Junction', 'North_Station'): 3.0,
            ('Central_Junction', 'West_Terminal'): 4.0,
            ('Central_Junction', 'East_Market'): 3.0,
            ('Central_Junction', 'South_Residential'): 4.0,
            ('Central_Junction', 'Airport_Road'): 6.0,
            ('Central_Junction', 'River_Bridge'): 5.0,
            ('Central_Junction', 'Stadium'): 5.0,
            ('Central_Junction', 'Police_HQ'): 6.0,
            ('Central_Junction', 'Traffic_Control_Center'): 4.0,
            ('Central_Junction', 'Fire_Station'): 6.0,
            ('Central_Junction', 'Industrial_Zone'): 8.0,
            
            # From North_Station
            ('North_Station', 'City_Hospital'): 6.0,
            ('North_Station', 'Central_Junction'): 3.0,
            ('North_Station', 'River_Bridge'): 4.0,
            ('North_Station', 'Traffic_Control_Center'): 2.0,
            ('North_Station', 'Stadium'): 5.0,
            ('North_Station', 'Airport_Road'): 7.0,
            
            # From River_Bridge
            ('River_Bridge', 'City_Hospital'): 7.0,
            ('River_Bridge', 'North_Station'): 4.0,
            ('River_Bridge', 'Stadium'): 2.0,
            ('River_Bridge', 'Police_HQ'): 2.0,
            
            # From Stadium
            ('Stadium', 'City_Hospital'): 5.0,
            ('Stadium', 'East_Market'): 2.0,
            ('Stadium', 'Airport_Road'): 5.0,
            ('Stadium', 'River_Bridge'): 2.0,
            
            # From East_Market
            ('East_Market', 'City_Hospital'): 3.0,
            ('East_Market', 'Central_Junction'): 3.0,
            ('East_Market', 'Stadium'): 2.0,
            
            # From City_Hospital
            ('City_Hospital', 'East_Market'): 3.0,
            ('City_Hospital', 'South_Residential'): 6.0,
            ('City_Hospital', 'Central_Junction'): 4.0,
            
            # From South_Residential
            ('South_Residential', 'City_Hospital'): 6.0,
            ('South_Residential', 'Airport_Road'): 2.0,
            ('South_Residential', 'Central_Junction'): 4.0,
            
            # From Airport_Road
            ('Airport_Road', 'South_Residential'): 2.0,
            ('Airport_Road', 'Stadium'): 5.0,
            
            # From West_Terminal
            ('West_Terminal', 'Central_Junction'): 4.0,
            ('West_Terminal', 'Fire_Station'): 2.0,
            ('West_Terminal', 'Industrial_Zone'): 4.0,
            
            # From Police_HQ
            ('Police_HQ', 'River_Bridge'): 2.0,
            ('Police_HQ', 'Traffic_Control_Center'): 2.0,
            
            # From Traffic_Control_Center
            ('Traffic_Control_Center', 'Police_HQ'): 2.0,
            ('Traffic_Control_Center', 'North_Station'): 2.0,
            
            # From Fire_Station
            ('Fire_Station', 'West_Terminal'): 2.0,
            
            # From Industrial_Zone
            ('Industrial_Zone', 'West_Terminal'): 4.0
        }
    
    # =====================================================================
    # FUNCTION 4: heuristic
    # =====================================================================
    def heuristic(self, node, goal):
        """
        Get heuristic estimate from node to goal.
        
        Returns an admissible estimate of the cost from the current
        node to the goal. If no precomputed heuristic exists, returns
        a conservative estimate.
        
        Args:
            node (str): Current node
            goal (str): Goal node
        
        Returns:
            float: Heuristic estimate (always <= actual cost)
        """
        if node == goal:
            return 0.0
        
        # Check both directions
        h = self.heuristic_table.get((node, goal))
        if h is not None:
            return h
        
        h = self.heuristic_table.get((goal, node))
        if h is not None:
            return h
        
        # Fallback: use a fraction of minimum edge cost
        return 1.0
    
    # =====================================================================
    # FUNCTION 5: bfs
    # =====================================================================
    def bfs(self, graph, start, goal):
        """
        Breadth-First Search for unweighted graphs.
        
        Finds the shortest path in terms of number of edges (hops).
        Uses a queue (FIFO) for frontier management.
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Args:
            graph (dict): Unweighted graph with 'nodes' and 'edges'
            start (str): Starting node
            goal (str): Goal node
        
        Returns:
            tuple: (path, cost) where path is list of nodes and cost is
                   number of edges
        
        Raises:
            NoValidRouteError: If no path exists
        
        Example:
            >>> search = SearchNavigationModule()
            >>> path, cost = search.bfs(search.unweighted_graph, 
            ...                         'Central_Junction', 'City_Hospital')
            >>> print(path)
            ['Central_Junction', 'East_Market', 'City_Hospital']
            >>> print(cost)
            2
        """
        # Validate nodes
        nodes = graph.get('nodes', [])
        if start not in nodes:
            raise InvalidRequestError('start', f"Node '{start}' not in graph")
        if goal not in nodes:
            raise InvalidRequestError('goal', f"Node '{goal}' not in graph")
        
        # Edge case: start is goal
        if start == goal:
            return [start], 0
        
        # BFS initialization
        frontier = [(start, [start])]  # Queue of (node, path)
        visited = {start}
        
        while frontier:
            current, path = frontier.pop(0)  # FIFO - queue
            
            # Get neighbors
            neighbors = get_graph_neighbors(graph, current)
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                new_path = path + [neighbor]
                
                # Goal check
                if neighbor == goal:
                    return new_path, len(new_path) - 1  # cost = number of edges
                
                visited.add(neighbor)
                frontier.append((neighbor, new_path))
        
        # No path found
        raise NoValidRouteError(start, goal)
    
    # =====================================================================
    # FUNCTION 6: ucs
    # =====================================================================
    def ucs(self, graph, start, goal):
        """
        Uniform Cost Search for weighted graphs.
        
        Finds the lowest-cost path from start to goal. Uses a priority
        queue ordered by cumulative path cost. Expands lowest-cost node first.
        
        Time Complexity: O((V + E) log V)
        Space Complexity: O(V)
        
        Args:
            graph (dict): Weighted graph with 'nodes' and 'edges'
            start (str): Starting node
            goal (str): Goal node
        
        Returns:
            tuple: (path, cost) where path is list of nodes and cost is
                   total edge weight
        
        Raises:
            NoValidRouteError: If no path exists
        
        Example:
            >>> search = SearchNavigationModule()
            >>> path, cost = search.ucs(search.weighted_graph,
            ...                         'Central_Junction', 'City_Hospital')
            >>> print(path)
            ['Central_Junction', 'East_Market', 'City_Hospital']
            >>> print(cost)
            6.0
        """
        # Validate nodes
        nodes = graph.get('nodes', [])
        if start not in nodes:
            raise InvalidRequestError('start', f"Node '{start}' not in graph")
        if goal not in nodes:
            raise InvalidRequestError('goal', f"Node '{goal}' not in graph")
        
        # Edge case: start is goal
        if start == goal:
            return [start], 0.0
        
        # UCS initialization
        # Priority queue: (cost, counter, node, path)
        counter = 0
        frontier = [(0.0, counter, start, [start])]
        visited = set()
        
        while frontier:
            cost, _, current, path = heapq.heappop(frontier)
            
            # Skip if already visited with lower cost
            if current in visited:
                continue
            
            # Goal check
            if current == goal:
                return path, cost
            
            visited.add(current)
            
            # Get neighbors and their costs
            edges = graph.get('edges', {}).get(current, {})
            
            for neighbor, edge_cost in edges.items():
                if neighbor in visited:
                    continue
                
                new_cost = cost + edge_cost
                new_path = path + [neighbor]
                counter += 1
                
                heapq.heappush(frontier, (new_cost, counter, neighbor, new_path))
        
        # No path found
        raise NoValidRouteError(start, goal)
    
    # =====================================================================
    # FUNCTION 7: astar
    # =====================================================================
    def astar(self, graph, start, goal):
        """
        A* Search for weighted graphs with heuristic guidance.
        
        Finds the optimal path using f(n) = g(n) + h(n) where:
        - g(n) is the actual cost from start to n
        - h(n) is the heuristic estimate from n to goal
        
        Time Complexity: O((V + E) log V) with good heuristic
        Space Complexity: O(V)
        
        Args:
            graph (dict): Weighted graph with 'nodes' and 'edges'
            start (str): Starting node
            goal (str): Goal node
        
        Returns:
            tuple: (path, cost) where path is list of nodes and cost is
                   total edge weight
        
        Raises:
            NoValidRouteError: If no path exists
        
        Example:
            >>> search = SearchNavigationModule()
            >>> path, cost = search.astar(search.weighted_graph,
            ...                           'Central_Junction', 'City_Hospital')
            >>> print(path)
            ['Central_Junction', 'East_Market', 'City_Hospital']
            >>> print(cost)
            6.0
        """
        # Validate nodes
        nodes = graph.get('nodes', [])
        if start not in nodes:
            raise InvalidRequestError('start', f"Node '{start}' not in graph")
        if goal not in nodes:
            raise InvalidRequestError('goal', f"Node '{goal}' not in graph")
        
        # Edge case: start is goal
        if start == goal:
            return [start], 0.0
        
        # A* initialization
        # Priority queue: (f_score, counter, g_score, node, path)
        # f_score = g_score + heuristic
        counter = 0
        start_h = self.heuristic(start, goal)
        frontier = [(start_h, counter, 0.0, start, [start])]
        
        # Track best g_score for each node
        best_g = {start: 0.0}
        
        while frontier:
            f, _, g, current, path = heapq.heappop(frontier)
            
            # Goal check
            if current == goal:
                return path, g
            
            # Skip if we've found a better path to this node
            if g > best_g.get(current, float('inf')):
                continue
            
            # Get neighbors and their costs
            edges = graph.get('edges', {}).get(current, {})
            
            for neighbor, edge_cost in edges.items():
                new_g = g + edge_cost
                
                # Only consider if this path is better
                if new_g < best_g.get(neighbor, float('inf')):
                    best_g[neighbor] = new_g
                    new_path = path + [neighbor]
                    h = self.heuristic(neighbor, goal)
                    f = new_g + h
                    counter += 1
                    
                    heapq.heappush(frontier, (f, counter, new_g, neighbor, new_path))
        
        # No path found
        raise NoValidRouteError(start, goal)
    
    # =====================================================================
    # FUNCTION 8: get_route
    # =====================================================================
    def get_route(self, graph, start, goal, algorithm='ucs'):
        """
        Unified interface for route computation.
        
        Selects and executes the appropriate search algorithm based on
        the specified parameter.
        
        Args:
            graph (dict): City graph (weighted or unweighted)
            start (str): Starting location
            goal (str): Destination location
            algorithm (str): 'bfs', 'ucs', or 'astar'
        
        Returns:
            dict: Route result with path, cost, algorithm used, etc.
        
        Raises:
            InvalidValueError: If algorithm is not recognized
        
        Example:
            >>> search = SearchNavigationModule()
            >>> result = search.get_route(search.weighted_graph,
            ...                           'Central_Junction', 'City_Hospital',
            ...                           'astar')
            >>> print(result['path'])
            ['Central_Junction', 'East_Market', 'City_Hospital']
        """
        if algorithm == 'bfs':
            path, cost = self.bfs(graph, start, goal)
            cost_type = 'hops'
        elif algorithm == 'ucs':
            path, cost = self.ucs(graph, start, goal)
            cost_type = 'cost'
        elif algorithm == 'astar':
            path, cost = self.astar(graph, start, goal)
            cost_type = 'cost'
        else:
            from src.utils.exceptions import InvalidValueError
            raise InvalidValueError(
                'algorithm',
                algorithm,
                "'bfs', 'ucs', or 'astar'"
            )
        
        return {
            'path': path,
            'cost': cost,
            'cost_type': cost_type,
            'algorithm': algorithm,
            'start': start,
            'goal': goal,
            'num_nodes': len(path)
        }
    
    # =====================================================================
    # FUNCTION 9: find_route (Main Entry Point)
    # =====================================================================
    def find_route(self, traffic_request, control_plan=None):
        """
        Main entry point for route finding.
        
        Determines the appropriate graph, algorithm, and route based on
        the traffic request and any approved control plan.
        
        For emergency requests, may use weighted graph with A* for
        fastest route. For standard requests, uses UCS or BFS.
        
        Args:
            traffic_request (TrafficRequest): Standardized request
            control_plan (dict, optional): Approved control plan from CSP
        
        Returns:
            dict: Complete route result
        
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
            >>> search = SearchNavigationModule()
            >>> route = search.find_route(request)
            >>> print(route['path'])
            ['Central_Junction', 'East_Market', 'City_Hospital']
        """
        if not isinstance(traffic_request, TrafficRequest):
            raise InvalidRequestError(
                'traffic_request',
                "Expected TrafficRequest object"
            )
        
        start = traffic_request.current_location
        goal = traffic_request.destination
        
        # Determine algorithm based on request type
        if traffic_request.request_category == 'Route_Request':
            # Simple route: use BFS for unweighted shortest path
            algorithm = 'bfs'
            graph = self.unweighted_graph
        elif traffic_request.is_emergency:
            # Emergency: use A* for fastest weighted path
            algorithm = 'astar'
            graph = self.weighted_graph
        else:
            # Standard weighted: use UCS
            algorithm = 'ucs'
            graph = self.weighted_graph
        
        # Compute route
        try:
            route_result = self.get_route(graph, start, goal, algorithm)
            
            # Add request-specific information
            route_result['is_emergency'] = traffic_request.is_emergency
            route_result['vehicle_type'] = traffic_request.vehicle_type
            route_result['request_id'] = traffic_request.request_id
            
            # Estimate travel time (assuming cost represents minutes)
            route_result['estimated_time_minutes'] = route_result['cost']
            
            # Add control plan info if available
            if control_plan:
                route_result['control_plan_applied'] = True
                route_result['signal_corridor'] = [
                    s['signal_id'] for s in control_plan.get('signals', [])
                    if s.get('status') == 'active'
                ]
            else:
                route_result['control_plan_applied'] = False
            
            return route_result
            
        except NoValidRouteError:
            return {
                'path': [],
                'cost': float('inf'),
                'error': f'No valid route from {start} to {goal}',
                'start': start,
                'goal': goal,
                'request_id': traffic_request.request_id
            }
    
    # =====================================================================
    # FUNCTION 10: compare_algorithms
    # =====================================================================
    def compare_algorithms(self, start, goal):
        """
        Compare all three search algorithms on the same route.
        
        Useful for analysis and demonstration of algorithm differences.
        
        Args:
            start (str): Starting node
            goal (str): Goal node
        
        Returns:
            dict: Comparison results for BFS, UCS, and A*
        """
        results = {}
        
        # BFS on unweighted graph
        try:
            path, cost = self.bfs(self.unweighted_graph, start, goal)
            results['bfs'] = {
                'path': path,
                'cost': cost,
                'cost_type': 'hops',
                'num_nodes': len(path)
            }
        except Exception as e:
            results['bfs'] = {'error': str(e)}
        
        # UCS on weighted graph
        try:
            path, cost = self.ucs(self.weighted_graph, start, goal)
            results['ucs'] = {
                'path': path,
                'cost': cost,
                'cost_type': 'weight',
                'num_nodes': len(path)
            }
        except Exception as e:
            results['ucs'] = {'error': str(e)}
        
        # A* on weighted graph
        try:
            path, cost = self.astar(self.weighted_graph, start, goal)
            results['astar'] = {
                'path': path,
                'cost': cost,
                'cost_type': 'weight',
                'num_nodes': len(path)
            }
        except Exception as e:
            results['astar'] = {'error': str(e)}
        
        return results
    
    # =====================================================================
    # FUNCTION 11: display_route
    # =====================================================================
    def display_route(self, route_result):
        """
        Display a formatted route result.
        
        Args:
            route_result (dict): Route from find_route() or get_route()
        
        Returns:
            str: Formatted display string
        """
        if 'error' in route_result and not route_result.get('path'):
            lines = [
                "=" * 70,
                "SEARCH & NAVIGATION - ROUTE RESULT",
                "=" * 70,
                f"Status: ERROR",
                f"Error:  {route_result['error']}",
                "=" * 70
            ]
            return '\n'.join(lines)
        
        lines = [
            "=" * 70,
            "SEARCH & NAVIGATION - ROUTE RESULT",
            "=" * 70,
            f"Request ID:       {route_result.get('request_id', 'N/A')}",
            f"Vehicle Type:     {route_result.get('vehicle_type', 'N/A')}",
            f"Emergency:        {route_result.get('is_emergency', False)}",
            f"Algorithm:        {route_result.get('algorithm', 'N/A').upper()}",
            f"Start:            {route_result.get('start', 'N/A')}",
            f"Goal:             {route_result.get('goal', 'N/A')}",
            "",
            "ROUTE PATH:",
            "-" * 70
        ]
        
        path = route_result.get('path', [])
        for i, node in enumerate(path):
            arrow = " → " if i < len(path) - 1 else ""
            lines.append(f"  [{i}] {node}{arrow}")
        
        lines.extend([
            "",
            "ROUTE STATISTICS:",
            "-" * 70,
            f"  Total Nodes:      {route_result.get('num_nodes', 0)}",
            f"  Total Cost:       {route_result.get('cost', 0)} {route_result.get('cost_type', '')}",
            f"  Est. Travel Time: {route_result.get('estimated_time_minutes', 0)} minutes"
        ])
        
        if route_result.get('control_plan_applied'):
            lines.extend([
                "",
                "CONTROL PLAN:",
                "-" * 70,
                f"  Applied: Yes",
                f"  Signals: {', '.join(route_result.get('signal_corridor', []))}"
            ])
        
        lines.append("=" * 70)
        
        return '\n'.join(lines)


# =============================================================================
# Standalone testing functionality
# =============================================================================
if __name__ == "__main__":
    """
    Standalone test for the Search & Navigation Module.
    Run this file directly to test module functionality.
    """
    import sys
    import os
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.modules.input_preprocessing import InputPreprocessingModule
    from src.modules.logic_knowledge_base import LogicKnowledgeBase
    from src.modules.csp_scheduler import CSPScheduler
    
    print("=" * 70)
    print("SEARCH & NAVIGATION MODULE - STANDALONE TEST")
    print("=" * 70)
    
    # Create modules
    input_module = InputPreprocessingModule()
    kb = LogicKnowledgeBase()
    csp = CSPScheduler()
    search = SearchNavigationModule()
    
    # Display graph info
    print("\n" + "=" * 70)
    print("CITY GRAPH INFORMATION")
    print("=" * 70)
    print(f"Unweighted Graph Nodes: {len(search.unweighted_graph.get('nodes', []))}")
    print(f"Weighted Graph Nodes:   {len(search.weighted_graph.get('nodes', []))}")
    print(f"Total Edges (weighted): {sum(len(v) for v in search.weighted_graph.get('edges', {}).values())}")
    
    # Test Case 1: BFS - Shortest Path (Unweighted)
    print("\n" + "=" * 70)
    print("TEST CASE 1: BFS - Shortest Path (Unweighted)")
    print("Route: Central_Junction -> City_Hospital")
    print("=" * 70)
    
    try:
        result_1 = search.get_route(
            search.unweighted_graph,
            'Central_Junction',
            'City_Hospital',
            'bfs'
        )
        print(search.display_route(result_1))
        assert result_1['algorithm'] == 'bfs'
        assert result_1['cost_type'] == 'hops'
        print("\n✓ Test Case 1 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 2: UCS - Lowest Cost (Weighted)
    print("\n" + "=" * 70)
    print("TEST CASE 2: UCS - Lowest Cost (Weighted)")
    print("Route: Central_Junction -> City_Hospital")
    print("=" * 70)
    
    try:
        result_2 = search.get_route(
            search.weighted_graph,
            'Central_Junction',
            'City_Hospital',
            'ucs'
        )
        print(search.display_route(result_2))
        assert result_2['algorithm'] == 'ucs'
        assert result_2['cost_type'] == 'cost'
        # Verify it's the lowest cost path
        assert result_2['path'] == ['Central_Junction', 'East_Market', 'City_Hospital']
        assert result_2['cost'] == 6.0  # 3 + 3
        print("\n✓ Test Case 2 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 3: A* - Optimal with Heuristic
    print("\n" + "=" * 70)
    print("TEST CASE 3: A* - Optimal with Heuristic")
    print("Route: Central_Junction -> City_Hospital")
    print("=" * 70)
    
    try:
        result_3 = search.get_route(
            search.weighted_graph,
            'Central_Junction',
            'City_Hospital',
            'astar'
        )
        print(search.display_route(result_3))
        assert result_3['algorithm'] == 'astar'
        # A* should find same optimal path as UCS
        assert result_3['path'] == ['Central_Junction', 'East_Market', 'City_Hospital']
        assert result_3['cost'] == 6.0
        print("\n✓ Test Case 3 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 4: Longer Route - North_Station to Airport_Road
    print("\n" + "=" * 70)
    print("TEST CASE 4: Longer Route - North_Station to Airport_Road")
    print("=" * 70)
    
    try:
        result_4 = search.get_route(
            search.weighted_graph,
            'North_Station',
            'Airport_Road',
            'astar'
        )
        print(search.display_route(result_4))
        assert len(result_4['path']) > 2
        print("\n✓ Test Case 4 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 5: Emergency Route with Control Plan
    print("\n" + "=" * 70)
    print("TEST CASE 5: Emergency Route with Control Plan")
    print("Route: Central_Junction -> City_Hospital (Ambulance)")
    print("=" * 70)
    
    try:
        raw_5 = {
            'request_id': 'REQ-005',
            'vehicle_type': 'Ambulance',
            'request_category': 'Emergency_Response_Request',
            'current_location': 'Central_Junction',
            'destination': 'City_Hospital',
            'incident_severity': 'High',
            'time_sensitivity': 'High',
            'traffic_density': 9,
            'priority_claim': 3
        }
        
        request_5 = input_module.process_request(raw_5)
        policy_5 = kb.validate_policy(request_5)
        plan_5 = csp.allocate_control(request_5, policy_5)
        route_5 = search.find_route(request_5, plan_5)
        
        print(search.display_route(route_5))
        assert route_5['is_emergency'] == True
        assert route_5['algorithm'] == 'astar'
        assert route_5['control_plan_applied'] == True
        print("\n✓ Test Case 5 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 6: Standard Civilian Route
    print("\n" + "=" * 70)
    print("TEST CASE 6: Standard Civilian Route")
    print("Route: North_Station -> Airport_Road")
    print("=" * 70)
    
    try:
        raw_6 = {
            'request_id': 'REQ-006',
            'vehicle_type': 'Civilian',
            'request_category': 'Route_Request',
            'current_location': 'North_Station',
            'destination': 'Airport_Road',
            'traffic_density': 4
        }
        
        request_6 = input_module.process_request(raw_6)
        route_6 = search.find_route(request_6)
        
        print(search.display_route(route_6))
        assert route_6['is_emergency'] == False
        assert route_6['algorithm'] == 'bfs'  # Route_Request uses BFS
        print("\n✓ Test Case 6 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 7: Algorithm Comparison
    print("\n" + "=" * 70)
    print("TEST CASE 7: Algorithm Comparison")
    print("Route: Central_Junction -> City_Hospital")
    print("=" * 70)
    
    try:
        comparison = search.compare_algorithms('Central_Junction', 'City_Hospital')
        
        for algo, result in comparison.items():
            print(f"\n{algo.upper()}:")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Path: {result['path']}")
                print(f"  Cost: {result['cost']} {result['cost_type']}")
                print(f"  Nodes: {result['num_nodes']}")
        
        # Verify all algorithms find valid paths
        assert all('error' not in r for r in comparison.values())
        print("\n✓ Test Case 7 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 8: Invalid Algorithm
    print("\n" + "=" * 70)
    print("TEST CASE 8: Invalid Algorithm Selection")
    print("=" * 70)
    
    try:
        result_8 = search.get_route(
            search.weighted_graph,
            'Central_Junction',
            'City_Hospital',
            'invalid_algo'
        )
        print("✗ Test Case 8 FAILED: Should have raised exception")
    except Exception as e:
        print(f"✓ Test Case 8 PASSED: Caught expected error")
        print(f"  Error: {e}")
    
    # Test Case 9: No Valid Route (Disconnected nodes - simulated)
    print("\n" + "=" * 70)
    print("TEST CASE 9: Invalid Node")
    print("=" * 70)
    
    try:
        result_9 = search.get_route(
            search.weighted_graph,
            'Central_Junction',
            'Invalid_Node',
            'ucs'
        )
        print("✗ Test Case 9 FAILED: Should have raised exception")
    except InvalidRequestError as e:
        print(f"✓ Test Case 9 PASSED: Caught expected error")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"✗ Test Case 9 FAILED: Unexpected error: {e}")
    
    # Test Case 10: Same Start and Goal
    print("\n" + "=" * 70)
    print("TEST CASE 10: Same Start and Goal")
    print("=" * 70)
    
    try:
        result_10 = search.get_route(
            search.weighted_graph,
            'Central_Junction',
            'Central_Junction',
            'astar'
        )
        print(search.display_route(result_10))
        assert result_10['path'] == ['Central_Junction']
        assert result_10['cost'] == 0
        print("\n✓ Test Case 10 PASSED")
    except Exception as e:
        print(f"\n✗ Test Case 10 FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)