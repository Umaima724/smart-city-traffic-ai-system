"""
graph_loader.py
Utility for loading city graph data from JSON files.
Provides functions to load unweighted and weighted graph representations.
"""


import json
import os


def load_json_data(file_path):
    """
    Load and parse JSON data from a file.
    
    This function reads a JSON file and returns its contents as a Python object.
    It includes error handling for missing files and invalid JSON.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict or list: Parsed JSON data
        
    Raises:
        DataLoadError: If file cannot be read or JSON is invalid
    """
    from src.utils.exceptions import DataLoadError
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise DataLoadError(file_path)
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        raise DataLoadError(f"{file_path} (Invalid JSON: {str(e)})")
    except IOError as e:
        raise DataLoadError(f"{file_path} (IO Error: {str(e)})")


def load_unweighted_graph(file_path=None):
    """
    Load the unweighted city graph from JSON.
    
    The unweighted graph represents intersections as nodes and roads as edges
    without cost information. Used for BFS routing.
    
    Args:
        file_path (str, optional): Path to graph file. 
                                  Defaults to config.CITY_GRAPH_UNWEIGHTED_PATH.
    
    Returns:
        dict: Adjacency list representation of the graph
              Format: {'nodes': [...], 'edges': {'node': ['neighbor1', ...]}}
    """
    from src.config import CITY_GRAPH_UNWEIGHTED_PATH
    
    if file_path is None:
        file_path = CITY_GRAPH_UNWEIGHTED_PATH
    
    data = load_json_data(file_path)
    
    # Validate graph structure
    if 'nodes' not in data or 'edges' not in data:
        from src.utils.exceptions import DataLoadError
        raise DataLoadError(
            f"{file_path} (Missing 'nodes' or 'edges' key)"
        )
    
    return data


def load_weighted_graph(file_path=None):
    """
    Load the weighted city graph from JSON.
    
    The weighted graph includes edge costs representing travel time or distance.
    Used for UCS and A* routing algorithms.
    
    Args:
        file_path (str, optional): Path to graph file.
                                  Defaults to config.CITY_GRAPH_WEIGHTED_PATH.
    
    Returns:
        dict: Weighted adjacency list representation
              Format: {'nodes': [...], 'edges': {'node': {'neighbor': cost, ...}}}
    """
    from src.config import CITY_GRAPH_WEIGHTED_PATH
    
    if file_path is None:
        file_path = CITY_GRAPH_WEIGHTED_PATH
    
    data = load_json_data(file_path)
    
    # Validate graph structure
    if 'nodes' not in data or 'edges' not in data:
        from src.utils.exceptions import DataLoadError
        raise DataLoadError(
            f"{file_path} (Missing 'nodes' or 'edges' key)"
        )
    
    return data


def get_graph_neighbors(graph_data, node):
    """
    Get neighbors of a node from graph data.
    
    Args:
        graph_data (dict): Loaded graph data
        node (str): Node to get neighbors for
        
    Returns:
        list: List of neighbor nodes
    """
    edges = graph_data.get('edges', {})
    
    # Handle both unweighted (list) and weighted (dict) formats
    node_edges = edges.get(node, [])
    
    if isinstance(node_edges, dict):
        # Weighted graph - return keys
        return list(node_edges.keys())
    else:
        # Unweighted graph - return list
        return node_edges


def get_edge_cost(graph_data, from_node, to_node):
    """
    Get the cost of an edge in a weighted graph.
    
    Args:
        graph_data (dict): Loaded weighted graph data
        from_node (str): Starting node
        to_node (str): Destination node
        
    Returns:
        float or None: Edge cost if exists, None otherwise
    """
    edges = graph_data.get('edges', {})
    node_edges = edges.get(from_node, {})
    
    if isinstance(node_edges, dict):
        return node_edges.get(to_node)
    
    return None  # Unweighted graph has no costs


def validate_graph_connectivity(graph_data):
    """
    Validate that the graph is connected (all nodes reachable from any node).
    
    Args:
        graph_data (dict): Graph data to validate
        
    Returns:
        bool: True if connected, False otherwise
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', {})
    
    if not nodes:
        return True  # Empty graph is trivially connected
    
    # BFS from first node
    start = nodes[0]
    visited = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        current = queue.pop(0)
        
        # Get neighbors
        node_edges = edges.get(current, [])
        if isinstance(node_edges, dict):
            neighbors = list(node_edges.keys())
        else:
            neighbors = node_edges
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(nodes)