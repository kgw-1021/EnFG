from typing import List, Dict

class Node:
    def __init__(self, name: str, dims: list) -> None:
        self._name = name
        self._dims = dims
        self.edges: List['Edge'] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dims(self) -> list:
        return self._dims

    def add_edge(self, edge):
        if edge not in self.edges:
            self.edges.append(edge)
            
    def remove_edge(self, edge):
        if edge in self.edges:
            self.edges.remove(edge)

class Edge:
    def __init__(self, node0: Node, node1: Node) -> None:
        self._node0 = node0
        self._node1 = node1
        self._messages = {} 
        
        node0.add_edge(self)
        node1.add_edge(self)

    def get_other(self, node: Node) -> Node:
        return self._node1 if node is self._node0 else self._node0

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def connect(self, node0: Node, node1: Node):
        edge = Edge(node0, node1)
        self.edges.append(edge)
        if node0 not in self.nodes: self.nodes.append(node0)
        if node1 not in self.nodes: self.nodes.append(node1)
        return edge


    def remove_edge(self, edge: Edge):
        if edge in self.edges:
            edge._node0.remove_edge(edge)
            edge._node1.remove_edge(edge)
            self.edges.remove(edge)


    def remove_node(self, node: Node):
        if node in self.nodes:
            for edge in list(node.edges):
                self.remove_edge(edge)
            
            self.nodes.remove(node)