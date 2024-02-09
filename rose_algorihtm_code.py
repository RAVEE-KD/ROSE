import networkx as nx
import csv
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import numpy as np
import timeit
from collections import deque


def construct_graph(dataset="jazz"):
    G = nx.Graph()
    node_mapping = {}
    sequential_number = 0

    with open(file_name, mode='r') as file:
        csvFile = csv.DictReader(file)
        for line in csvFile:
            source = int(line['Source']) - 1
            target = int(line['Target']) - 1

            if source not in node_mapping:
                node_mapping[source] = sequential_number
                sequential_number += 1

            if target not in node_mapping:
                node_mapping[target] = sequential_number
                sequential_number += 1

            G.add_edge(node_mapping[source], node_mapping[target])

    return G


def find_center_nodes(graph, node1, node2):
    shortest_path = nx.shortest_path(graph, source=node1, target=node2)
    num_nodes = len(shortest_path)
    if num_nodes % 2 == 1:
        center_node = shortest_path[num_nodes // 2]
        return [center_node]
    else:
        center_nodes = shortest_path[(num_nodes // 2) - 1: (num_nodes // 2)]
        return center_nodes


def nodes_at_distance(graph, source_node, distance):
    shortest_path_lengths = nx.single_source_shortest_path_length(graph, source_node)

    nodes_at_distance_d = {node for node, dist in shortest_path_lengths.items() if dist == distance}

    return nodes_at_distance_d


def construct_RIT(G, source, kb, kr):
    is_infected_kb, is_infected_kr = False, False
    RIT = nx.Graph()
    Infected = np.full(len(G), -1)
    Infected[source] = 1
    # Model selection
    model = ep.IndependentCascadesModel(G)

    # Model Configuration
    config = mc.Configuration()
    config.add_model_initial_configuration("Infected", [source])

    # Setting the edge parameters
    threshold = 1
    for e in G.edges():
        config.add_edge_configuration("threshold", e, threshold)

    model.set_initial_status(config)

    # Information Propoagation
    iteration = model.iteration()
    RIT.add_node(source)
    while (not is_infected_kb) and (not is_infected_kr):
        iteration = model.iteration()
        Infected_prev = Infected.copy()
        for node in iteration['status']:
            if iteration['status'][node] != 1:
                continue
            Infected[node] = 1
            if node == kb:
                is_infected_kb = True
            if node == kr:
                is_infected_kr = True
            max_degree = 0
            parent = -1
            for neighbor in G.neighbors(node):
                if Infected_prev[neighbor] == 1 and G.degree(neighbor) > max_degree:
                    max_degree = G.degree(neighbor)
                    parent = neighbor
            RIT.add_edge(parent, node)

    return RIT


def SERO(G, kb, kr, node):
    RIT = construct_RIT(G, node, kb, kr)
    visited = set()
    visited.add(node)
    queue = deque([node])
    Scount = 0
    Ekb, Ekr = 0, 0
    while queue:
        current_node = queue.popleft()
        Scount += 1
        neighbors = RIT.neighbors(current_node)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                score_child = Scount
                queue.append(neighbor)
                if neighbor == kb:
                    Ekb = score_child
                if neighbor == kr:
                    Ekr = score_child

    return Ekb, Ekr


def get_node_score(G, node, kb, kb_value, kr, kr_value, Ekb, Ekr):
    d_node_kb = nx.shortest_path_length(G, source=node, target=kb)
    d_node_kr = nx.shortest_path_length(G, source=node, target=kr)
    return (abs(Ekb - kb_value) / d_node_kb) + (abs(Ekr - kr_value) / d_node_kr)


def ROSE(G, observer_shares):
    kb = min(observer_shares, key=observer_shares.get)
    kb_value = observer_shares[kb]
    del observer_shares[kb]

    kr = min(observer_shares, key=observer_shares.get)
    kr_value = observer_shares[kr]

    central_nodes = find_center_nodes(G, kb, kr)

    radius = len(nx.shortest_path(G, kb, kr)) // 2

    sub_nodes = set()
    for node in central_nodes:
        sub_nodes = sub_nodes.union(nodes_at_distance(G, node, radius))

    min_score = float('inf')
    predicted_source = -1

    for node in sub_nodes:
        if node != kb and node != kr:
            Ekb, Ekr = SERO(G, kb, kr, node)
            score = get_node_score(G, node, kb, kb_value, kr, kr_value, Ekb, Ekr)
            if (score < min_score):
                min_score = score
                predicted_source = node

    return [predicted_source]


if __name__ == "__main__":

    # Initial Configuration
    G = construct_graph("wiki-vote_giant_edge_list")

    number_of_nodes = len(G)
    number_of_source = 1
    number_of_observers = number_of_nodes // 10
    Y = np.full(number_of_nodes, -1)

    # Getting sources and observers
    sources = random.sample(range(0, number_of_nodes), number_of_source)
    observers = random.sample(range(0, number_of_nodes), number_of_observers)
    while (set(sources).intersection(observers)):
        observers = random.sample(range(0, number_of_nodes), number_of_observers)

    # Model selection
    model = ep.IndependentCascadesModel(G)

    # Model Configuration
    config = mc.Configuration()
    config.add_model_initial_configuration("Infected", sources)

    # Setting the edge parameters
    threshold = 1
    for e in G.edges():
        config.add_edge_configuration("threshold", e, threshold)

    model.set_initial_status(config)

    # Information Propoagation
    observer_shares = {key: 0 for key in observers}
    shares = 0
    diameter = nx.diameter(G)
    iteration = model.iteration()
    for node in sources:
        Y[node] = 1
    for _ in range(diameter + 1):
        iteration = model.iteration()
        Y_prev = Y.copy()
        parent_shares = {}
        for node in iteration['status']:
            if iteration['status'][node] != 1:
                continue
            Y[node] = 1
            max_degree = 0
            parent = -1
            for neighbor in G.neighbors(node):
                if Y_prev[neighbor] == 1 and G.degree(neighbor) > max_degree:
                    max_degree = G.degree(neighbor)
                    parent = neighbor
            if parent in parent_shares:
                if node in observers:
                    observer_shares[node] = parent_shares[parent]
            else:
                shares += 1
                parent_shares[parent] = shares
                if node in observers:
                    observer_shares[node] = parent_shares[parent]

    # Predict Sources
    start_time = timeit.default_timer()
    predicted_sources = ROSE(G, observer_shares)
    end_time = timeit.default_timer()

    print("Actual Source : ", sources)
    print("Predicted Source : ", predicted_sources)
    print("Distance error : ", nx.shortest_path_length(G, sources[0], predicted_sources[0]))
    print("Time taken : ", end_time - start_time)
