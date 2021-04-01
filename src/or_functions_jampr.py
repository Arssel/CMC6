from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import math
import time


def total_distance(data, manager, routing, solution):
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    total_routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route = [index]
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), solution.Min(time_var),
                solution.Max(time_var))
            index = solution.Value(routing.NextVar(index))
            from_node = manager.IndexToNode(index)
            #print(data['time_windows'][from_node])
            if from_node != data['time_matrix'].shape[0] + 1:
                route.append(from_node)
        total_routes.append(route)
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    solution.Min(time_var),
                                                    solution.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            solution.Min(time_var))
        #print(plan_output)
        total_time += solution.Min(time_var)
    penalty = 0
    dropped_nodes = 'Dropped nodes:'
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if solution.Value(routing.NextVar(node)) == node:
            penalty += 72000
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
    #print(dropped_nodes)
    return total_time + penalty, total_routes


def compute_distance(data, eps=1e-5, time_limit=1):
    if 'ends' in data:
        manager = pywrapcp.RoutingIndexManager(data['time_matrix'].shape[0],
                                               data['num_vehicles'], data['starts'], data['ends'])
    else:
        manager = pywrapcp.RoutingIndexManager(data['time_matrix'].shape[0],
                                               data['num_vehicles'], 0)

    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['time_matrix'][from_node][to_node]/eps)


    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    time_name = 'Time'
    routing.AddDimension(
        transit_callback_index,
        30000000000,  # allow waiting time
        30000000000,  # maximum time per vehicle
        True,
        time_name)
    time_dimension = routing.GetDimensionOrDie(time_name)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == (len(data['time_windows']) - 1) or location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        #print(location_idx, index, int(time_window[0]/eps), int(time_window[1]/eps))
        time_dimension.CumulVar(index).SetRange(int(time_window[0]/eps), int(time_window[1]/eps))
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(int(data['time_windows'][0][0]/eps),
                                                int(data['time_windows'][0][1]/eps))
    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        time_dimension.SetSpanCostCoefficientForVehicle(1, i)
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(data['demands'][from_node])
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    #print(data['pickups_deliveries'])
    if 'pickups_deliveries' in data:
        for request in data['pickups_deliveries']:
            #print(request[0])
            #print(request[1])
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(
                    delivery_index))
            routing.solver().Add(
                time_dimension.CumulVar(pickup_index) <=
                time_dimension.CumulVar(delivery_index))

    penalty = int(720/eps)
    for node in range(1, len(data['time_matrix']) - 1):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.FromMilliseconds(int(time_limit*1000))
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    solution = routing.SolveWithParameters(search_parameters)
    
    dist, routes = total_distance(data, manager, routing, solution)
    return dist*eps/720, routes