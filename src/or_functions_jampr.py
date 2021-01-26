from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import math
import time

def total_distance(data, manager, routing, solution):
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        #plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            #plan_output += '{0} Time({1},{2}) -> '.format(
            #    manager.IndexToNode(index), solution.Min(time_var),
            #    solution.Max(time_var))
            index = solution.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        #plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
        #                                            solution.Min(time_var),
        #                                            solution.Max(time_var))
        #plan_output += 'Time of the route: {}min\n'.format(
        #    solution.Min(time_var))
        #print(plan_output)
        total_time += solution.Min(time_var)
    return total_time

def compute_distance(data, eps=1e-5, time_limit=1):

    manager = pywrapcp.RoutingIndexManager(data['time_matrix'].shape[0],
                                           data['num_vehicles'], 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['time_matrix'][from_node][to_node]/eps)

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

       
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        30000000000,  # allow waiting time
        30000000000,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(int(time_window[0]/eps), int(time_window[1]/eps))
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(int(data['time_windows'][0][0]/eps),
                                                int(data['time_windows'][0][1]/eps))

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))
    
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
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

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    if time_limit < 1:
        search_parameters.time_limit.FromMilliseconds(int(time_limit*1000))
    else:
        search_parameters.time_limit.seconds = time_limit
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    return total_distance(data, manager, routing, solution)*eps