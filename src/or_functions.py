from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import math
import time

def total_distance(data, manager, routing, solution):
    total_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        print(vehicle_id)
        index = routing.Start(vehicle_id)
        print(index)
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        total_distance += route_distance
    return total_distance

def compute_distance(input_data, eps=1e-5, time_limit=1):

    data = {}
    if 'dist' in input_data:
        data['distance_matrix'] = input_data['dist']/eps
    else:
        1/0
    if 'num_vehicles' in input_data:
        data['num_vehicles'] = input_data['num_vehicles']
    else:
        data['num_vehicles'] = 1
    if 'time_windows' in input_data:
        data['time_windows'] = input_data['time_windows']/eps
        if 'penalty' in input_data:
            data['penalty'] = input_data['penalty']
        else:
            data['penalty'] = int(10/eps)
    else:
        data['time_windows'] = None
    if 'depot' in input_data:
        data['depot'] = input_data['depot']
    else:
        if 'demand' in input_data:
            data['depot'] = len(data['distance_matrix']) - 1
        else:
            data['depot'] = 0
    if 'pickup_and_delivery' in input_data:
        data['pickup_and_delivery'] = input_data['pickup_and_delivery']
    else:
        data['pickup_and_delivery'] = None
    if 'service_time' in input_data:
        data['service_time'] = int(input_data['service_time']/eps)
    if 'demand' in input_data:
        input_data['demand'][data['depot']] = 0
        data['demand'] = input_data['demand']
    else:
        data['demand'] = None
    if 'vehicle_capacities' in input_data:
        data['vehicle_capacities'] = input_data['vehicle_capacities']
    else:
        data['vehicle_capacities'] = [20]*data['num_vehicles']
    n = data['distance_matrix'].shape[0]
    manager = pywrapcp.RoutingIndexManager(data['distance_matrix'].shape[0],
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
       
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        30000000000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    
    if not data['time_windows'] is None:
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            int(1/eps),  # allow waiting time
            int(1/eps),  # maximum time per vehicle
            True,  # Don't force start cumul to zero.
            time)
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == n-1:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                    data['time_windows'][0][1])
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i)))
            
        penalty = data['penalty']
        for node in range(len(data['distance_matrix'])-1):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
            
        def service_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['service_time']

    if not data['pickup_and_delivery'] is None:
        for request in data['pickup_and_delivery']:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(
                    delivery_index))
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <=
                distance_dimension.CumulVar(delivery_index))
            
    if not data['demand'] is None:

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0, 
            data['vehicle_capacities'], 
            True,
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
    
    print(solution is None)
    
    return total_distance(data, manager, routing, solution)*eps