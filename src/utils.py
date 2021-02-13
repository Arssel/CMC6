import numpy as np
import torch 

def graph_generation(n=20, bsz=16, graph_type=None):
    d = {}
    if graph_type is None or len(graph_type) == 0:
        dots = np.random.rand(bsz, n, 2)
    elif graph_type['distribution'] == 'uniform':
        dots = np.random.rand(bsz, n, 2)
    elif graph_type['distribution'] == 'normal_capped':
        if graph_type['sigma'] == 'default':
            sigma = 0.5/15
        else:
            sigma = graph_type['sigma']
        if graph_type['cov'] == 'default':
            cov = np.eye(2)
        else:
            if graph_type['cov'] == 'random':
                A = np.random.rand(2,2)
                A = np.matmul(A.T, A)
                cov = A/np.linalg.norm(A)
                d['cov'] = cov
            else:
                cov = graph_type['cov']
        dots = np.random.multivariate_normal(mean=(0.5,0.5),cov=cov*sigma, size=(bsz, n))
        dots[dots > 1] = 1
        dots[dots < 0] = 0
    elif graph_type['distribution'] == 'centroids':
        if graph_type['num_cent'] == 'poisson':
            if graph_type['lambda'] ==  'default':
                l = 1.5
            else:
                l = graph_type['lambda']
            num_cent = np.random.poisson(l) + 2
            d['num_cent'] = num_cent
        elif graph_type['num_cent'] == 'default':
            num_cent = 3
        else:
            num_cent = graph_type['num_cent']
        centroids = np.random.rand(bsz, num_cent, 2)
        d['centroids'] = centroids
        near_which_cent = np.random.choice(np.arange(num_cent), size=(bsz, n - num_cent))
        dots_by_batch = []
        for j in range(bsz):
            dots_j_batch = centroids[j].squeeze()
            for i in range(num_cent):
                near_which_cent_j = near_which_cent[j]
                num_dots_i_cent_j_batch = (near_which_cent_j==i).sum()
                if num_dots_i_cent_j_batch == 0:
                    continue
                range_to_board = min(centroids[j, i].min(), (1-centroids[j, i]).min())
                sigma = range_to_board/30
                #A = np.random.rand(2,2)
                #A = np.matmul(A.T, A)
                #cov = A/np.linalg.norm(A)
                cov = np.eye(2)
                dots_to_append = np.random.multivariate_normal(mean=centroids[j, i],cov=cov*sigma, size=(num_dots_i_cent_j_batch,))
                dots_j_batch = np.append(dots_j_batch, dots_to_append, axis=0)
            dots_by_batch.append(dots_j_batch[np.newaxis,:,:])
        dots = np.concatenate(dots_by_batch, axis=0)
    else:
        assert 0, 'unknown distribution: ' + str(graph_type['distribution'])
    return dots, d

def pickup_and_delivery_generation(n=20, bsz=16, p_and_d=None):
    if p_and_d is None or len(p_and_d) == 0:
        return None
    else:
        if p_and_d['pickup_and_delivery'] == True:
            assert n % 2 == 0, 'even number of vertxes needed'
            pairs = []
            for j in range(bsz):
                permutation = np.random.permutation(np.arange(n))
                pair = np.zeros((n//2, 2), dtype=int)
                pair[:, 0] = permutation[::2]
                pair[:, 1] = permutation[1::2]
                pairs.append(pair[np.newaxis,:,:])
            pairs = np.concatenate(pairs, axis = 0)
            return pairs
        else:
            return None

def p_and_d_features(pairs, dots):
    bsz = dots.shape[0]
    n = dots.shape[1]
    pd_features = np.zeros(dots.shape)
    for i in range(bsz):
        pd_features[i, pairs[i, :, 0], :] = dots[i, pairs[i, :, 1],:]
        pd_features[i, pairs[i, :, 1], :] = dots[i, pairs[i, :, 0],:]
    return pd_features
        
def demand_generation(n=20, bsz=16, demand_type=None, p_and_d=None):
    if p_and_d is None or len(p_and_d) == 0:
        pd_flag = True
    else:
        assert n % 2 == 0, 'even number of vertxes needed'
        pd_flag = False
    if demand_type is None or len(demand_type) == 1:
        demand = np.zeros((bsz, n, 1))
        return demand
    else:
        if demand_type['distribution'] == 'uniform':
            if demand_type['max_demand'] == 'default':
                max_demand = 10
            else:
                max_demand = demand_type['max_demand']
            demand_range = np.arange(1, max_demand+1)
            if pd_flag:
                demand = np.random.choice(demand_range, size=(bsz, n, 1))
            else:
                demand = np.random.choice(demand_range, size=(bsz, n//2, 1))
        elif demand_type['distribution'] == 'poisson':
            if demand_type['lambda'] ==  'default':
                l = 3.5
            else:
                l = demand_type['lambda']
            if pd_flag:
                demand = np.random.poisson(l, size=(bsz, n, 1)) + 1
            else:
                demand = np.random.poisson(l, size=(bsz, n//2, 1)) + 1
            demand[demand > demand_type['capacity']] = demand_type['capacity']
        else:
            assert 0, 'unknown demand distribution: ' + str(demand_type['distribution'])
        if 'supply' in demand_type and pd_flag:
            if demand_type['supply'] == True:
                if demand_type['supply_prob'] == 'default':
                    supply_prob = 1/2
                else:
                    supply_prob = demand_type['supply_prob'] 
                supply = -(2*np.random.binomial(size=(bsz, n), n=1, p=supply_prob)-1)
                demand = supply*demand
        if not pd_flag:
            total_demand = np.zeros((bsz, n, 1))
            for i in range(bsz):
                total_demand[i, p_and_d[i, :, 0], :] = -demand[i]
                total_demand[i, p_and_d[i, :, 1], :] = demand[i]
            demand = total_demand
    return demand

def tw_generation(n=20, bsz=16, tw_type=None, p_and_d=None):
    if p_and_d is None:
        pd_flag = True
    else:
        assert n % 2 == 0, 'even number of vertxes needed'
        pd_flag = False
    if tw_type is None or len(tw_type) == 0:
        return np.zeros((bsz, n, 2))
    elif pd_flag:
        if tw_type['distribution'] == 'half':
            tw = np.zeros((bsz, n, 2))
            tw[:, :, 0] = np.random.binomial(size=(bsz, n), n=1, p=1/2)/2
            tw[:, :, 1] = a + 1/2
        elif tw_type['distribution'] == 'uniform':
            a = np.random.rand(bsz, n)
            b = np.zeros((bsz, n))
            for i in range(bsz):
                for j in range(n):
                    b[i,j] = np.random.rand()*(1-a[i,j]) + a[i,j]
            tw = np.zeros((bsz, n, 2))
            tw[:, :, 0] = a
            tw[:, :, 1] = b
        else:
            assert 0, 'unknown demand distribution: ' + str(demand_type['distribution'])
    else:
        assert n % 2 == 0, 'even number of vertex needed'
        if tw_type['distribution'] == 'half':
            tw = np.zeros((bsz, n, 2))
            tw[p_and_d[:, :, 0], 0] = 0
            tw[p_and_d[:, :, 0], 1] = 1/2
            tw[p_and_d[:, :, 1], 0] = 1/2
            tw[p_and_d[:, :, 1], 1] = 1
        elif tw_type['distribution'] == 'uniform':
            a_1 = np.random.rand(bsz, n//2)
            b_1 = np.zeros((bsz, n//2))
            a_2 = np.zeros((bsz, n//2))
            b_2 = np.zeros((bsz, n//2))
            for i in range(bsz):
                for j in range(n//2):
                    b_1[i,j] = np.random.rand()*(1-a_1[i,j]) + a_1[i,j]
                    a_2[i,j] = np.random.rand()*(1-a_1[i,j]) + a_1[i,j]
            for i in range(bsz):
                for j in range(n//2):
                    b_2[i,j] = np.random.rand()*(1-a_2[i,j]) + a_2[i,j]                
            tw = np.zeros((bsz, n, 2))
            for i in range(p_and_d.shape[0]):
                tw[i, p_and_d[i, :, 0], 0] = a_1[i]
                tw[i, p_and_d[i, :, 0], 1] = b_1[i]
                tw[i, p_and_d[i, :, 1], 0] = a_2[i]
                tw[i, p_and_d[i, :, 1], 1] = b_2[i]
    return tw

def path_distance(matrix, path):
    """
    :param matrix: - an NxN matrix representing the TSP graph distances.
    :param path: - a list of N nodes, representing TSP solution.
    """
    batch_size = matrix.shape[0]
    N = matrix.shape[1]
    assert N == path.shape[1], \
        "Number of visited nodes must be equal to matrix shape:" \
        f"Expected {N}, but got {len(path)}."
    assert (path < N).all()
    distance = np.array([0.] * batch_size)
    matrix = matrix.reshape(-1, N**2)
    batch_range = np.arange(batch_size)
    path = path.squeeze()
    for i in range(1, N):
        distance += matrix[(batch_range, path[:, i-1]*N + path[:,i])]
    distance += matrix[(batch_range, path[:, i] * N + path[:, 0])]
    return distance

def path_distance_new(matrix, path):
    """
    :param matrix: - an NxN matrix representing the TSP graph distances.
    :param path: - a list of N nodes, representing TSP solution.
    """
    batch_size = matrix.shape[0]
    N = path.shape[1]
    num_nodes = matrix.shape[1]
    distance = np.array([0.] * batch_size)
    matrix = matrix.reshape(-1, num_nodes**2)
    batch_range = np.arange(batch_size)
    for i in range(1, N):
        distance += matrix[(batch_range, path[:, i-1]*num_nodes + path[:,i])]
    distance += matrix[(batch_range, path[:, i] * num_nodes + path[:, 0])]
    return distance

def check_repeating_vertexes(vertexes):
    bsz = vertexes.shape[0]
    mask = vertexes[:,1:] != vertexes[:,:-1]
    mask = torch.cat((torch.ones((bsz, 1), dtype=bool), mask), dim=1)
    return mask

def check_missing_vertexes(vertexes, flags, opts, n):
    penalties = []
    size = n
    for b in vertexes:
        left = set(np.arange(size)) - set(b.numpy())
        penalties.append(len(left))
    return np.array(penalties)


def path_distance_jampr(matrix, plan):
    batch_size = matrix.shape[0]
    plan_flat = plan.view(batch_size, -1)
    N = plan_flat.shape[1]
    num_nodes = matrix.shape[1]
    matrix = matrix.reshape(-1, num_nodes**2)
    batch_range = np.arange(batch_size)
    distance = matrix[batch_range, plan_flat[:, 0]]
    for i in range(1, N):
        distance += matrix[(batch_range, plan_flat[:, i-1]*num_nodes + plan_flat[:,i])]
    return distance

def check_missing_vertexes_jampr(plan, n):
    bsz = plan.shape[0]
    plan_flat = plan.view(bsz, -1)
    penalties = []
    for p in plan_flat:
        #print(p)
        left = set(np.arange(n+1)) - set(p.numpy()) - {0}
        penalties.append(len(left))
    return torch.Tensor(penalties)
        