from src.train import train
from src.architecture import AttentionModel
import argparse
import pickle 

parser = argparse.ArgumentParser()
parser.add_argument("output", help="output file/directory")
#parser.add_argument("--widths", help="widths of correlations to check", type=str, default='5,10')

args = parser.parse_args()


model = AttentionModel().to("cuda")
opts = {#'demand_type': {'distribution': 'uniform', 'max_demand':10},
        #'tw_type': {'distribution': 'uniform'},
        #"pickup_and_delivery":True
       }
weights, name = train(model, opts, 'cuda', problem_size='random', T=1000, lr=1e-4, batch_size=1024)
f = open(args.output + name +'.pkl', 'wb')
pickle.dump(weights, f)
f.close()

model = AttentionModel().to("cuda")
opts = {'metric': 1, #demand_type': {'distribution': 'uniform', 'max_demand':10},
        #'tw_type': {'distribution': 'uniform'},
        #"pickup_and_delivery":True
       }
weights, name = train(model, opts, 'cuda', problem_size=20, T=1000, lr=1e-4, batch_size=1024)
f = open(args.output + name +'_1.pkl', 'wb')
pickle.dump(weights, f)
f.close()

model = AttentionModel().to("cuda")
opts = {'metric': 1, #demand_type': {'distribution': 'uniform', 'max_demand':10},
        #'tw_type': {'distribution': 'uniform'},
        #"pickup_and_delivery":True
       }
weights, name = train(model, opts, 'cuda', problem_size=50, T=1000, lr=1e-4, batch_size=1024)
f = open(args.output + name +'_1.pkl', 'wb')
pickle.dump(weights, f)
f.close()
    