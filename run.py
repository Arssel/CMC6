from src.train_jampr import train
from src.architecture_jampr_2 import AttentionModel
import argparse
import pickle 

parser = argparse.ArgumentParser()
parser.add_argument("output", help="output file/directory")
#parser.add_argument("--widths", help="widths of correlations to check", type=str, default='5,10')

args = parser.parse_args()

model = AttentionModel(active_num=1).to("cuda")

best_weights, weights, name, loss, reward = train(model, 'cuda', epochs=200, problem_size=20, T=1000, lr=1e-4, batch_size=128,
          penalty_num_vertexes=1000, penalty_num_vehicles=0)
f = open(args.output + name +'.pkl', 'wb')
pickle.dump(weights, f)
f.close()
f = open(args.output + name +'_best.pkl', 'wb')
pickle.dump(best_weights, f)
f.close()
f = open(args.output + name +'_loss.pkl', 'wb')
pickle.dump(loss, f)
f.close()
f = open(args.output + name +'_reward.pkl', 'wb')
pickle.dump(reward, f)
f.close()
