from src.train_jampr import train
from src.architecture_jampr_2 import AttentionModel
import argparse
import pickle 

parser = argparse.ArgumentParser()
parser.add_argument("output", help="output file/directory")
#parser.add_argument("--widths", help="widths of correlations to check", type=str, default='5,10')

args = parser.parse_args()

model = AttentionModel(active_num=1).to("cuda")

with open('../../../home/loc_distance_islands.pkl', 'rb') as f:
    r = pickle.load(f)

best_weights, weights, name, loss, reward = train(model, device="cuda", problem_size=100, num_vehicles=12,
                            batch_size=128, epochs=50, T=1000, lr=1e-4, decay=0.001, save_inbetween=True,
                                                  output=args.output, r=r)
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
