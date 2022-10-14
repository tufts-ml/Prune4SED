import argparse
import os
import time
import torch.optim
import torch_geometric as tg
import config
import train
from tqdm import tqdm
from prune_sed import PruneSED
import numpy as np
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='hyperparameters.')
parser.add_argument('--dataset_name', type=str, default='')
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--input_dim', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--l2', type=float, default=1e-3)
parser.add_argument('--cycle_patience', type=int, default=5)
parser.add_argument('--step_size_up', type=int, default=2000)
parser.add_argument('--step_size_down', type=int, default=2000)
parser.add_argument('--heads', type=int, default=5)
parser.add_argument('--head_layer', type=int, default=4)
parser.add_argument('--predictor_layer', type=int, default=8)
parser.add_argument('--hop', type=int, default=4)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
print(args)


if __name__ == '__main__':
    train_set  = torch.load(f'./data/{args.dataset_name}/train.pt', map_location='cpu')
    val_set  = torch.load(f'./data/{args.dataset_name}/val.pt', map_location='cpu')
    test_set  = torch.load(f'./data/{args.dataset_name}/inner_test.pt', map_location='cpu')

    loader = tg.loader.DataLoader(list(zip(*train_set)), batch_size=args.batch_size, shuffle=True)
    val_loader = tg.loader.DataLoader(list(zip(*val_set)), batch_size=args.batch_size, shuffle=False)
    test_loader = tg.loader.DataLoader(list(zip(*test_set)), batch_size=args.batch_size, shuffle=False)

    args.device = config.device

    model = PruneSED(args).to(config.device)

    if args.test:
        model.load_state_dict(torch.load(f'./saved_model/{args.dataset_name}.pth', map_location=torch.device('cpu')))
        train.test_full(model, test_loader, test_set[2].to(config.device), test_set[3].to(config.device), args.batch_size)
    else:
        dump_path = os.path.join(f'./runlogs/{args.dataset_name}', str(time.time()))
        os.mkdir(dump_path)
        train.train_full(model, loader, val_loader, lr=args.lr, weight_decay=args.l2,
                         cycle_patience=args.cycle_patience, step_size_up=args.step_size_up, step_size_down=args.step_size_down, dump_path=dump_path)
