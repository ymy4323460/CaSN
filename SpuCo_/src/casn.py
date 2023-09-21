import argparse
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from spuco.datasets import WILDSDatasetWrapper
from spuco.evaluate import Evaluator
from spuco.robust_train import CaSN
from spuco.models import casn_model_factory
from spuco.utils import set_seed
from spuco.datasets import SpuCoAnimals

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=7)
parser.add_argument("--bias", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="./data")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="./results/spucoanimals_casn")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

args = parser.parse_args()

#args.logits_save_dir = os.path.join(args.logits_save_dir, f"lr={args.lr}_wd={args.weight_decay}_seed={args.seed}")
#os.makedirs(args.logits_save_dir, exist_ok=True)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

trainset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    transform=transform,
)
trainset.initialize()

valset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="val",
    transform=transform,
)
valset.initialize()

testset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="test",
    transform=transform,
)
testset.initialize()

model = casn_model_factory("resnet18", trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)

model = CaSN(
    model=model,
    val_evaluator=valid_evaluator,
    num_epochs=args.num_epochs,
    trainset=trainset,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True,
    bias=args.bias
)
model.train()

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()

results = pd.DataFrame(index=[0])
results["timestamp"] = pd.Timestamp.now()
results["seed"] = args.seed
results["pretrained"] = args.pretrained
results["lr"] = args.lr
results["weight_decay"] = args.weight_decay
results["momentum"] = args.momentum
results["num_epochs"] = args.num_epochs
results["batch_size"] = args.batch_size

results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model.best_model,
    device=device,
    verbose=True
)
evaluator.evaluate()

results["early_stopping_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["early_stopping_average_accuracy"] = evaluator.average_accuracy

print('results[early_stopping_worst_group_accuracy]', results["early_stopping_worst_group_accuracy"])
print('results[early_stopping_average_accuracy]', results["early_stopping_average_accuracy"])



print('Done!')
print('Results saved to', args.results_csv+'.csv')
results_csv_path = args.results_csv+'.csv'
if os.path.exists(results_csv_path):
   results_df = pd.read_csv(results_csv_path)
else:
   results_df = pd.DataFrame()

results_df = pd.concat([results_df, results], ignore_index=True)
results_df.to_csv(results_csv_path, index=False)
