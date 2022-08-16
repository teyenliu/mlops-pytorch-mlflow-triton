import os
import torch
from elb_model import MLP

model = MLP()

if not os.path.isdir("model_zoo"):
    os.makedirs("model_zoo")
torch.save(model, os.path.join("model_zoo", "mlp.pth"))
