
"""
Use curl to make post request:
curl -X POST -H 'content-type: image/jpg' --data-binary @/home/danny/data/CIFAR-10-images/train/bird/0000.jpg localhost:5000
"""

import io
import json
import os

import flask
import numpy as np
import torch
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from .cifar10_train import Net

# Assuming that we are on a CUDA machine, this should print a CUDA device:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load(os.path.join("model_zoo", "cifar_net.pth"), map_location=device)
model.eval()

application = flask.Flask(__name__)


@application.route("/", methods=["POST"])
def predict():
    """The predict function accept POST with binary image data.
    """
    data = flask.request.data

    img = Image.open(io.BytesIO(data))
    img = np.asarray(img)
    #change image channel ordering between channels first and channels last in numpy array
    img = np.moveaxis(img, -1, 0)
    
    #增加一個維度在第0軸
    #batch_t = torch.unsqueeze(img_t, 0)

    inputs = torch.from_numpy(np.expand_dims(img, axis=0)).to(torch.float32)

    #change image channel ordering between channels first and channels last in torch tensor
    #inputs.permute(2, 0, 1)
    logits = model(inputs.to(device)).to('cpu').detach().numpy()[0]
    results = {"logits": logits.tolist(), "cls": f"{np.argmax(logits)}"}

    return flask.Response(json.dumps(results), status=200, mimetype="text/json")


@application.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    return flask.Response(response='\n', status=200, mimetype='application/json')


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000, debug=False)
