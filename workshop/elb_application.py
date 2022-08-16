import io
import json
import os

import flask
import numpy as np
import torch
from PIL import Image

model = torch.load(os.path.join("model_zoo", "mlp.pth"))
model.eval()

application = flask.Flask(__name__)

@application.route("/", methods=["POST"])
def predict():
    """The predict function accept POST with binary image data.
    """
    data = flask.request.data

    img = Image.open(io.BytesIO(data))
    img = np.asarray(img)

    inputs = torch.from_numpy(np.expand_dims(img, axis=0)).to(torch.float32)
    logits = model(inputs).detach().numpy()[0]
    results = {"logits": logits.tolist(), "cls": f"{np.argmax(logits)}"}

    return flask.Response(json.dumps(results), status=200, mimetype="text/json")


application.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    return flask.Response(response='\n', status=200, mimetype='application/json')


"""
Use curl to make post request:
curl -X POST -H 'content-type: image/jpg' --data-binary \
    @./CIFAR-10-images/train/airplane/0010.jpg localhost:5000
"""

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000, debug=False)
