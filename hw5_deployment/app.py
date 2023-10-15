import pickle
from flask import Flask, jsonify, request

MODEL_PATH = "model2.bin"
DV_PATH = "dv.bin"

model, dv = None, None
with open(MODEL_PATH, "rb") as f_model:
    model = pickle.load(f_model)

with open(DV_PATH, "rb") as f_dv:
    dv = pickle.load(f_dv)

app = Flask("scoring")

@app.route("/predict_score", methods=['POST'])
def predict_score():
    customer = request.get_json()

    X = dv.transform([customer])
    y_proba = model.predict_proba(X)[0, 1]

    output = {'proba': float(y_proba.round(3))}
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='localhost', port=9696, debug=True)
