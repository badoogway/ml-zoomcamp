import pickle

MODEL_PATH = "model1.bin"
DV_PATH = "dv.bin"

def load_model(model_path, dv_path):
    with open(model_path, "rb") as f_model:
        model = pickle.load(f_model)

    with open(dv_path, "rb") as f_dv:
        dv = pickle.load(f_dv)

    return model, dv

def predict_score(model, dv, customer):
    X = dv.transform([customer])
    y_proba = model.predict_proba(X)[0, 1]

    return y_proba

customer = {"job": "retired", "duration": 445, "poutcome": "success"}
model, dv = load_model(MODEL_PATH, DV_PATH)
customer_score = predict_score(model, dv, customer)

print(customer_score.round(3))
