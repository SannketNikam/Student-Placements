from flask import Flask, render_template, request
import config
import pickle
import numpy as np

model = pickle.load(open(config.MODEL_PATH, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/", methods = ['POST'])
def predict():
    cgpa = float(request.form.get('cgpa'))
    iq = int(request.form.get('iq'))
    profile_score = int(request.form.get('profile_score'))

    result = model.predict(np.array([cgpa, iq, profile_score]).reshape(1, 3))
    print(f"{result}")

    if result[0] == 1:
        result = "Student will get placed!"
    else:
        result = "Student will not get placed!"

    return render_template('index.html', predictions = result)

if __name__ == "__main__":
    app.run(host = config.HOST, port = config.PORT, debug=True)