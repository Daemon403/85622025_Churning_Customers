from flask import Flask,render_template, request, jsonify
import pickle,joblib
import numpy as np

app = Flask(__name__)
model = joblib.load(open('./churnmodel.joblib', 'rb'))

@app.route("/")
def predict():
    return render_template("index.html")


@app.route("/sub", methods = ["POST"])
def submit():
    if request.method == "POST":
        features = [int(x) for x in request.form.values()]
        features = np.array([features])
        final = model.predict(features)
        confidence =round((final[0][0]*100))
        final = (final > 0.5).astype(int)
        
        if final == 1:
            finito = "CHURNS"
        else:
            finito = "DOES NOT CHURN"
        
        return render_template("sub.html", fini = finito,confidence=confidence)


if __name__ =="__main__":
    app.run() 