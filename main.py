import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle5 as pickle
import matplotlib.pyplot as plt
import matplotlib as m
m.use('Agg')

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
#     features = request.form.values()
    features = np.array([float(x) for x in request.form.values()]).reshape(-1,1)
    print("features:\t{}".format(features))
    fin_features = scaler.transform(features)

    final_features = np.array(fin_features).reshape(1,1,10)
    prediction = model.predict(final_features)
    prediction = scaler.inverse_transform(prediction)
    output = np.round(prediction[0], 2)
    nan = np.nan
    features = np.append(features, nan)
    predicted = np.zeros(len(features))
    predicted[:] = nan
    predicted[-1] = output
    predicted[-2] = features[-2]
    fig = plt.figure()
    plt.plot(features, label = "Current")
    plt.plot(predicted, label ="Future")
    plt.legend(loc='best')
    filename='static/plot{}.png'.format(np.random.randint(100,200))
    plt.savefig(filename)
    
    return render_template('index.html', prediction_text='Dew Point Average for tomorrow: {}'.format(output), filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
