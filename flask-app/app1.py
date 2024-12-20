from flask import Flask, render_template,request
import mlflow
from preprocessing_utility import preprocess_comment
import dagshub

dagshub.init(repo_owner='himadri06', repo_name='youtube-comment-analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/himadri06/youtube-comment-analysis.mlflow")

app = Flask(__name__)

#Load Model from model registry

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    text = request.form['text']
    
    text = preprocess_comment(text)

    return text

app.run(debug=True)