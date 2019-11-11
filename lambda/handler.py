try:
    import unzip_requirements
except ImportError:
    pass

import json
import os
import boto3
import pickle
from base64 import b64decode
from twython import Twython, TwythonError


bucket_name = 'language-model'
model_dir = 'model'
model_filename = 'iris.pkl'

def decrypt_env_var(key_name):
    ENCRYPTED = os.environ[key_name]
    DECRYPTED = boto3.client('kms').decrypt(CiphertextBlob=b64decode(ENCRYPTED))['Plaintext']
    return DECRYPTED

CONSUMER_KEY = decrypt_env_var('CONSUMER_KEY')
CONSUMER_SECRET = decrypt_env_var('CONSUMER_SECRET')
ACCESS_TOKEN = decrypt_env_var('ACCESS_TOKEN')
ACCESS_SECRET = decrypt_env_var('ACCESS_SECRET')


class Model():
    def __init__(self, bucket_name, model_dir, model_filename):
        self.bucket_name = bucket_name
        self.model_dir = model_dir
        self.model_filename = model_filename

        # Load model
        bucket= boto3.resource('s3').Bucket(bucket_name)
        bucket.download_file(os.path.join(self.model_dir, self.model_filename), os.path.join('/tmp', self.model_filename))
        self.model = pickle.load(open(os.path.join('/tmp', self.model_filename), 'rb'))

    def predict(self, data):
        result = self.model.predict(data)
        return result


class Twitter():
    def __init__(self, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET):
        self.CONSUMER_KEY = CONSUMER_KEY
        self.CONSUMER_SECRET = CONSUMER_SECRET
        self.ACCESS_TOKEN = ACCESS_TOKEN
        self.ACCESS_SECRET = ACCESS_SECRET
        self.twitter = Twython(self.CONSUMER_KEY, self.CONSUMER_SECRET, self.ACCESS_TOKEN, self.ACCESS_SECRET)

    def update_status(self, status):
        self.twitter.update_status(status=status)


model = Model(bucket_name, model_dir, model_filename)
twitter = Twitter(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET)

def lambda_handler(event, context):
    # Extract data from event
    data = event['body']

    # Predict
    result = model.predict(data)

    # Twitter result
    try:
        twitter.update_status(status=str(result))
    except TwythonError as e:
        print(e)