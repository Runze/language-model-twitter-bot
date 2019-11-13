import subprocess
subprocess.call('rm -rf /tmp/*', shell=True)

try:
    import unzip_requirements
except ImportError:
    pass
print('Unzipped requirements.')

# Check the usage of the `/tmp` folder
subprocess.call('rm -rf /tmp/sls-py-req/tensorboard', shell=True)
subprocess.call('rm -rf /tmp/sls-py-req/tensorflow_estimator', shell=True)
subprocess.call('du -sh /tmp/sls-py-req', shell=True)
subprocess.call('ls -lt /tmp/sls-py-req', shell=True)


import json
import os
import boto3
import pickle
from base64 import b64decode
from keras.models import load_model
from model import LM
from twitter import Twitter


# Initiate the language model
# Load model
bucket_name = 'language-model'
bucket = boto3.resource('s3').Bucket(bucket_name)

model_dir = 'model'
model_filename = 'model.h5'
bucket.download_file(os.path.join(model_dir, model_filename), os.path.join('/tmp', model_filename))
model = load_model(os.path.join('/tmp', model_filename))

# Load meta data
meta_data_filename = 'meta_data.pkl'
bucket.download_file(os.path.join(model_dir, meta_data_filename), os.path.join('/tmp', meta_data_filename))
meta_data = pickle.load(open(os.path.join('/tmp', meta_data_filename), 'rb'))

lm = LM(model, meta_data)
print('Initiated the language model.')

# Decrypt access key/token of the Twitter client
def decrypt_env_var(key_name):
    ENCRYPTED = os.environ[key_name]
    DECRYPTED = boto3.client('kms').decrypt(CiphertextBlob=b64decode(ENCRYPTED))['Plaintext']
    return DECRYPTED

CONSUMER_KEY = decrypt_env_var('CONSUMER_KEY')
CONSUMER_SECRET = decrypt_env_var('CONSUMER_SECRET')
ACCESS_TOKEN = decrypt_env_var('ACCESS_TOKEN')
ACCESS_SECRET = decrypt_env_var('ACCESS_SECRET')

# Initiate the client
twitter = Twitter(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
print('Initialized the Twitter client.')

def lambda_handler(event, context):
    # Generate sequence using the language model
    seed_phrase = 'it is a truth universally acknowledged'
    gen_phrase = lm.gen_seq_w_seed(seed_phrase)
    print(gen_phrase)

    # Post on Twitter
    try:
        twitter.update_status(status=gen_phrase)
        print('Tweet posted successfully!')
    except TwythonError as e:
        print(e)