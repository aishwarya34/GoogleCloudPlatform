#tf.estimator modeling

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import shutil
from google.datalab.ml import TensorBoard

print(tf.__version__)
tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = ['accepted', 'answer_count', 'comment_count', 'favorite_count', 'score', 'view_count', 'days_posted']
DEFAULTS = [[0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

#DEFAULTS = [tf.constant([0], dtype=tf.int32),
#            tf.constant([0.0], dtype=tf.float32),
#            tf.constant([0.0], dtype=tf.float32),
#           tf.constant([0.0], dtype=tf.float32),
#            tf.constant([0.0], dtype=tf.float32),
#            tf.constant([0.0], dtype=tf.float32),
#            tf.constant([0.0], dtype=tf.float32) ]

#i=0
def read_dataset(filename, mode, batch_size = 512):
    def decode_line(row):
        #print(row)
        cols = tf.decode_csv(row, record_defaults = DEFAULTS)
        #print(cols)
        features = dict(zip(CSV_COLUMNS,cols))
        #print(i+1)
        label = features.pop('accepted')  # remove label from features and store
        #print("features: {} \n label: {}".format(features, label))
        return features, label
  
    # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
    filenames_dataset = tf.data.Dataset.list_files(filename, shuffle=False)
    # Read lines from text files
    #textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset).skip(1)
    textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
    # Parse text lines as comma-separated values (CSV)
    dataset = textlines_dataset.map(decode_line)
  
  # Note:
  # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
  # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)
  
    if(mode == tf.estimator.ModeKeys.TRAIN):
        num_epochs = 10  # loop indefinitely
        dataset = dataset.shuffle(buffer_size = 10*batch_size, seed=2)
    else:
        num_epochs = 1
  
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    return dataset

  
def get_train_input_fn(folder_name):
    dataset = read_dataset(folder_name + '/stackoverflow-train-*.csv', tf.estimator.ModeKeys.TRAIN)
    features1, label1 = dataset.make_one_shot_iterator().get_next()
    #print("Training set :  \nfeatures1 : {}\nlabel: {}".format(features1, label1))
    with tf.Session() as sess:
        print(sess.run(tf.shape(label1))) # output: [ 0.42116176  0.40666069]
    return features1, label1 

def get_valid_input_fn(folder_name):
    dataset = read_dataset(folder_name + '/stackoverflow-valid-*.csv', tf.estimator.ModeKeys.EVAL)
    features1, label1 = dataset.make_one_shot_iterator().get_next()
    return features1, label1 

def get_test_input_fn(folder_name):
    dataset = read_dataset(folder_name + '/stackoverflow-test-*.csv', tf.estimator.ModeKeys.PREDICT)
    features1, label1 = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        print(sess.run(tf.shape(label1))) # output: [ 0.42116176  0.40666069]
    return features1, label1 


#get_train_input_fn()
FEATURE_NAMES = CSV_COLUMNS[1:]
LABEL_NAME = CSV_COLUMNS[0]

featcols = [ tf.feature_column.numeric_column(feat) for feat in  FEATURE_NAMES ]
#print(featcols)

def serving_input_fn():
  
    json_features_placeholder = {
        'answer_count' : tf.placeholder(tf.float32, [None]), #Batch size
        'comment_count' : tf.placeholder(tf.float32, [None]), 
        'favorite_count' : tf.placeholder(tf.float32, [None]), 
        'score' : tf.placeholder(tf.float32, [None]), 
        'view_count' : tf.placeholder(tf.float32, [None]),  
        'days_posted' : tf.placeholder(tf.float32, [None])
    }
  
    features = json_features_placeholder
  
    return tf.estimator.export.ServingInputReceiver(features, json_features_placeholder)

## Create train and evaluate function using tf.estimator
def train_and_evaluate(args):
    tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
    
    run_config = tf.estimator.RunConfig(model_dir = args['output_dir'], save_summary_steps = 100, save_checkpoints_steps = 1000)
  
    estimator = tf.estimator.DNNClassifier(
        hidden_units =  args['hidden_units'],    #[1024, 512, 128, 32],  # specify neural architecture
        feature_columns = featcols,
        n_classes=2,
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
        #model_dir = OUTDIR,
        config = run_config 
    )
  
    train_spec = tf.estimator.TrainSpec(input_fn = lambda : get_train_input_fn(args['train_data_paths']), max_steps = args['train_steps'])
  
    exporter_latest =  tf.estimator.LatestExporter('exporter', serving_input_receiver_fn = serving_input_fn)
  
    eval_spec = tf.estimator.EvalSpec(input_fn = lambda : get_valid_input_fn(args['eval_data_paths']), 
                                   steps = None,
                                   start_delay_secs = args['eval_delay_secs'], # start evaluating after N seconds
                                   throttle_secs = args['throttle_secs'],   # evaluate every N seconds
                                   exporters = exporter_latest)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)