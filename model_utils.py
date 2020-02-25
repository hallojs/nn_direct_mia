import os

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

import numpy as np

import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import math

import matplotlib.pyplot as plt

# for the empirical cdf
from statsmodels.distributions.empirical_distribution import ECDF
# for the interpolation
from scipy.interpolate import pchip


def create_dir_structure(dataset_path):
  dirs = ['target_models', 'reference_models', 'high_level_features']

  for d in dirs:
    if not os.path.isdir(dataset_path + d):
      os.makedirs(dataset_path + d)


def create_model(input_shape, classes):
  #https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/
  model = Sequential()

  # first connvolution layer
  # https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
  model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding='same', input_shape=input_shape))
  model.add(Activation('relu'))

  # max pooling layer
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

  # second convolution layer
  model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same'))
  model.add(Activation('relu'))

  # max pooling layer
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

  # fully connected layer
  # https://missinglink.ai/guides/keras/using-keras-flatten-operation-cnn-models-code-examples/
  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation('relu'))

  # drop out
  model.add(Dropout(rate=0.5))

  # fully connected layer
  model.add(Dense(classes))
  model.add(Activation('softmax'))

  return model


def get_records_per_target_model(dataset):
  records_per_target_model = np.array([])
  for i in range(0, int(dataset.n_target_models / 2)):
    np.random.seed(i)
    selection = np.random.choice(np.arange(dataset.n_trgt_knwldg), dataset.n_trgt_knwldg, replace=False)
    if(i > 0):
      records_per_target_model = np.vstack((records_per_target_model, selection[:dataset.n_training_set]))
      records_per_target_model = np.vstack((records_per_target_model, selection[dataset.n_training_set:]))
    else:
      records_per_target_model = np.vstack((selection[:dataset.n_training_set], selection[dataset.n_training_set:]))
  path = dataset.dataset_path + 'target_models/records_per_target_model.csv'
  np.savetxt(path, records_per_target_model, delimiter=",")

  return records_per_target_model


def get_records_per_reference_model(dataset):
  records_per_reference_model = np.array([])
  for i in range(dataset.n_reference_models):
    np.random.seed(i)
    # sampling
    idx = np.random.choice(dataset.n_bckgrnd_knwldg, dataset.n_training_set, replace=False)
    if i > 0:
      records_per_reference_model = np.append(records_per_reference_model, [idx], axis=0)
    else:
      records_per_reference_model = [idx]
  path = dataset.dataset_path + 'reference_models/records_per_reference_model.csv'
  np.savetxt(path, records_per_reference_model, delimiter=",")

  return records_per_reference_model


def train_target_models(data, records_per_target_model, epochs, batch_size):
  for idx, records in enumerate(records_per_target_model):
    print('Train model ', str(idx))
    model = create_model(data.input_shape, 10)
    model.layers[10]._name = 'intermediate_layer'
    #model.summary()
    adam = optimizers.Adam(lr=0.0001)
    model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(data.target_train_images[records], data.target_train_labels_cat[records], epochs=epochs, 
                      batch_size=batch_size, verbose=0)#, validation_data=(test_images, test_labels))
    model.save(data.dataset_path + 'target_models/model' + str(idx) + '.h5')


def train_reference_models(data, records_per_reference_model, epochs, batch_size):
  for idx, records in enumerate(records_per_reference_model):
    print('Train model ', str(idx))
    model = create_model(data.input_shape, 10)
    model.layers[10]._name = 'intermediate_layer'
    adam = optimizers.Adam(lr=0.0001)
    model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(data.reference_train_images[records], data.reference_train_labels_cat[records], epochs=epochs, 
                    batch_size=batch_size, verbose=0)#, validation_data=(test_images, test_labels))
    model.save(data.dataset_path + 'reference_models/model' + str(idx) + '.h5')


def evaluate_models(models, data, n_batches):
    accuracys = []
    for model in models:
        _, accuracy = model.evaluate(x=data.test_images, y=data.test_labels_cat, batch_size=n_batches, verbose=3)
        accuracys.append(accuracy)
    accuracys = np.asarray(accuracys)
    mean_acc = np.mean(accuracys)
    min_acc = np.min(accuracys)
    print('mean_acc: ', mean_acc, ' - min_acc: ', min_acc)
    
    return mean_acc, min_acc
    
    
def load_models(path, n_models):
  all_models = np.array([])
  for i in range(n_models):
    print('load ' + path + 'model' + str(i) + '.h5')
    model = models.load_model(path + 'model' + str(i) + '.h5')
    all_models = np.append(all_models, model)

  return all_models


def gen_intermediate_models(models):
  intermediate_models = np.array([])
  for i, model in enumerate(models):
    # https://androidkt.com/get-output-of-intermediate-layers-keras/
    layer_output = model.get_layer('intermediate_layer').output
    intermediate_models = np.append(intermediate_models, Model(inputs=model.input, outputs=layer_output))

  return intermediate_models


def gen_high_level_features(data, intermediate_models, train_images, filename):
  feature_vecs = np.empty((len(train_images), 0))
  for i, model in enumerate(intermediate_models):
    predictions = model.predict(train_images)
    feature_vecs = np.append(feature_vecs, predictions, axis=1)
  np.savetxt(data.dataset_path + 'high_level_features/' + filename, feature_vecs, delimiter=',')

  return feature_vecs


def get_model_inference(idx_records, records, labels, models):
  inferences = []
  for record in idx_records:
    predictions = np.array([])
    for model in models:
      label = labels[record]
      prediction = model.predict(records[record:record+1])[0][label]
      predictions = np.append(predictions, prediction)
    inference = -np.log(predictions)
    inferences.append(inference)
  inferences = np.asarray(inferences)
  
  return inferences


def plot_high_level_features(high_level_features, data):
  pca = PCA(n_components=3)
  reduced_features = pca.fit_transform(high_level_features)
  print('pca explained_variance_ratio: ', pca.explained_variance_ratio_)
  fig = px.scatter_3d(x=reduced_features[:,0], y=reduced_features[:,1], 
                    z=reduced_features[:,2], 
                    color=data.train_labels.astype(str), width=750, height=750)
  fig.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
  fig.show()


def select_target_records(features_target, features_reference, 
                          neighbor_threshold, probability_threshold, metric,
                          data, n_jobs):
  # calculate cosine distances
  distances = pairwise_distances(features_target, features_reference, metric=metric, n_jobs=n_jobs)
  print('min_distance: ', np.min(distances))
  if(np.min(distances) >= neighbor_threshold):
    print('neighbor_threshold is smaller then all distances!')
  n_neighbors = np.count_nonzero(distances < neighbor_threshold, axis=1)
  print('mean n_neighbors: ', np.mean(n_neighbors))
  est_n_neighbors = n_neighbors * (data.n_trgt_knwldg / data.n_bckgrnd_knwldg)
  print('mean est_n_neighbors: ', np.mean(est_n_neighbors))
  target_records = np.where(est_n_neighbors < probability_threshold)[0]

  return target_records


def plot_target_records(target_records, input_shape, data):
  rows = math.ceil(len(target_records) / 3)
  fig = plt.figure(figsize=[15, rows * 3])
  for idx, target_record in enumerate(target_records):
    title = 'r=' + str(target_record)
    plt.subplot(rows, 3 ,idx+1, title=title)
    plt.imshow(data.target_train_images[target_record,:,:].reshape((input_shape[0], input_shape[1])))


def sample_reference_losses(target_records, reference_inferences):
  rows = math.ceil(len(target_records) / 3)
  # empirical cdf
  ecdf_references = []
  # piecewise cubic interpolation
  pchip_references = []
  fig = plt.figure(figsize=[15, rows * 4])
  for idx in range(len(target_records)):
    ecdf_val = ECDF(reference_inferences[idx,:])
    ecdf_references.append(ecdf_val)
    pchip_val = pchip(ecdf_val.x[1:], ecdf_val.y[1:])
    pchip_references.append(pchip_val)
    max_x = np.max(ecdf_val.x[1:])
    min_x = np.min(ecdf_val.y[1:])
    x = np.linspace(min_x,max_x,100)

    title = 'Empirical CDF of $\mathcal{D}(L)$, with $r=$' + str(target_records[idx])
    plt.subplot(rows, 3 ,idx+1, title=title)
    plt.plot(ecdf_val.x, ecdf_val.y, color='green', linewidth=3, label='emprical cdf')
    plt.plot(x, pchip_val(x), color='red', linestyle='dotted', linewidth=3, label='pchip')
    plt.legend()
  plt.show()

  return pchip_references


def hypothesis_test(data, records_per_target_model, target_records, cut_off_p_value, pchip_references, target_inferences):
  # create ground truth array
  ground_truth = np.zeros((data.n_trgt_knwldg, data.n_target_models))
  for i in range(data.n_target_models):
    ground_truth[records_per_target_model[i,:], i] = 1

  # left-tailed hypothesis test
  p_values = []
  for idx in range(len(target_records)):
    p_values.append(pchip_references[idx](target_inferences[idx,:]))

  n_attacks_successfull = 0
  sum_precision = 0
  sum_recall = 0
  sum_tp = 0
  sum_fp = 0
  for idx, target_record in enumerate(target_records):
    fn = 0;
    tn = 0;
    fp = 0;
    tp = 0;
    for i in range(0, data.n_target_models):
      hpt = p_values[idx][i]
      gt = ground_truth[target_record,i]
      if(hpt >= cut_off_p_value and gt == 1):
        fn += 1
      if(hpt >= cut_off_p_value and gt == 0):
        tn += 1
      if(hpt < cut_off_p_value and gt == 0):
        fp += 1
      if(hpt < cut_off_p_value and gt == 1):
        tp += 1

    print('target_record: ', target_record)
    print('fn: ', fn, 'tn: ', tn, 'fp: ', fp, 'tp: ', tp)

    if(tp > 0):
      n_attacks_successfull += 1
      precision = tp / (fp + tp)
      recall = tp / (fn + tp)
      sum_precision += precision
      sum_recall += recall
      sum_tp += tp
      sum_fp += fp
      print('precision: ', precision)
      print('recall: ', recall)

    print('\n')
  if(n_attacks_successfull > 0):
    print('precsion over all target_records: ', sum_precision / n_attacks_successfull)
    print('recall over all target_records: ', sum_recall / n_attacks_successfull)
    print('true positives over all target_records: ', sum_tp)
    print('false positives over all target_records: ', sum_fp)

    return p_values
  
def test():
  return 'hallo!!'