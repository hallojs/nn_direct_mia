import os
import numpy as np
import math
import plotly.express as px
import matplotlib.pyplot as plt
# for the empirical cdf
from statsmodels.distributions.empirical_distribution import ECDF
# for the interpolation
from scipy.interpolate import pchip

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

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


def create_dir_structure(dataset_path):
  """Create directiory structure for all data produced.

  target_models and reference_mdoels cotains all trained models and the records
  used as training sets for each model.

  high_level_features will contain all high level features obtained by the
  reference and training models from the last layer before the Softmax layer.
  """
  dirs = ['target_models', 'reference_models', 'high_level_features']

  for d in dirs:
    if not os.path.isdir(dataset_path + d):
      os.makedirs(dataset_path + d)


def create_model(input_shape, classes):
  """Architecture of the attacker and reference models.
  """
  model = Sequential()

  # first connvolution layer
  model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(
      1, 1), padding='same', input_shape=input_shape))
  model.add(Activation('relu'))

  # max pooling layer
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  # second convolution layer
  model.add(Conv2D(filters=64, kernel_size=(
      5, 5), strides=(1, 1), padding='same'))
  model.add(Activation('relu'))

  # max pooling layer
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  # fully connected layer
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
  """Create training datasets for the target models.

  Each trainingsset contains exactly 50 percent of the target knowledge. Each
  record is in 50% of the target models.
  """
  records_per_target_model = np.array([])
  for i in range(0, int(dataset.n_target_models / 2)):
    np.random.seed(i)
    selection = np.random.choice(
        np.arange(dataset.n_trgt_knwldg), dataset.n_trgt_knwldg, replace=False)
    if(i > 0):
      records_per_target_model = np.vstack(
          (records_per_target_model, selection[:dataset.n_training_set]))
      records_per_target_model = np.vstack(
          (records_per_target_model, selection[dataset.n_training_set:]))
    else:
      records_per_target_model = np.vstack(
          (selection[:dataset.n_training_set],
              selection[dataset.n_training_set:]))

  path = dataset.dataset_path + 'target_models/records_per_target_model.csv'
  np.savetxt(path, records_per_target_model, delimiter=",")

  return records_per_target_model


def get_records_per_reference_model(dataset):
  """Create training datasets for the reference models using the Bootstrap
  Method.
  """
  records_per_reference_model = np.array([])
  for i in range(dataset.n_reference_models):
    np.random.seed(i)
    # sampling
    idx = np.random.choice(dataset.n_bckgrnd_knwldg,
                           dataset.n_training_set, replace=False)
    if i > 0:
      records_per_reference_model = np.append(
          records_per_reference_model, [idx], axis=0)
    else:
      records_per_reference_model = [idx]

  path = dataset.dataset_path \
      + 'reference_models/records_per_reference_model.csv'
  np.savetxt(path, records_per_reference_model, delimiter=",")

  return records_per_reference_model


def train_target_models(data, records_per_target_model, epochs, batch_size):
  """Train the target models using the adam optimizer.
  """
  for idx, records in enumerate(records_per_target_model):
    print('Train model ', str(idx))
    model = create_model(data.input_shape, 10)
    model.layers[10]._name = 'intermediate_layer'
    # model.summary()
    adam = optimizers.Adam(lr=0.0001)
    model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(data.target_train_images[records],
              data.target_train_labels_cat[records], epochs=epochs,
              batch_size=batch_size, verbose=0)
    model.save(data.dataset_path + 'target_models/model' + str(idx) + '.h5')


def train_reference_models(data, records_per_reference_model,
                           epochs, batch_size):
  """Train the reference models using the adam optimizer.
  """
  for idx, records in enumerate(records_per_reference_model):
    print('Train model ', str(idx))
    model = create_model(data.input_shape, 10)
    model.layers[10]._name = 'intermediate_layer'
    adam = optimizers.Adam(lr=0.0001)
    model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(data.reference_train_images[records],
              data.reference_train_labels_cat[records], epochs=epochs,
              batch_size=batch_size, verbose=0)
    model.save(data.dataset_path + 'reference_models/model' + str(idx) + '.h5')


def evaluate_models(models, data, n_batches):
  """Evaluate the accuray of trained models.
  """
  accuracys = []
  for model in models:
    _, accuracy = model.evaluate(x=data.test_images, y=data.test_labels_cat,
                                 batch_size=n_batches, verbose=3)
    accuracys.append(accuracy)

  accuracys = np.asarray(accuracys)
  mean_acc = np.mean(accuracys)
  min_acc = np.min(accuracys)
  print('mean_acc: ', mean_acc, ' - min_acc: ', min_acc)

  return mean_acc, min_acc


def load_models(path, n_models):
  """Load trained and saved models.
  """
  all_models = np.array([])
  for i in range(n_models):
    print('load ' + path + 'model' + str(i) + '.h5')
    model = models.load_model(path + 'model' + str(i) + '.h5')
    all_models = np.append(all_models, model)

  return all_models


def gen_intermediate_models(models):
  """Generate intermediate models.

  This intermediate models are used later to extract the high level features of
  the target and refernce models.
  """
  intermediate_models = np.array([])
  for i, model in enumerate(models):
    # https://androidkt.com/get-output-of-intermediate-layers-keras/
    layer_output = model.get_layer('intermediate_layer').output
    intermediate_models = np.append(intermediate_models, Model(
        inputs=model.input, outputs=layer_output))

  return intermediate_models


def gen_high_level_features(data, intermediate_models, train_images, filename):
  """Extract the high level features from the intermediate models.
  """
  feature_vecs = np.empty((len(train_images), 0))
  for i, model in enumerate(intermediate_models):
    predictions = model.predict(train_images)
    feature_vecs = np.append(feature_vecs, predictions, axis=1)
  np.save(data.dataset_path + 'high_level_features/' + filename, feature_vecs)

  return feature_vecs


def get_model_inference(idx_records, records, labels, models):
  """Predict on trained models.
  """
  inferences = []
  for record in idx_records:
    predictions = np.array([])
    for model in models:
      label = labels[record]
      prediction = model.predict(records[record:record + 1])[0][label]
      predictions = np.append(predictions, prediction)
    inference = -np.log(predictions)
    inferences.append(inference)
  inferences = np.asarray(inferences)

  return inferences


def plot_high_level_features(high_level_features, data):
  """Generate a 3D plot of the high level features.

  A PCA is used to reduce the dimensions for the plot.
  """
  pca = PCA(n_components=3)
  reduced_features = pca.fit_transform(high_level_features)
  print('pca explained_variance_ratio: ', pca.explained_variance_ratio_)
  fig = px.scatter_3d(x=reduced_features[:, 0], y=reduced_features[:, 1],
                      z=reduced_features[:, 2],
                      color=data.train_labels.astype(str), width=750,
                      height=750)
  fig.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
  fig.show()


def calc_pairwise_distances(features_target, features_reference, data, metric,
                            n_jobs=1):
  """Calculate pairwise distances between given features.

  The distance between features_target[i] and features_reference[j] is saved in
  distances[i][j].
  """
  distances = pairwise_distances(
      features_target, features_reference, metric=metric, n_jobs=n_jobs)
  np.save(data.dataset_path + 'high_level_features/pairwise_distances_' +
          metric + '.npy', distances)

  return distances


def select_target_records(neighbor_threshold, probability_threshold, data,
                          distances):
  """Select vulnerable target records.

  A record is selected as target record if it has few neighbours regarding its
  high level features. We estimate the number of neighbours of a record in the
  target training set over the number of neighbours in the reference training
  sets.
  """
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
  """Plot target records to get a understanding of our selection algorithm.
  """
  rows = math.ceil(len(target_records) / 3)
  plt.figure(figsize=[15, rows * 3])
  for idx, target_record in enumerate(target_records):
    title = 'r=' + str(target_record)
    plt.subplot(rows, 3, idx + 1, title=title)
    plt.imshow(data.target_train_images[target_record, :, :].reshape(
        (input_shape[0], input_shape[1])))


def sample_reference_losses(target_records, reference_inferences):
  """Sample reference log losses.

  Sample the log losses of a record regarding its label. Estimate the CDF of
  this samples and smooth the estimated CDF with the shape-preserving
  piecewise cubic interpolation.
  """
  rows = math.ceil(len(target_records) / 3)
  plt.figure(figsize=[15, rows * 4])
  # empirical cdf
  ecdf_references = []
  # piecewise cubic interpolation
  pchip_references = []

  cnt = 0
  used_target_records = []

  for idx in range(len(target_records)):
    ecdf_val = ECDF(reference_inferences[idx, :])
    ecdf_references.append(ecdf_val)

    try:
      pchip_val = pchip(ecdf_val.x[1:], ecdf_val.y[1:])
    except:
      continue

    used_target_records.append(target_records[idx])
    pchip_references.append(pchip_val)
    max_x = np.max(ecdf_val.x[1:])
    min_x = np.min(ecdf_val.y[1:])
    x = np.linspace(min_x, max_x, 1000)

    title = 'Empirical CDF of $\mathcal{D}(L)$, with $r=$' + \
        str(target_records[idx])
    plt.subplot(rows, 3, cnt + 1, title=title)
    plt.plot(ecdf_val.x, ecdf_val.y, color='green',
             linewidth=3, label='emprical cdf')
    plt.plot(x, pchip_val(x), color='red',
             linestyle='dotted', linewidth=3, label='pchip')
    plt.legend()
    cnt += 1
  plt.show()

  used_target_records = np.asarray(used_target_records)
  return used_target_records, pchip_references


def hypothesis_test(data, records_per_target_model, target_records,
                    cut_off_p_value, pchip_references, target_inferences):
  """Left-tailed hypothesis test.
  """
  ground_truth = np.zeros((data.n_trgt_knwldg, data.n_target_models))
  for i in range(data.n_target_models):
    ground_truth[records_per_target_model[i, :], i] = 1

  p_values = []
  for idx in range(len(target_records)):
    p_values.append(pchip_references[idx](target_inferences[idx, :]))

  n_attacks_successfull = 0
  sum_precision = 0
  sum_recall = 0
  sum_tp = 0
  sum_fp = 0
  for idx, target_record in enumerate(target_records):
    fn = 0
    tn = 0
    fp = 0
    tp = 0
    for i in range(0, data.n_target_models):
      hpt = p_values[idx][i]
      gt = ground_truth[target_record, i]
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
    print('precsion over all target_records: ',
          sum_precision / n_attacks_successfull)
    print('recall over all target_records: ',
          sum_recall / n_attacks_successfull)
    print('true positives over all target_records: ', sum_tp)
    print('false positives over all target_records: ', sum_fp)

    return p_values
