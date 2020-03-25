"""Utiliy module with the whole functionality needed for direct mia on a cnn.

Implementation and analysis of the direct mia from:

Long, Yunhui and Bindschaedler, Vincent and Wang, Lei and Bu, Diyue and Wang,
Xiaofeng and Tang, Haixu and Gunter, Carl A and Chen, Kai (2018).
Understanding membership inferences on well-generalized learning models.
arXiv preprint arXiv:1802.04889.
"""
import os
import numpy as np
import math
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
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

SAVE_PLOTS_IN_FILE = False


def create_dir_structure(dataset_path):
  """Create directiory structure for all data produced.

  target_models and reference_models cotains all trained models and the records
  used as training sets for each model.

  high_level_features will contain all high level features obtained by the
  reference and training models from the last layer before the softmax layer.

  Parameters
  ----------
  dataset_path : str
      Directory for storing files of the dataset.
  """
  dirs = ['target_models', 'reference_models', 'high_level_features']

  for d in dirs:
    if not os.path.isdir(dataset_path + d):
      os.makedirs(dataset_path + d)


def create_model(input_shape, n_categories):
  """Architecture of the attacker and reference models.

  Parameters
  ----------
  input_shape : tuple
      Dimensions of the input for the target/training
  n_categories : int
      number of categories for the prediction
  models.

  Returns
  -------
  tensorflow.python.keras.engine.sequential.Sequential
      A convolutional neuronal network model.
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
  model.add(Dense(n_categories))
  model.add(Activation('softmax'))

  return model


def get_records_per_target_model(dataset):
  """Create training datasets for the target models.

  Each trainingsset contains exactly 50 percent of the target knowledge. Each
  record is in 50 percent of the target models. For detailed explanation see
  paper section 5.1.

  Parameters
  ----------
  dataset : data.Data
      Data object for the attacked dataset.

  Returns
  -------
  numpy.ndarray
      Describes which record is used to train which model.
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
  """Create training datasets for the reference models.

  Parameters
  ----------
  dataset : data.Data
      Data object for the attacked dataset.

  Returns
  -------
  numpy.ndarray
      Describes which record is used to train which model.
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

  Parameters
  ----------
  data : data.Data
      Data object for the attacked dataset.
  records_per_target_model : numpy.ndarray
      Describes which record is used to train which model.
  epochs : int
      Number of training epochs.
  batch_size : int
      Size of mini batches for stochatic gradient descent.
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

  Parameters
  ----------
  data : data.Data
      Data object for the attacked dataset.
  records_per_reference_model : numpy.ndarray
      Describes which record is used to train which model.
  epochs : int
      Number of training epochs.
  batch_size : int
      Size of mini batches for stochastic gradient descent.
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


def evaluate_models(models, data, batch_size):
  """Evaluate the accuray of trained models.

  Parameters
  ----------
  models : numpy.ndarray
      Array of trained models.
  data : data.Data
      Data object for the attacked dataset.
  batch_size : int
      Size of mini batches for stochastic gradient descent.

  Returns
  -------
  tuple
      mean and minimal accuracy over all models
  """
  accuracys = []
  for model in models:
    _, accuracy = model.evaluate(x=data.test_images, y=data.test_labels_cat,
                                 batch_size=batch_size, verbose=3)
    accuracys.append(accuracy)

  accuracys = np.asarray(accuracys)
  mean_acc = np.mean(accuracys)
  min_acc = np.min(accuracys)
  print('mean_acc: ', mean_acc, ' - min_acc: ', min_acc)

  return mean_acc, min_acc


def load_models(path, n_models):
  """Load trained and saved models.

  Parameters
  ----------
  path : str
      Path to saved models.
  n_models : int
      Describes how many models should be loaded.

  Returns
  -------
  numpy.ndarray
      Array of loaded models.
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

  Parameters
  ----------
  models : numpy.ndarray
      Array of models from which the intermediate models should be extracted.

  Returns
  -------
  numpy.ndarray
      Array of intermediate models.
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

  For details see paper section 4.3.

  Parameters
  ----------
  data : data.Data
      Data object for the attacked dataset.
  intermediate_models : numpy.ndarray
      Array of intermediate models.
  train_images : numpy.ndarray
      Images for which the high-level features should be generated.
  filename : str
      Filename to save the high level features.

  Returns
  -------
  numpy.ndarray
      Array of high level features. One feature for each image.
  """
  feature_vecs = np.empty((len(train_images), 0))
  for i, model in enumerate(intermediate_models):
    predictions = model.predict(train_images)
    feature_vecs = np.append(feature_vecs, predictions, axis=1)
  np.save(data.dataset_path + 'high_level_features/' + filename, feature_vecs)

  return feature_vecs


def get_model_inference(idx_records, records, labels, models):
  """Predict on trained models and calculate the log loss.

  Parameters
  ----------
  idx_records : numpy.ndarray
      Index array of records to predict on.
  records : numpy.ndarray
      Array of records used for prediction.
  labels : numpy.ndarray
      Array of labels used to predict on.
  models : numpy.ndarray
      Array of models used for prediction.

  Returns
  -------
  numpy.array
      Array of log losses of predictions.
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

  Parameters
  ----------
  high_level_features : numpy.ndarray
      Array of high dimensional high level features.
  data : data.Data
      Data object for the attacked dataset.
  """
  pca = PCA(n_components=3)
  reduced_features = pca.fit_transform(high_level_features)
  print('pca explained_variance_ratio: ', pca.explained_variance_ratio_)
  fig = px.scatter_3d(x=reduced_features[:, 0], y=reduced_features[:, 1],
                      z=reduced_features[:, 2],
                      color=data.train_labels.astype(str), width=650,
                      height=650)
  fig.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
  fig.show()


def plot3D_high_level_features_target(high_level_features_all, data,
                                      target_records_idx, colorscale):
  """Generate a 3D plot of the high level features.

  Generate 3D-plot of the high level target space with embedded high level
  features of the target records. A PCA is used to reduce the dimensions for
  the plot.

  Parameters
  ----------
  high_level_features_all : np.ndarray
      Array of high level features belonging to target and reference training
      sets.
  data : data.Data
      Data object for the attacked dataset.
  target_records_idx : np.ndarray
      Index array of target records.
  colorscale : np.ndarray
      Color array (hex) for plotting class details.
  """
  pca = PCA(n_components=3)
  reduced_features = pca.fit_transform(high_level_features_all)
  print('pca explained_variance_ratio: ', pca.explained_variance_ratio_)

  # choose all reduced high level features belonging to the target traning data
  high_level_features_target = reduced_features[:data.n_trgt_knwldg, :]

  # choose all reduced high level features belonging to the target records
  high_level_features_target_records = \
      high_level_features_target[target_records_idx, :]

  # choose all reduced high level features belonging to the reference training
  # set
  high_level_features_reference = reduced_features[data.n_trgt_knwldg:, :]

  target_records_color = data.target_train_labels[target_records_idx]
  reference_color = data.reference_train_labels

  # build figure
  fig = go.Figure()

  # add scatter trace with opaque markers for reduced high level features
  # belonging to records from the reference training set
  fig.add_trace(
      go.Scatter3d(
          mode='markers',
          opacity=0.15,
          x=high_level_features_reference[:, 0],
          y=high_level_features_reference[:, 1],
          z=high_level_features_reference[:, 2],
          marker=dict(size=2, color=reference_color,
                      colorscale=colorscale, symbol='circle'),
          hovertext=reference_color,
          showlegend=False
      )
  )

  # add scatter trace with square bold markers for reduced high level features
  # belonging to the selected target records
  fig.add_trace(
      go.Scatter3d(
          mode='markers',
          x=high_level_features_target_records[:, 0],
          y=high_level_features_target_records[:, 1],
          z=high_level_features_target_records[:, 2],
          marker=dict(size=6, color=target_records_color,
                      colorscale=colorscale, symbol='square',
                      line=dict(color='Black', width=1)),
          name='target records',
          hovertext=target_records_color,
          showlegend=False
      )
  )

  fig.show()


# !!!Buggy!!! use plot2D_high_level_features_target_plt instead
def plot2D_high_level_features_target(high_level_features_all, data,
                                      target_records_idx, dist_colorscale):
  """Generate a 2D plot of the high level features.

  Generate 2D-plot of the high level target space with embedded high level
  features of the target records. A PCA is used to reduce the dimensions for
  the plot.

  Parameters
  ----------
  high_level_features_all : np.ndarray
      Array of high level features belonging to target and reference training
      sets.
  data : data.Data
      Data object for the attacked dataset.
  target_records_idx : np.ndarray
      Index array of target records.
  colorscale : np.ndarray
      Color array (hex) for plotting class details.

  Deleted Parameters
  ------------------
  high_level_features : numpy.ndarray
      Array of high dimensional high level features.
  """
  pca = PCA(n_components=2)
  reduced_features = pca.fit_transform(high_level_features_all)
  print('pca explained_variance_ratio: ', pca.explained_variance_ratio_)

  # choose all reduced high level features belonging to the target traning data
  high_level_features_target = reduced_features[:data.n_trgt_knwldg, :]

  # choose all reduced high level features belonging to the target records
  high_level_features_target_records = \
      high_level_features_target[target_records_idx, :]

  # choose all reduced high level features belonging to the reference training
  # set
  high_level_features_reference = reduced_features[data.n_trgt_knwldg:, :]

  target_records_color = data.target_train_labels[target_records_idx]

  print(target_records_color)

  reference_color = data.reference_train_labels

  # build figure
  fig = go.Figure()

  # add scatter trace with opaque markers for reduced high level features
  # belonging to records from the reference training set
  fig.add_trace(
      go.Scatter(
          mode='markers',
          opacity=1,
          x=high_level_features_reference[:, 0],
          y=high_level_features_reference[:, 1],
          marker=dict(size=2, color=reference_color,
                      colorscale=dist_colorscale, symbol='circle'),
          name='non_target_records',
          hovertext=reference_color,
          showlegend=False
      )
  )

  # add scatter trace with square bold markers for reduced high level features
  # belonging to the selected target records
  fig.add_trace(
      go.Scatter(
          mode='markers',
          x=high_level_features_target_records[:, 0],
          y=high_level_features_target_records[:, 1],
          marker=dict(size=10, color=target_records_color,
                      colorscale=dist_colorscale, symbol='square',
                      line=dict(color='Black', width=1)),
          name='target records',
          hovertext=target_records_color,
          showlegend=False
      )
  )

  fig.show()


def plot2D_high_level_features_target_plt(high_level_features_all, data,
                                          target_records_idx,
                                          mean_distances_target,
                                          dist_colorscale, zoom_x, zoom_y,
                                          annotate):
  """Generate a 2D plot of the high level features.

  A PCA is used to reduce the dimensions for the plot.

  Parameters
  ----------
  high_level_features_all : np.ndarray
      Array of high level features belonging to target and reference training
      sets.
  data : data.Data
      Data object for the attacked dataset.
  target_records_idx : np.ndarray
      Index array of target records.
  mean_distances_target : np.ndarray
      Array of mean distances from one target record to all reference records
      in the high level feature space.
  dcolorscale : np.ndarray
      Color array (hex) for plotting class details.
  zoom_x : int
      Zoom along the x-axis for the plot.
  zoom_y : int
      Zoom along the y-axis for the plot.
  annotate : bool
      Activate annotations.
  """
  pca = PCA(n_components=2)
  reduced_features = pca.fit_transform(high_level_features_all)
  print('pca explained_variance_ratio: ', pca.explained_variance_ratio_)

  # choose all reduced high level features belonging to the target traning data
  high_level_features_target = reduced_features[:data.n_trgt_knwldg, :]

  # choose all reduced high level features belonging to the target records
  high_level_features_target_records = \
      high_level_features_target[target_records_idx, :]

  # choose all reduced high level features belonging to the reference training
  # set
  high_level_features_reference = reduced_features[data.n_trgt_knwldg:, :]

  reference_labels = data.reference_train_labels

  colours = ListedColormap(dist_colorscale)
  color_targets = dist_colorscale[data.target_train_labels[target_records_idx]]
  color_targets = color_targets.flatten()
  # create the figure and axes objects
  fig, ax = plt.subplots(1, figsize=(10, 10))

  # plot reduced high level features belonging to records from the reference
  # training set
  scatter1 = ax.scatter(x=high_level_features_reference[:, 0],
                        y=high_level_features_reference[:, 1],
                        c=reference_labels, cmap=colours, s=1)

  scatter_size = mean_distances_target
  scatter_size /= np.max(scatter_size)
  scatter_size = scatter_size ** 4 * 300

  # plot reduced high level features belonging to target records
  ax.scatter(x=high_level_features_target_records[:, 0],
             y=high_level_features_target_records[:, 1],
             color=color_targets, s=scatter_size, marker='D',
             linewidths=1, edgecolors='black')

  # Add labels for each point
  labels = np.arange(0, len(target_records_idx))
  for x_pos, y_pos, label in zip(high_level_features_target_records[:, 0],
                                 high_level_features_target_records[:, 1],
                                 labels):
    if annotate:
      t = ax.annotate(label, xy=(x_pos, y_pos), xytext=(15, 0),
                      textcoords='offset points', ha='left', va='center',
                      size=8)
      t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

  ax.legend(handles=scatter1.legend_elements()[0],
            labels=data.categories)
  ax.margins(x=zoom_x, y=zoom_y)
  plt.grid()

  if(SAVE_PLOTS_IN_FILE):
    fig.savefig('plots/' + SAVE_PLOTS_IN_FILE +
                '_high_level_features_target_zoom.pdf', bbox_inches='tight')


def calc_pairwise_distances(features_target, features_reference, data, metric,
                            n_jobs=1):
  """Calculate pairwise distances between given features.

  Parameters
  ----------
  features_target : numpy.ndarray
      First array for pairwise distances.
  features_reference : numpy.ndarray
      Second array for pairwise distances.
  data : Data.data
      Data object for the attacked dataset.
  metric : str
      Metric used for the distance calculations.
  n_jobs : int, optional
      Number of parallel computation jobs.

  Returns
  -------
  numpy.ndarray
      The distance between features_target[i] and features_reference[j] is
      saved in distances[i][j].
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
  sets. For details see section 4.3 from the paper.

  Parameters
  ----------
  neighbor_threshold : float
      If distance is smaller then the neighbor threshold the record is selected
      as target record.
  probability_threshold : float
      For details see section 4.3 from the paper.
  data : data.Data
      Data object for the attacked dataset.
  distances : numpy.ndarray
      Distance array used for target selection.

  Returns
  -------
  numpy.array
      Selected target records
  """
  print('min_distance: ', np.min(distances))
  if(np.min(distances) >= neighbor_threshold):
    print('neighbor_threshold is smaller then all distances!')

  n_neighbors = np.count_nonzero(distances < neighbor_threshold, axis=1)
  print('mean n_neighbors: ', np.mean(n_neighbors))

  est_n_neighbors = n_neighbors * (data.n_trgt_knwldg / data.n_bckgrnd_knwldg)
  print('mean est_n_neighbors: ', np.mean(est_n_neighbors))

  target_records = np.where(est_n_neighbors < probability_threshold)[0]

  mean_distances_target = np.mean(distances[target_records], axis=1)

  print('number of target_records: ', len(target_records))
  print('target_records: ', target_records)
  print('number of neighbors: ', n_neighbors[target_records])
  print('mean distances of target records to records of the reference' +
        'training set:', mean_distances_target)

  return target_records, mean_distances_target


def plot_target_records(target_records, data):
  """Plot target records to get a understanding of our selection algorithm.

  Parameters
  ----------
  target_records : numpy.ndarray
      Selected target records which should be plotted.
  data : data.Data
      Data object for the attacked dataset.
  """
  rows = math.ceil(len(target_records) / 3)
  plt.figure(figsize=[15, rows * 4])
  for idx, target_record in enumerate(target_records):
    title = 'r=' + str(target_record) + ' label=' \
            + str(data.target_train_labels[target_record])

    plt.subplot(rows, 3, idx + 1, title=title)

    input_shape = data.input_shape
    if(input_shape[2] <= 1):
      re_shape = (input_shape[0], input_shape[1])
    else:
      re_shape = (input_shape[0], input_shape[1], input_shape[2])

    plt.imshow(data.target_train_images[target_record, :, :].reshape(re_shape))

  if(SAVE_PLOTS_IN_FILE):
    plt.savefig('plots/' + SAVE_PLOTS_IN_FILE + '_target_records.pdf',
                bbox_inches='tight')


def sample_reference_losses(target_records, reference_inferences):
  """Sample reference log losses.

  Sample the log losses of a record regarding its label. Estimate the CDF of
  this samples and smooth the estimated CDF with the shape-preserving
  piecewise cubic interpolation. For details see section 4.4 from the paper.

  Parameters
  ----------
  target_records : numpy.ndarray
      Array of target records for sampling the reference log losses.
  reference_inferences : numpy.ndarray
      Array of log losses of the predictions on the reference models.

  Returns
  -------
  tuple
      successfully used target records and smoothed ecdf
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
    except ValueError:
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

  if(SAVE_PLOTS_IN_FILE):
    plt.savefig('plots/' + SAVE_PLOTS_IN_FILE + '_reference_log_losses.pdf',
                bbox_inches='tight')

  used_target_records = np.asarray(used_target_records)
  print('number of user target records: ' + str(len(used_target_records)))
  print(used_target_records)
  return used_target_records, pchip_references


def hypothesis_test(data, records_per_target_model, target_records,
                    cut_off_p_value, pchip_references, target_inferences,
                    mean_distances_target):
  """Left-tailed hypothesis test.

  Parameters
  ----------
  data : data.Data
      Data object for the attacked dataset.
  records_per_target_model : numpy.ndarray
      Describes which record is used to train which target model.
  target_records : numpy.ndarray
      Target records finally used for the attack.
  cut_off_p_value : float
      Level of significance used for the hypothesis test.
  pchip_references : list
      Interpolated ecdfs of smapled log losses
  target_inferences : numpy.array
      Array of log losses of the predictions on the target models.
  mean_distances_target : np.ndarray
      Array of mean distances from one target record to all reference records
      in the high level feature space.

  Returns
  -------
  list
      P-values of the hypothesis tests.
  """
  ground_truth = np.zeros((data.n_trgt_knwldg, data.n_target_models))
  for i in range(data.n_target_models):
    ground_truth[records_per_target_model[i, :], i] = 1

  p_values = []
  for idx in range(len(target_records)):
    p_values.append(pchip_references[idx](target_inferences[idx, :]))

  sum_precision = 0
  sum_recall = 0
  sum_tp = 0
  sum_fp = 0

  precision_list = []
  recall_list = []
  successfull_attacked_targets_idx = []

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
      successfull_attacked_targets_idx.append(idx)
      precision = tp / (fp + tp)
      recall = tp / (fn + tp)
      sum_precision += precision
      sum_recall += recall
      sum_tp += tp
      sum_fp += fp
      print('precision: ', precision)
      print('recall: ', recall)
      print('mean distance to records from reference dataset: ',
            mean_distances_target[idx])

      precision_list.append([precision])
      recall_list.append([recall])

    print('\n')

  n_attacks_successfull = len(successfull_attacked_targets_idx)
  print('n_attacks_successfull: ', n_attacks_successfull)

  if(n_attacks_successfull):
    print('precsion over all target_records: ',
          sum_precision / n_attacks_successfull)
    print('recall over all target_records: ',
          sum_recall / n_attacks_successfull)
    print('true positives over all target_records: ', sum_tp)
    print('false positives over all target_records: ', sum_fp)

    scatter_size = mean_distances_target[successfull_attacked_targets_idx]
    scatter_size /= np.max(scatter_size)
    scatter_size = scatter_size ** 6 * 400

    plt.scatter(precision_list, recall_list,
                s=scatter_size)
    plt.title('precision-recall scatter plot')

    if(SAVE_PLOTS_IN_FILE):
      plt.savefig('plots/' + SAVE_PLOTS_IN_FILE +
                  '_precision_recall_scatter.pdf', bbox_inches='tight')

    return p_values
