"""A Module for data-handling.

classes
-------
Data : Prepocess and store all relevant parts of a dataset
"""
import tensorflow as tf


class Data:
  """A data class to preprocess and store all relevant parts of a dataset.

  Attributes
  ----------
  categories : list
      List of labels.
  dataset_path : str
      Directory for storing files of the dataset.
  input_shape : tuple
      Dimensions of the input for the target/training models.
  n_bckgrnd_knwldg : int
      Size of the background knowledge of the attacker.
  n_categories : int
      Number of categories for the prediction.
  n_reference_models : int
      Even number of reference models.
  n_target_models : int
      Even number of target models.
  n_training_set : int
      Size of training set of target and refernce models
  n_trgt_knwldg : int
      Size of kownledge of the target.
  reference_train_images : numpy.ndarray
      Train images for the reference models.
  reference_train_labels : numpy.ndarray
      Train labels for the reference models.
  reference_train_labels_cat : numpy.ndarray
      Train labels (one-hot encoding) for the reference models.
  target_train_images : numpy.ndarray
      Train images for the target models.
  target_train_labels : numpy.ndarray
      Train labels for the target models.
  target_train_labels_cat : numpy.ndarray
      Train labels (one-hot encoding) for the training models.
  test_images : numpy.ndarray
      All test images.
  test_labels : numpy.ndarray
      All test labels.
  test_labels_cat : numpy.ndarray
      All test_labels (one-hot encoding).
  train_images : numpy.ndarray
      All train images.
  train_labels : numpy.ndarray
      All train labels.
  train_labels_cat : numpy.ndarray
      All train lables (one-hot encodind).
  """

  def __init__(self, train_images, train_labels, test_images, test_labels,
               dataset_details):
    """Create object for the prepocessing and data-handling of a new dataset.

    Parameters
    ----------
    train_images : numpy.ndarray
        Training images of the dataset.
    train_labels : numpy.ndarray
        Training labels of the dataset.
    test_images : numpy.ndarray
        Test images of the dataset.
    test_labels : numpy.ndarray
        Test labels of the dataset.
    dataset_details : dict
        Details of the dataset as dictionary:
        * n_trgt_knwldg: size of kownledge of the target
        * n_bckgrnd_knwldg: size of the background knowledge of the attacker
        * n_training_set: size of training set of target and refernce models
        * n_target_models: even number of target models
        * n_reference_models: even number of reference models
        * n_categories: number of categories for the prediction
        * input_shape: sample dimensions
    """
    self.n_trgt_knwldg = dataset_details['n_trgt_knwldg']
    self.n_bckgrnd_knwldg = dataset_details['n_bckgrnd_knwldg']
    self.n_training_set = dataset_details['n_training_set']
    self.n_target_models = dataset_details['n_target_models']
    self.n_reference_models = dataset_details['n_reference_models']
    self.n_categories = len(dataset_details['categories'])
    self.input_shape = dataset_details['input_shape']
    self.dataset_path = dataset_details['dataset_path']
    self.categories = dataset_details['categories']

    img_x, img_y, img_z = self.input_shape

    self.train_images = train_images.reshape(len(train_images), img_x, img_y,
                                             img_z).astype('float32') / 255.0
    self.train_labels = train_labels

    self.test_images = test_images.reshape(len(test_images), img_x, img_y,
                                           img_z).astype('float32') / 255.0
    self.test_labels = test_labels

    self.train_labels_cat = tf.keras.utils.to_categorical(train_labels,
                                                          self.n_categories)
    self.test_labels_cat = tf.keras.utils.to_categorical(test_labels,
                                                         self.n_categories)

    self.target_train_images = self.train_images[:self.n_trgt_knwldg]
    self.target_train_labels_cat = self.train_labels_cat[:self.n_trgt_knwldg]
    self.target_train_labels = self.train_labels[:self.n_trgt_knwldg]

    self.reference_train_images = self.train_images[self.n_trgt_knwldg:]
    self.reference_train_labels_cat = self.train_labels_cat[self.n_trgt_knwldg:
                                                            ]
    self.reference_train_labels = self.train_labels[self.n_trgt_knwldg:]
