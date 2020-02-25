import tensorflow as tf
class Data:
  def __init__(self, train_images, train_labels, test_images, test_labels, dataset_details):
    self.n_trgt_knwldg = dataset_details['n_trgt_knwldg']
    self.n_bckgrnd_knwldg = dataset_details['n_bckgrnd_knwldg']
    self.n_training_set = dataset_details['n_training_set']
    self.n_target_models = dataset_details['n_target_models']
    self.n_reference_models = dataset_details['n_reference_models']
    self.n_categories = dataset_details['n_categories']
    self.input_shape = dataset_details['input_shape']
    self.dataset_path = dataset_details['dataset_path']

    img_x, img_y, img_z = self.input_shape

    self.train_images = train_images.reshape(len(train_images), img_x, img_y, img_z).astype('float32') / 255.0
    self.train_labels = train_labels

    self.test_images = test_images.reshape(len(test_images), img_x, img_y, img_z).astype('float32') / 255.0
    self.test_labels = test_labels

    self.train_labels_cat = tf.keras.utils.to_categorical(train_labels, self.n_categories)
    self.test_labels_cat = tf.keras.utils.to_categorical(test_labels, self.n_categories)

    self.target_train_images = self.train_images[:self.n_trgt_knwldg]
    self.target_train_labels_cat = self.train_labels_cat[:self.n_trgt_knwldg]
    self.target_train_labels = self.train_labels[:self.n_trgt_knwldg]

    self.reference_train_images = self.train_images[self.n_trgt_knwldg:]
    self.reference_train_labels_cat = self.train_labels_cat[self.n_trgt_knwldg:]
    self.reference_train_labels = self.train_labels[self.n_trgt_knwldg:]