# Deep Face Registration

Based on our survey, we could not find a dataset used for face registration problem. Therefore, we had to build our face registration dataset based on one of the existing state-of-the-art datasets. We decided to go with [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset.

In this research, we built the system using [Tensorflow](https://www.tensorflow.org/guide/keras) using deep learning. The model has an accuracy of 98.55% on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.


## A) Obtaing the Dataset based on the LFW

   LFW data set contains more than 13,000 images of faces collected from the web. Each face has been labeled with the name of the person pictured. 1680 of the people pictured have two or more distinct photos in the data set. The only constraint on these faces is that they were detected by the Viola-Jones face detector. The images are in color scale, and the size is 250x250 pixels with variant expressions, timing, pose, and gender.. Figure 1 shows a sample of the LFW dataset.

![](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/LFW.png)
   
   Figure 1 Sample images of LFW dataset.
   
-All images resized to 224x224.

-We detected the faces in each image then we eliminated any image has more than one face.

-We detected the eyes and we eliminated any image has only one eye detected.

-The new dataset will contain the images as an input and the Rotation, scaling and X_shifting and y_shifting as output.



#### Obtaing the Dataset steps:

###### First: 
We selected one of the LFW images as a reference image for all other images in the dataset. The reference image is centered and has a frontal face with the assumption that, no any transformation (rotation, scaling and shifting) applied to the reference image. Figure 2 shows our reference image which we used to find the transformation parameters.

![](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/Refrenace%20Image.png)

   Figure 2. Refrenace Image.
   
 ###### next: 

We applied the haar-cascade face detection algorithm on the reference image to find the face boundary. Then, we detected the 6 facial landmarks associated with each eye based on the haar-cascade algorithm as it is shown in Figure 3.  In addition to the reference image, we apply the same method to all images in the dataset. Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person), and then working clockwise around the remainder of the region. The total of points we will use to calculate the transformation parameter is 12 points, and we will eliminate any image has less than that from the training dataset.

![](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/The%206%20facial%20landmarks%20associated%20with%20the%20eye.jpg)

Figure 3. The 6 facial landmarks associated with the eye.

###### next: 

To find the transformation parameters for an image, we passed the 12 points of the reference image and the 12 points of the target image to the minimized cost function method which we described below. Minimized cost function will return the transformation parameters (Rotation, Scaling shifting). We obtained our training dataset by applying the same method on all images.


#### Minimized Cost Function

Matrix equation (1) is used to minimize the cost of the image registration (without shear) when the related points in two images X and Y are identified. Summation indicates the sum over all points in an image.  We used in our implementation the sigmoidal function as the neuron activation functions.

![](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/eq1.png)          (equation 1)


To find the optimal transformation that will align image 2 to image 1, take the partial derivatives of the above cost with respect to a, b, t1 and t2 and set these to 0 (∂C/∂a = 0, ∂C/∂b = 0, ∂C/∂t1 = 0, ∂C/∂t2 = 0). We express the four resulting equations in matrix form as is shown in equation (2).

![](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/eq2.png)          (equation 2)

After calculating the sum over all points for the 4x4 matrix and the right-hand 4x1 vector in equation (2), we can compute the required transformation by: 
       Matrix Ainv = A.Inverse and  Matrix Res = Ainv * B.
       
       
### The Data will be saved in two numpy files. One contains the input images [Input data]() and the second file contains the image name and the 4 output values [Output data](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/outputValues224withTheImageName.zip). 


## B) Deep Face Registration System:

- We used the [Google Colab system](https://colab.research.google.com) to tain our network. 
- We uploaded our code and the [Input data]() and the [Output data](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/outputValues224withTheImageName.zip) files to the google drive. 


### import the necessary packages:

```python
from __future__ import print_function
import os.path  
import glob
import os
import numpy as np
from matplotlib import pyplot as plt 
import math
import argparse
import cv2
import random
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import tensorflow as tf
```
### import the necessary functions:

```python
def RoScTr_perImage2(Image,angle,scale,tx,ty):
    
    print("Orginal Image:")
    TargetImage = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    rows,cols = TargetImage.shape[:2]
    
    #Applaying the rotation and the scaling
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(TargetImage,M,(cols,rows)) 
    M2 = cv2.getRotationMatrix2D((cols/2,rows/2),0,scale)
    dst2 = cv2.warpAffine(dst,M2,(cols,rows))
      
    #Applaying the Translation  
    M3 = np.float32([[1,0,tx],[0,1,ty]])
    dst3 = cv2.warpAffine(dst2,M3,(cols,rows))
    plt.imshow(dst3)
    plt.show()
  
    return 
```

### Mount the Google Drive:
```python
from google.colab import drive
drive.mount('/content/gdrive')
```

### Read the input and the output data then split them into training and testing datastes:
```python
seed=7
test_size = 0.33

inputData= np.load('/content/gdrive/My Drive/ImagesPointsonly224x224.npy')
outputDataWithName= np.load('/content/gdrive/My Drive/outputValues224withTheImageName.npy')
print("inputData  Shape:",inputData.shape)
print("outputDataWithName   Shape:",outputDataWithName.shape)

outputData=outputDataWithName[0:,1:]
input_train, input_test = train_test_split(inputData, test_size=test_size,random_state=seed)
output_train, output_test = train_test_split(outputData, test_size=test_size,random_state=seed)

print("*************************************")
input_train = input_train.astype('float32')/255
input_test = input_test.astype('float32') /255
output_train = output_train.astype('float32')
output_test = output_test.astype('float32')
print("input_train  Shape:",input_train.shape)
print("input_test   Shape:",input_test.shape)
print("output_train Shape:",output_train.shape)
print("output_test  Shape:",output_test.shape)
```

### Load the Deep learing model (ResNet50):
```python
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=4,#MAC classes=1000
             **kwargs):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu', name='fc1')(x)
        x = layers.Dense(64, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, name='fc3')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    model.summary()
    orig_stdout = sys.stdout
    f = open('ResNet50.txt', 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    return model
print("Done")
```

### Configure the Deep network and strat training:

```python
import keras
import time
model = ResNet50(input_shape = (224, 224, 3), classes = 4)
start_time=time.time()
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit(input_train, output_train,epochs=50,batch_size=32,shuffle=True,validation_data=(input_test, output_test))
print("Time used : %.2fs" % (time.time()-start_time))
```
