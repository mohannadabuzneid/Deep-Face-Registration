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

### Read the input and the output data then split them to training and testing datastes:
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

#### Find and manipulate facial features in pictures

Get the locations and outlines of each person's eyes, nose, mouth and chin.

![](https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
```

Finding facial features is super useful for lots of important stuff. But you can also use for really stupid stuff
like applying [digital make-up](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py) (think 'Meitu'):

![](https://cloud.githubusercontent.com/assets/896692/23625283/80638760-025d-11e7-80a2-1d2779f7ccab.png)

#### Identify faces in pictures

Recognize who appears in each photo.

![](https://cloud.githubusercontent.com/assets/896692/23625229/45e049b6-025d-11e7-89cc-8a71cf89e713.png)

```python
import face_recognition
known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
```
