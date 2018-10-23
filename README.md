# Deep Face Registration

Based on our survey, we could not find a dataset used for face registration problem. Therefore, we had to build our face registration dataset based on one of the existing state-of-the-art datasets. We decided to go with [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset.

In this research, we built the system using [Tensorflow](https://www.tensorflow.org/guide/keras) using deep learning. The model has an accuracy of 98.55% on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.


## A) Obtaining the Dataset based on the LFW (We already obtained the Dataset and it will be available soon):

   LFW data set contains more than 13,000 images of faces collected from the web. Each face has been labeled with the name of the person pictured. 1680 of the people pictured have two or more distinct photos in the data set. The only constraint on these faces is that they were detected by the Viola-Jones face detector. The images are in color scale, and the size is 250x250 pixels with variant expressions, timing, pose, and gender.. Figure 1 shows a sample of the LFW dataset.

![](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/LFW.png)
   
   ######      Figure 1 Sample images of LFW dataset.


#### How we obtained the Dataset:

-All images resized to 224x224.

-We detected the faces in each image then we eliminated any image has more than one face.

-We detected the eyes and we eliminated any image has only one eye detected.

-The new dataset will contain the images as an input and the Rotation, scaling and X_shifting and y_shifting as output.

#### steps:

###### First: 
We selected one of the LFW images as a reference image for all other images in the dataset. The reference image is centered and has a frontal face with the assumption that, no any transformation (rotation, scaling and shifting) applied to the reference image. Figure 2 shows our reference image which we used to find the transformation parameters.

![](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/Refrenace%20Image.png)

   ######     Figure 2. Refrenace Image.
   
 ###### next: 

We applied the haar-cascade face detection algorithm on the reference image to find the face boundary. Then, we detected the 6 facial landmarks associated with each eye based on the haar-cascade algorithm as it is shown in Figure 3.  In addition to the reference image, we apply the same method to all images in the dataset. Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person), and then working clockwise around the remainder of the region. The total of points we will use to calculate the transformation parameter is 12 points, and we will eliminate any image has less than that from the training dataset.

![](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/The%206%20facial%20landmarks%20associated%20with%20the%20eye.jpg)

######      Figure 3. The 6 facial landmarks associated with the eye.

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

#### The full code is available  [Coming Soon]().


## B) Deep Face Registration System:

- We used the [Google Colab system](https://colab.research.google.com) to tain our network. 
- We uploaded our code and the [Input data]() and the [Output data](https://github.com/mohannadabuzneid/Deep-Face-Registration/blob/master/outputValues224withTheImageName.zip) files to the google drive. 
- Full code is Available [Coming Soon](). 

## 1) Train the Network from scratch:

### Import the necessary packages:

```python
Coming Soon
```
### Import the necessary functions:

```python
Coming Soon 
```

### Mount the Google Drive:
```python
from google.colab import drive
drive.mount('/content/gdrive')
```

### Read the input and the output data then split them into training and testing datastes:
```python
Coming Soon
```

### Load the Deep learing model (ResNet50):
```python
Coming Soon
```

### Configure the Deep Network and start training:

```python
model = ResNet50(input_shape = (224, 224, 3), classes = 4)
start_time=time.time()
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit(input_train, output_train,epochs=50,batch_size=32,shuffle=True,validation_data=(input_test, output_test))
print("Time used : %.2fs" % (time.time()-start_time))
```

### Predict the output of the test dataset:

```python
start_time=time.time()
decoded_test_imgs = model.predict(input_test)
print("Time used : %.2fs" % (time.time()-start_time))
```

### Serialize the model to json file and the the weights to HDF5 files:

```python
from keras.models import model_from_json
model_json = model.to_json()
with open("/content/gdrive/My Drive/ResNet50with224.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/content/gdrive/My Drive/ResNet50with224.h5")
print("Saved model to disk")
```
### Compute the accuracy for the test dataset:

```python
scoresForTheTestImages = model.evaluate(input_test, output_test, verbose=0)
print("Error rate for the test images = ",scoresForTheTestImages )
print("Accurcy for the test images:", (100-scoresForTheTestImages))
```

### Test Example:

```python
imageNumber = 4000   #select one of the test images.
image=input_test[imageNumber]
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

values=decoded_test_imgs[imageNumber]
values = round((values[0]/(3.14/180)),2),round(1/values[1],2),round(values[2],0),round(values[3],0)
print("Returned Values from the CNN after Rounding: ",values)
RoScTr_perImage2(image,values[0],values[1],values[2],values[3])
```

## Load the trained model from json file and the weights from HDF5 to test our result:

From previous section repeat:
- Import the necessary packages.
- Mount the Google Drive.
- Read the input and the output data then split them into training and testing datastes.

#### Then

### Load the model from json file and the the weights to HDF5 files:

```python
from keras.models import model_from_json
#load json and create model
json_file = open('/content/gdrive/My Drive/ResNet50with224.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/gdrive/My Drive/ResNet50with224.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss='mean_absolute_error')
```
### Predict the output of the test dataset:

```python
start_time=time.time()
decoded_test_imgs = model.predict(input_test)
print("Time used : %.2fs" % (time.time()-start_time))
```
### Compute the accuracy for the test dataset:

```python
scoresForTheTestImages = model.evaluate(input_test, output_test, verbose=0)
print("Error rate for the test images = ",scoresForTheTestImages )
print("Accurcy for the test images:", (100-scoresForTheTestImages))
```

### Test Example:

```python
imageNumber = 4000   #select one of the test images.
image=input_test[imageNumber]
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

values=decoded_test_imgs[imageNumber]
values = round((values[0]/(3.14/180)),2),round(1/values[1],2),round(values[2],0),round(values[3],0)
print("Returned Values from the CNN after Rounding: ",values)
RoScTr_perImage2(image,values[0],values[1],values[2],values[3])
```

# C) Proposed Method Configurations:

Our implementation for deep face registration evaluated few models architectures. We started with a simple CNN network with few convolution layers followed by MaxPooling layer and batch normalization (BN) right after each convolution and before activation, and we do not use dropout. In the classification layers, we used two fully-connected layers one with 128 neurons and one with 64 neurons followed with the output layers with four neurons. We randomly initialize the weights, and we use the Relu activation function. We used Adam optimizer with a mini-batch size of 32. We used the default learning rate which is 0.001, and the model was trained up to 80 epochs. Figure 4 shows the model architecture.


In the second model, we went deeper and used the VGGNet. Ones with 16 layers (VGG16) and the second one with 19 layers (VGG19) as Figure 5 shows. We added two fully-connected layers one with 128 neurons and one with 64 neurons followed with the output layers with four neurons at the end to fit our problem. We used Adam optimizer with a mini-batch size of 32. However, we changed to the learning rate to 0.0001, and the model was trained up to 80 epochs. 



The last model, we used the ResNet50 model. We added a drop out layer before the fully-connected layers to make the model generalized. We used the same fully-connected and the output layers as the VGGNet model. We used Adam optimizer with a mini-batch size of 32. We used the default learning rate which is 0.001, and the model was trained up to 80 epochs. Figure 3.26 shows the ResNet50 model.




# Result
