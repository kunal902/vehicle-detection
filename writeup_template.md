**Vehicle Detection Project**

The goals / steps of this project are the following:

* Apply a color transform and append binned color features, as well as histograms of color
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier and append the features to the feature vector calculated above
* Append the features from above two steps, normalize it, randomize the selection for training and testing sets and train a linear SVM classifier
* Implement a multi-scale sliding-window technique and use the trained classifier to search for vehicles in images
* Use techniques such as heat map or labels to remove multiple detections and false positives
* Run the pipeline on project and test video stream

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_example_1.png
[image3]: ./output_images/hog_example_2.png
[image4]: ./output_images/hog_example_3.png
[image5]: ./output_images/hog_example_4.png
[image6]: ./output_images/sliding_window_test.png
[image7]: ./output_images/sliding_window.png
[image8]: ./output_images/heat_map_1.png
[image9]: ./output_images/heat_map_2.png
[image10]: ./output_images/label.png
[image11]: ./output_images/final_output_1.png
[image12]: ./output_images/final_output_2.png

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of some of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and found out that YCrCb color space is working fine for the problem in hand. Then I performed spatial binning on the images and by experimentation found out that even going all the way down to 32 x 32 pixel resolution, the car itself is still clearly identifiable by eye, and this means that the relevant features are still preserved at this resolution and with lesser features. Then I applied color histogram to the different channels of the image and combine them. Then I performed Histogram of Oriented Gradient(HOG) whose parameters such as orientations, pixels_per_cell and cells_per_block are taken by experimenting with the values. Finally I combined the feature vectors obtained from each step into one to be passed to the classifier for training.
Extraction of features is described in detail along with comments in `extract_features` function in IPython Notebook file. Below are some visualization of all these steps:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found out that `orientation` which represents the number of orientation bins that the gradient information will be split up into in the histogram to be 15.

The `pixels_per_cell` parameter which specifies the cell size over which each gradient histogram is computed to be (8, 8).

The `cells_per_block` which specifies the local area over which the histogram counts in a given cell will be normalized to be (2,2).

The HOG features are calculated for all the three channels and then combined later.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before training the classifier I split the dataset into separate train and test sets to avoid overfitting or improve generalization, randomly shuffle the data set to avoid problems due to ordering of the data, normalize the dataset to avoid individual features or set of features to dominate the response of the classifier. Also I have around 8792 car images and 8968 non car images which are roughly the same, this is important to avoid the problem of classifying everything belonging to the majority class.
After all these steps, I trained a linear SVM using feature vectors calculated in the above steps and got an accuracy of 98.99%. All the steps of training is described in detail along with the comments in code cell 8 of IPython Notebook.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After looking at the image height I decided to ignore the top 400 pixels and start looking for the vehicles in the below parts of the image i.e after 400 pixels. Then I varied the scales with increasing value as I searched in the lower parts of the image. I have experimented with varying values of `ystart`, `ystop` and `scale` in `find_cars` function in IPython Notebook and found out the most suitable values for this project. Below are some sample images for the same:

![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on all three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

[Youtube Link](https://youtu.be/wMz1v98yi1g)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are four frames and their corresponding heatmaps:

![alt text][image8]
![alt text][image9]


### Here is the output of `scipy.ndimage.measurements.label()` on the heatmap of the first test image:

![alt text][image10]

### Here the resulting bounding boxes are drawn onto the test frame in the series:

![alt text][image11]
![alt text][image12]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project I followed the techniques and recommendations given by Udacity in the classroom lectures. The pipeline worked well on the project video with some false positive detection on the guardrail to the left

#### Potential Shortcomings

One potential shortcoming is when the two cars are very close to each other the current pipeline is overlapping

Another shortcoming is pipeline will most likely fail in case of difficult lighting and environmental conditions

Also oncoming cars as well as distant cars are problematic. In case of distant cars, smaller window scales tend to produce more false positives

#### Possible Improvements

It is possible to improve further the classifier by additional data augmentation or classifier parameters tuning by Grid Search Cross Validation techniques

Multi scale sliding window scanning is done using experimental scales which can be improved by some automated way of calculating scales

The pipeline may fail in difficult lighting conditions which then can be improved further partly by classifier improvement
