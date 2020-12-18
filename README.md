 ### Object Detection in Grocery Images

### Dataset preparation :
Used basic techniques to prepare the data. Scaling images and bboxes. Creating a Data Generator for training/validation.

Dataset downloaded from : (https://github.com/gulvarol/grocerydataset, https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz)
grocery data :!git clone https://github.com/gulvarol/grocerydataset
yolo3 :-https://drive.google.com/drive/folders/1xtLAYfkVWLDh0aU2O5g_XaJrboRsAU7X?usp=sharing

#### ASSUMPTIONS MADE

1. assumed that the Entire dataset that is hosted on github is not necessary. But included a script to take them into consideration as well. Recognising brands etc

2. assumed that I only had to predict the boxes.

3. felt that the products are really similar (All are cigarette packs).

4. assumed that I needn’t predict the brands and compare them based on the deliverables asked.

5. assumed that Data Augmentation is not necessary for a mean Average Precision of 0.7

I. assumed all the data given is Accurate and precise.



#### PLAN EMPLOYED :

I immediately felt that the YOLO algorithm would do the purpose really well. Yolo would take data in a specific format. The format and Data preparation are discussed below. I wanted the model to converge really fast.So I obtained Pretrained Darknet Weights instead of initializing them randomly. Also I wrote the script considering all the classes under one object.I also wanted to test the longer one. So I immediately began preparing for a 11 class prediction. This means predicting 11 different Objects with just 200+  images. Although it seemed very difficult, I felt it to be more cool. Now, training requires a GPU .Google colab seemed to be a perfect choice. Hence I started Coding in google colab. 


#### DESCRIPTION OF YOLO ALGORITHM

YOLO is a Convolutional Neural Network which treats the Object detection as a single regression problem. A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes. YOLO trains on full images and directly optimizes detection performance. This unified model has several benefits over traditional methods of object detection. The system divides the input image into an S × S grid.If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object. Each grid cell predicts B bounding boxes and confidence scores for those boxes.Each bounding box consists of 5 predictions: x, y, w, h, and confidence. The detection network has 24 convolutional layers followed by 2 fully connected layers. Alternating 1 × 1 convolutional layers reduce the features space from preceding layers. We pretrain the convolutional layers on the ImageNet classification task at half the resolution (224 × 224 input image) and then double the resolution for detection. YOLO imposes strong spatial constraints on boundingbox predictions since each grid cell only predicts two boxes and can only have one class. This spatial constraint limits the number of nearby objects that our model can predict. Our model struggles with small objects that appear in groups, such as flocks of birds. 

#### DATA PREPARATION :
The Data set Given contains images in Two folders. Their bounding boxes are in a file Annotations.txt on Github. The format of the given data includes a text file with each line corresponding to an image either in Test or Train folders. The format of the given data was

‘Image name (String) , number of detections(int) , [X( centre), Y(centre), Width , Height, class ] ...... Number of detection times.

Whereas, YOLO was trained to accept data in a specific format. Every Image should contain a txt file exactly with the same name as the image. IMAGEAME.JPG , IMAGENAME.TXT

Each text file should contain the objects in a New line. And each line should be ,

Class(int), X(float between zero and one ) , Y( float between zero and one) , Width( float between zero and one) , Height ( Float between zero and one)

To obtain float X and Width should be divided by width of the image whereas. Y and height should be divided by height of the image . Width and height of the image are not the same as width and height of detection.


#### HYPER PARAMETER TUNING:
Hyper parameter tuning is very important for any Deep learning model to perform well.
So, I personally observed the images and found out that all are small boxes and mostly CIgarette packs of equal size. Which means we don’t need Many anchor boxes . One or Two would suffice. Also It seemed more likely that the pre established anchor boxes in yolo algorithm would suffice this assignment.
I felt that the Anchor boxes are in two different ratios, 2:5 and 1.5-1.8:5 in a very few cases after a rough manual division.
Based on the images, I understood that there is only one class. And Around 4000-5000 objects which are all very similar. I guessed that training 2000 iterations would suffice.
I gave the batch size as 64 and subdivisions as 16 , Classes as 1 in one model and Classes as 11 in the other model.
I trained till 1000 iterations and the model with 11 classes stopped converging. As the data is insufficient. Should have done Augmentation. I continued with the model with only one class.
As Andrew NG said, Build your first model quickly and then tune everything else to perfection, I started doing the same and the environment was quite unsupportive.
I had to tune other parameters such as Filters based on the number of classes,batch size (set to 6000 for 1 class and 22000 for 11 classes) anchor boxes etc.
I also downloaded a pre trained weights file. For easier and smoother convergence.

#### TRAINING THE MODEL :

The data should be in Darknet/data/obj/ folder. Both training and testing images with their corresponding test files are pasted in it.
Two text files namely train.txt and test.txt which contains names of train and test images are pasted in the Darknet/data folder.
Now Two files were written Namely obj.data Containing lines about what to do during testing and what to do during validation and where to backup. And obj.names containing the names for classes.

#### TESTING AND RESULTS :

I was monitoring the Convergence of loss throughout the process. Stopped the Training at 1900 iterations Due to connectivity issues and found the loss to be very low.
At a Conf_threshold of 0.25 , the Precision is 0.95 after 1900 iterations
Average Intersection over union is 74.5 percent. More iterations would have helped this
True positives = 2251 FP = 107 FN = 397
F1 score is 0.90 Please refer to the Image in the zip file for validation results



