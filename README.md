# Reverse Image Search
Given a query image, find similar images from the dataset using image features like ORB/SIFT/SURF features.

# Technologies used
* Python
* OpenCV
* Scikit-learn
* Scipy
* Numpy

# Idea
Images can be distinguished using distinctive items present in an image. For example, in the case of human faces, it could be mouth, eye corners, nose, etc. These are called **keypoints**. Keypoints can be found using algorithms like **ORB/SIFT/SURF** which use mathematical methods like local extremas to find keypoints. 

 <img src="https://cdn.educba.com/academy/wp-content/uploads/2021/03/OpenCV-KeyPoint-1.jpg.webp" alt="Image Source:https://www.educba.com/opencv-keypoint/" width="200"/>


Keypoints can be represented  using vectors called **descriptors** which describe the local structure around the keypoints. A collection of such descriptors can be obtained by running the ORB/SIFT/SURF algorithms over all the images in a dataset. 

Now, clustering algorithms like **K-means** could be used to detect cluster centroids which represent the common items present in the images of the datasets. These items are called **visual words**. 

<img src="https://miro.medium.com/max/754/1*yDysYXCrt6ONX6bAY8BIvw.png" alt="Image Source:https://miro.medium.com/max/754/1*yDysYXCrt6ONX6bAY8BIvw.png" width="200"/>


Thus every image could be represented using a **Bag of Visual Words (BOVW)** vector which can be obtained by associating every descriptor of the image to its nearest cluster centroid. BOVW is like a histogram of the closest centroids.

For a query image, its BOVW vector could be compared with that of every other dataset image using distance measures like **Cosine distance** and then the most similar images could be returned as output.

# Algorithm
1. Find keypoints and descriptors from the all dataset images using algorithms like ORB/SIFT/SURF.
2. Accumulate the descriptors from all the images
3. Find n (choice) clusters from the descriptors using K-means clustering algorithm
4. For an image, associate each of its descriptors with the closest visual word (cluster centroid) and this can be described with a vector
5. Calculate the cosine distance between the BOVW vector of the query image and with that of all the dataset images
6. Output the images with the least distances

# Pros
* **The same approach can be used to find duplicate images in a dataset**
* Algorithms like ORB/SIFT/SURF are proven approaches to accurately identify keypoints in an image.
* SIFT algorithm is rotation and scale invariant
* The usage of vector space models provide us with a ranked retrieval
* Partial matching is also possible
* With enough amount of data, this approach can be generalized

# Cons
* Keypoint detection using algorithms like SIFT may be time consuming. The ORB algorithm was developed to improve the time complexity but the performance has dropped.
* As the dataset expands and grows, the k value of the K-means algorithm needs to be updated to accommodate for new visual words
* For each query image, the cosine distance needs to be calculated with all dataset images every time, which is expensive
* The SURF algorithm is patented
* Some algorithms perform poor with low lighting and blurring
* K means algorithm could be sensitive to outliers

# Alternate Approaches
* Hashing algorithms like dHash could be used which looks at the difference between adjacent pixel values and calculating a hash value based on that. The idea is that similar images would be having similar/close hash values. 
* Mean Squared Error (MSE) between the pixel values of two images could be calculated and checked whether it falls within a threshold or not. If yes, the images are similar
* **Both the above approaches could be used to detect duplicate images, especially the MSE method.**



