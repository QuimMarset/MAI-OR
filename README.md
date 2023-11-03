# MAI-OR

## Master in Artificial Intelligence - Object Recognition - UB

This repository features the 3 assignments we did for the Object Recognition subject.

The first one consisted of Contextual Data Augmentation, in which we had to train a multi-label classification model on the VOC Pascal dataset but modify the images first. 
In particular, we had to augment the training images by stitching random objects in random places, allowing or not overlapping between the objects, and enforcing class balancing or not.
We had to study how the performance changed depending on how we augmented the images, as well as trying different mechanisms to obtain the best results.

The second consisted of Fashion Parsing, in which we had to perform semantic segmentation on images to extract the different clothes according to a set of classes (e.g. watches, shoes and pants)
We had to select a recent model to train the task and try to obtain the best results possible.

The third consisted of Body and Cloth Depth Estimation, in which we had to train a U-Net model to predict the depth of the different image pixels belonging to the clothes of the person in the middle of the image, featuring different poses.
We had to preprocess the images first, and then train the model, trying again to obtain the best results by testing different strategies. We also had to study how the performance changed depending on the image resolution, performing or not 
contextual data augmentation, and tuning the loss function.

For each assignment, we have uploaded the corresponding code, assignment statement, and results report.

