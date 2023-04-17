# CancerImageDetection
Attempt to detect cancer from publically available CT scans using both traditional statistical learning methods and deep learning. 

## Background
This project is a classification task of the LDRI-IDRI database of lung nodules on CT scans (link [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)). The database contains slices of the CT scans as well as annotated values of each nodule found in the scans. A nodule is a growth of cells in the body which, if malignant, can be cancer.

## Goal
To classify each nodule found in the dataset to find instances of cancer among the CT scans.

## Methods
We used two methods to find cancer among the nodules:

1) Multinomial statistical learning models trained on each set of annotations describing a nodule. Annotations consistent of markings of the images made by radiologists describing the nodules structure, shape, texture, and other features helpful for determining presence of cancer.

2) Deep learning methods trained on the images themselves 

## Relevant files for statistical methods used:

### EDA:

01_annotation_EDA.Rmd: Explores relationships of original nodule characteristic predictors and their summary statistics with the response variable. Also, explores distributions of radiologist annotations by radiologist (not used in final report)

### Modeling:

02_RandomForestModeling.Rmd: Code for random forest model, including use of mean gini decrease as a variable selection method.

03_LogisticRegressionModeling.Rmd: Code for logistic regression models

To view fitting and results of CNN, go to this link to run the relevant Google Colab file (takes about 10-15 minutes to train): https://colab.research.google.com/drive/1ZXjOXir2pCCJrp7tAIL_8tbW-2jt2OLw?usp=sharing

## Note

All files not listed above and that are not numbered were used to create the datasets, both the annotation csvs and the tensor data used to train the vision model. They require the original dataset to run, which is 125 GBs of CT scan images and xml files. We did not include them for obvious reasons. Please refer to the Methods section of the report for a description of how the datasets used in this project were created.
