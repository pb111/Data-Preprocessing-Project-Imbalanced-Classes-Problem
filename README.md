# Data Preprocessing Project – Imbalanced Classes Problem


Imbalanced classes is one of the major problems in machine learning. In this data preprocessing project, I discuss the imbalanced classes problem. Also, I discuss various approaches to deal with this imbalanced classes problem. 


===============================================================================


## Table of Contents


I have divided this project into various sections which are listed in the table of contents below:-


1.	Introduction to imbalanced classes problem


2.	Problems with imbalanced learning


3.	Example of imbalanced classes


4.	Classification metrics


5.	Approaches to handle imbalanced classes


6.	Undersampling methods


       - Random undersampling
       
       
       - Informative undersampling
       
       
       - Near miss undersampling
       
       
       - Tomek links
       
       
       - Edited nearest neighbours
       
       
7. Oversampling methods


      - Random oversampling
        
        
      - Cluster based oversampling
        
        
      - Informative oversampling
        
        
8.	Synthetic data generation


      - Synthetic Minority Oversampling Technique (SMOTE)
        
        
      - Adaptive Synthetic Technique (ADASYN)
        
        
9.	Cost sensitive learning


10.	Algorithmic ensemble methods


11.	Imbalanced learn library


12.	Conclusion


13.    References


===============================================================================


## 1. Introduction to imbalanced classes problem


Any real world dataset may come along with several problems. The problem of **imbalanced classes** is one of them. The problem of imbalanced classes arises when one set of classes dominate over another set of classes. The former is called majority class while 
the latter is called minority class. It causes the machine learning model to be more biased towards majority class. It causes poor classification of minority classes. Hence, this problem throw the question of "accuracy" out of question. This is a very common 
problem in machine learning where we have datasets with a disproportionate ratio of observations in each class.


**Imbalanced classes problem** is one of the major problems in the field of data science and machine learning. It is very important 
that we should properly deal with this problem and develop our machine learning model accordingly.  If this not done, 
then we may end up with higher accuracy. But this higher accuracy is meaningless because it comes from a meaningless metric which 
is not suitable for the dataset in question. Hence, this higher accuracy no longer reliably measures model performance.  


Now, I will consider an example of **imbalanced classes problem** to understand the problem in depth. Suppose, we are developing a classifier to predict whether a patient has an extremely rare disease. We train the classifier and it yields 99% accuracy on the test set. This is a very high accuracy. But, it is meaningless because it does not measure our model performance to predict whether a patient has an extremely rare disease or not. 


As the disease is extremely rare, so there are only 1% of patient who actually have the disease as compared to 99% patients who 
don’t have the disease. Our classifier returns high level of accuracy simply by returning "No Disease" to every new patient. But 
the classifier does not fulfil our goal of detecting patients with the rare disease. Hence, it is meaningless.
This is an example of the imbalanced classification problem. Here the number of data points belonging to the minority class 
(“Disease”) is far smaller than the number of data points belonging to the majority class ("No Disease").


===============================================================================


## 2. Problems with imbalanced learning


The problem of imbalanced classes is very common and it is bound to happen. For example, in the above example the number of patients 
who do not have the rare disease is much larger than the number of patients who have the rare disease. So, the model does not correctly classify the patients who have the rare disease. This is where the problem arises.


The problem of learning from imbalanced data have new and modern approaches. This learning from imbalanced data is referred to as **imbalanced learning**.  


Significant problems may arise with imbalanced learning. These are as follows:-


1.	The class distribution is skewed when the dataset has underrepresented data.


2.	The high level of accuracy is simply misleading. In the previous example, it is high because most patients do not have the disease not because of the good model.


3.	There may be inherent complex characteristics in the dataset. Imbalanced learning from such dataset requires new approaches, principles, tools and techniques. But, it cannot guarantee an efficient solution to the business problem.


===============================================================================


## 3. Example of imbalanced classes


The problem of imbalanced classes may appear in many areas including the following:-


1.	Disease detection


2.	Fraud detection


3.	Spam filtering


4.	Earthquake prediction


===============================================================================


## 4. Classification metrics


Standard classification metrics do not represent the model performance in the case of imbalanced classes. Consider the above example, where we build a classifier to predict whether a patient has an extremely rare disease. The classifier yields 99% accuracy which looks good. But this 99% accuracy correctly classifies the 99% of healthy people as disease-free and incorrectly classifies the 1% of people which have the rare disease as healthy. There are several standard metrics which are used to evaluate classification model performance. These metrics are discussed in the following sections.


### Confusion matrix


A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.


Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-


**True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.


**True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.


**False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**


**False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**

These four outcomes are summarized in a confusion matrix given below.


![Confusion Matrix](https://github.com/pb111/Data-Preprocessing-Project-Imbalanced-Classes-Problem/blob/master/Images/Confusion%20matrix%20.png)



There are three metrices which are used to evaluate a classification model performance. These are `accuracy`, `precision` and `recall`. These matrices are summarized below.


### Accuracy


Accuracy is defined as the percentage of correctly classified observations. It can be calculated by dividing the number of correct predictions by the total number of predictions.


`Accuracy = Correct predictions/Total number of predictions`


Mathematically, in confusion matrix terminology, accuracy can be given as


`Accuracy = (True Positives + True Negatives)/Total Sample Size`


### Precision


Precision is defined as the percentage of relevant observations that were actually belong to a certain class among all the samples 
which were predicted to belong to the same class. 


Mathematically, it can be given as


`Precision = True Positives / (True Positives + False Positives)`


![Precision](https://github.com/pb111/Data-Preprocessing-Project-Imbalanced-Classes-Problem/blob/master/Images/Precision.png)



### Sensitivity or Recall


Sensitivity or Recall is defined as the percentage of observations that were predicted to belong to a certain class among all the samples that truly belong to that class.


Mathematically, it can be given as


`Recall = True Positives / (True Positives + False Negatives)`


![Recall](https://github.com/pb111/Data-Preprocessing-Project-Imbalanced-Classes-Problem/blob/master/Images/Recall.png)


===============================================================================
 

## 5. Approaches to handle imbalanced classes


In this section, I will list various approaches to deal with the imbalanced class problem. These approaches may fall under two categories – dataset level approach and algorithmic ensemble techniques approach. The various methods to deal with imbalanced class problem are listed below. I will describe these techniques in more detail in the following sections.


1.	Undersampling methods


2.	Oversampling methods


3.	Synthetic data generation


4.	Cost sensitive learning


5.	Ensemble methods


===============================================================================


## 6. Undersampling methods


The undersampling methods work with the majority class. In these methods, we randomly eliminate instances of the majority class. It reduces the number of observations from majority class to make the dataset balanced. It results in severe loss of information. This method is applicable when the dataset is huge and reducing the number of training samples make the dataset balanced.


Undersampling methods are of two types – **random** and **informative**.


### Random undersampling


In random undersampling method, we balance the imbalanced class distribution by choosing and eliminating observations from majority class to make the dataset balanced. 


This approach has several advantages and disadvantages which are listed below:-


**Advantages**


- If the dataset is huge, we might face run time and storage problems. Undersampling can help to handle these problems 
successfully by improving run time and storage problems by reducing the number of training data samples.


**Disadvantages**


- This method can discard potentially useful information which could be important for building the classifiers.


- The sample chosen by random under sampling may be a biased one. It may not be an accurate representation of the population. 
So, it results in inaccurate results with the actual dataset.


### Informative undersampling


In informative undersampling, we follow a pre-defined selection criterion to remove the observations from majority class. 
Within this informative undersampling technique, we have **EasyEnsemble** and **BalanceCascade** algorithms. These algorithms 
produce good results and are relatively easy to follow.


**Easy ensemble** - This technique extracts several subsets of independent samples with replacement from majority class. Then it develops multiple classifiers based on combination of each subset with minority class. It works just like a unsupervised learning algorithm.


**BalanceCascade** - This method takes a supervised learning approach where it develops an ensemble of classifier and systematically selects which majority class to ensemble.


There are other types of undersampling strategy like **near miss undersampling** , **tomeks links undersampling** and **edited nearest neighbors**. These are described in the following sections.


### Near miss undersampling


In near miss undersampling, we only sample the data points from the majority class which are necessary to distinguish the majority 
class from other classes.


#### NearMiss-1


In NearMiss-1 sampling technique, we select samples from the majority class for which the average distance of the N closest samples 
of a minority class is smallest.


![NearMiss-1](https://github.com/pb111/Data-Preprocessing-Project-Imbalanced-Classes-Problem/blob/master/Images/NearMiss-1.png)



#### NearMiss-2


In NearMiss-2 sampling technique, we select samples from the majority class for which the average distance of the N farthest samples of a minority class is smallest.


![NearMiss-2](https://github.com/pb111/Data-Preprocessing-Project-Imbalanced-Classes-Problem/blob/master/Images/NearMiss-2.png)



### Tomek links


A Tomeks link can be defined as the set of two observations of different classes which are nearest neighbours of each other.


The figure below illustrate the concept of Tomek links.


![Tomek Links](https://github.com/pb111/Data-Preprocessing-Project-Imbalanced-Classes-Problem/blob/master/Images/Tomek%20links.png)



We can see in the above image that the Tomek links (circled in green) are given by the pairs of red and blue data points that are nearest neighbors. Most of the classification algorithms face difficulty due to these points. So, we will remove these points and increase the separation gap between two classes.  Now, the algorithms produce more reliable output.


This technique will not produce a balanced dataset. It will simply clean the dataset by removing the Tomek links. It may result in 
an easier classification problem. Thus, by removing the Tomek links, we can improve the performance of the classifier even if we don't have a balanced dataset.


## Edited nearest neighbours


In this type of undersampling technique, we apply a nearest neighbours algorithm. We modify the dataset by removing samples which differ from their neighbourhood. 


We select a subset of data to be under sampled. For each sample in the subset, the nearest neighbours are computed and if the selection criteria is not fulfilled, the sample is removed.


This technique is very much similar to Tomek’s links approach. We are not trying to achieve a class imbalance. Instead we try to remove noisy observations in the dataset to make for an easier classification problem.


==============================================================================


## 7. Oversampling methods


The Oversampling methods work with the minority class. In these methods, we duplicate random instances of the minority class. So, it replicates the observations from minority class to balance the data. It is also known as **upsampling**. It may result in overfitting due to duplication of data points.  


This method can also be categorized into three types - **random oversampling**, **cluster based oversampling** and **informative oversampling**.


### Random oversampling


In random oversampling, we balance the data by randomly oversampling the minority class.


**Advantages**


-	An advantage of this method is that it leads to no information loss. 


-	This method outperform under sampling.


**Disadvantages**


-	This method increases the likelihood of overfitting as it replicates the minority class labels.


### Cluster based oversampling


In this method, the K-Means clustering algorithm technique is independently applied to minority and majority class labels. 
Thus we will identify clusters in the dataset. Subsequently, each cluster is oversampled such that all clusters of the same class 
have an equal number of instances and all classes have the same size. 


**Advantages**


-	This clustering technique helps to overcome the challenge of imbalanced class distribution.


-	Also, this technique overcome the challenges within class imbalance, where a class is composed of different sub clusters 
and each sub cluster does not contain the same number of examples.


**Disadvantages**


-	The disadvantage associated with this technique is the possibility of overfitting the training data.


### Informative oversampling


In informative oversampling, we use a pre-specified criterion and synthetically generates minority class observations. This technique 
is followed to avoid overfitting which occurs when exact replicas of minority instances are added to the main dataset. In this technique, we create a subset of data from the minority class and then new synthetic similar instances are created. These synthetic instances are then added to the original dataset. The new dataset is then used as a sample to train the classification models.


**Advantages**


-	This technique reduces the problem of overfitting.


-	It does not result in loss of useful information.


**Disadvantages**


-	Generating synthetic examples SMOTE does not take into account neighbouring examples from other classes. It may result in overlapping of classes and can introduce additional noise.


-	SMOTE is not very effective for high dimensional data.


===============================================================================


## 8. Synthetic data generation


In synthetic data generation technique, we overcome the data imbalances by generating artificial data. So, it is also a type of oversampling technique.


### Synthetic Minority Oversampling Technique or SMOTE.


In the context of synthetic data generation, there is a powerful and widely used method known as **synthetic minority oversampling technique** or **SMOTE**. Under this technique, artificial data is created based on feature space. Artificial data is generated with bootstrapping and k-nearest neighbours algorithm.  It works as follows:-


1.	First of all, we take the difference between the feature vector (sample) under consideration and its nearest neighbour.


2.	Then we multiply this difference by a random number between 0 and 1.


3.	Then we add this number to the feature vector under consideration.


4.	Thus we select a random point along the line segment between two specific features. 


So, **SMOTE** generates new observations by interpolation between existing observations in the dataset.



![Synthetic Minority Oversampling Technique or SMOTE](https://github.com/pb111/Data-Preprocessing-Project-Imbalanced-Classes-Problem/blob/master/Images/smote.png)




 ### Adaptive Synthetic Technique or ADASYN
 

This technique works in a similar way as SMOTE. But the number of samples generated is proportional to the number of nearby 
samples which do not belong to the same class. Thus it focusses on outliers when generating the new training samples.


===============================================================================


## 9. Cost sensitive learning


Cost sensitive learning is another commonly used method to handle imbalanced classification problem. This method evaluates the cost associated with misclassifying the observations.


This method does not create balanced data distribution. Rather it focusses on the imbalanced learning problem by using cost matrices which describes the cost for misclassification in a particular scenario. Researches have shown that this cost sensitive learning may outperform sampling methods. So, it provides likely alternative to sampling methods.


===============================================================================


## 10. Algorithmic ensemble methods


So far we have looked at techniques to provide balanced datasets. In this section, we will take a look at an alternative approach 
to deal with imbalanced datasets. In this approach, we modify the existing classification algorithms to make them appropriate for imbalanced datasets.


In this approach, we construct several two stage classifiers from the original data and then we aggregate their predictions. The main aim of this ensemble technique is to improve the performance of single classifiers.


The ensemble technique are of two types - **bagging** and **boosting**. These techniques are discussed below:-


### Bagging


**Bagging** is an abbreviation of **Bootstrap Aggregating**. In the conventional bagging algorithm, we generate n different bootstrap training samples with replacement. Then we train the algorithm on each bootstrap training samples separately and then aggregate the predictions at the end. Bagging is used to reduce overfitting in order to create strong learners so that we can generate strong predictions. Bagging allows replacement in the bootstrapped training sample.


The machine learning algorithms like logistic regression, decision tree and neural networks are fitted to each bootstrapped training sample. These classifiers are then aggregated to produce a compound classifier. This ensemble technique produces a strong compound classifier since it combines individual classifiers to come up with a strong classifier. 


**Advantages**


-	This technique improves stability and accuracy of machine learning algorithms.


-	It reduces variance and overcomes overfitting.


-	It improves misclassification rate of the bagged classifier.


-	In noisy data situations bagging outperforms boosting.


**Disadvantages**


-	Bagging works only if the base classifiers are not bad to begin with. Bagging with bad classifiers can further degrade the performance.


### Boosting


Boosting is an ensemble technique to combine weak learners to create a strong learner so that we can make accurate predictions. 
In boosting, we start with a base or weak classifier that is prepared on the training data.


The base learners are weak learners. So, the prediction accuracy is only slightly better than average. A classifier learning algorithm is said to be weak when small changes in data results in big changes in the classification model.


===============================================================================


## 11. Imbalanced Learn


There is a Python library which enable us to handle the imbalanced datasets. It is called **Imbalanced-Learn**. It is a Python library which contains various algorithms to handle the imbalanced datasets. It can be easily installed with the *pip* command. This library contains a *make_imbalance* method to exasperate the level of class imbalance within a given dataset.


===============================================================================


## 12. Conclusion


Imbalanced data is one of the major problem in the area of machine learning. This problem can be solved by analyzing the dataset 
in hand. There are several approaches that will help us to handle the problem of imbalanced classes. These are oversampling, undersampling, synthetic data generation (SMOTE), adaptive synthetic technique and ensemble methods. These methods are discussed 
in the previous sections.


Some combination of these approaches will help us to create a better classifier.  Simple sampling techniques may handle slight 
imbalance whereas more advanced methods like ensemble methods are required for extreme imbalances.  The most effective technique 
will vary according to the dataset.


So, based on above discussion, we can conclude that there is no one solution to deal with the imbalanced class problem. We should try out multiple methods to select the best-suited sampling techniques for the dataset in hand. The most effective technique will vary according to the characteristics of the dataset.


==============================================================================


## 13. References


The ideas and concepts in this project are taken from the following websites:-


1.	https://imbalanced-learn.org/en/stable/index.html 


2.	https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/.


3.	https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/


4.	https://www.jeremyjordan.me/imbalanced-data/


5.	https://blog.dominodatalab.com/imbalanced-datasets/


6.	https://elitedatascience.com/imbalanced-classes


7.	https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets


8.	https://www.svds.com/learning-imbalanced-classes/

