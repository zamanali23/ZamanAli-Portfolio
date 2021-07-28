# Data Science Portfolio

Here are some of my best Data Science Projects. I have explored various machine-learning algorithms for different datasets. Feel free to contanct me to learn more about my experience working with these projects.

***

[Examining the effect of environmental factors and weather on Bike rentals](https://github.com/zamanali23/zaman-ali.github.io/blob/master/linear_reg_project.ipynb)

<img src="images/seoul-bikes.jpeg?raw=true"/>

- Used Linear Regression to predict the number of bikes rented in the city of Seoul
- The data had quite a few categorical variables which were encoded for use in the model
- Encoded categorical variables to numeric using Sklearn due to the presence of many string columns
- Fit a multiple linear regression model with high prediction accuracy through iteration
- Calculated the root mean squared error which was 0.4364, which is less than 10% of the mean value of the independent variable temperature(C)

***

[Diagnosis of breast cancer using a logistic classifier](https://github.com/zamanali23/zaman-ali.github.io/blob/master/Log_reg_project.ipynb)

<img src="images/breast-cancer.jpeg?raw=true"/>

- Used logistic regression to identify a tumour as malignant or benign based on various attributes
- Classified tumors as benign or malignant by studying patterns in measured attributes of those tumors
- Used Logistic regression classifier & optimized the accuracy by using the ROC curve
- Explored a machine learning approach to medical diagnosis
- Calculated Random forest,Adaboost Ensemble model,Bagging classifier model and Gradient Boost classifier model accuracy which was 1.0 in all the models

***

[Identifying symptoms of orthopedic patients as normal or abnormal](https://github.com/zamanali23/zaman-ali.github.io/blob/master/KNN_project.ipynb)

<img src="images/knee-brace-ortho.png?raw=true"/>

- Used the K Nearest Neighbours algorithm to classify a patient's condition as normal or abnormal based on various orthopedic parameters
- Compared predictive performance by fitting a Naive Bayes model to the data
- Selected best model based on train and test performance
- Tuned hyperparameter best score was 0.8586
- The accuracy of the KNN with K = 15 is 85.48%
- The accuracy of the NB is 82.26%

***

[Recognising Handwritten numbers using Neural Networks](https://github.com/zamanali23/zaman-ali.github.io/blob/master/Handwriting_Recognition_using_CNN_project.ipynb)

<img src="images/Plot-of-a-Subset-of-Images-from-the-MNIST-Dataset-1024x768.png?raw=true"/>

- Learned how to augment images for classfication 
- Implemented a CNN model 
- Created a pipeline to improve it 
- Predicted Image from our Final model which was 7

***

[Deployed Bagging and Boosting models to classify app-based transactions as fraudlent or not](https://github.com/zamanali23/zaman-ali.github.io/blob/master/Bagging_Boosting_Project.ipynb)

<img src="images/click-fraud1.jpg?raw=true"/>

- Explored the dataset for anomalies and missing values 
- By using Pandas derive new features
- Applied XGBoostClassifier with default parameters
- Calculated AUC/ROC score with default hyperparameters.
- Computed feature importance score and name the top 5 features/columns
- Applyied BaggingClassifier with base_estimator LogisticRegression and compute AUC/ROC score.
- On the basis of AUC/ROC score compared BaggingClassifier and XGBoostClassifier 
- For XGB Classifier auc the score was 0.8941
- For baggingClassifier Classifier the auc score was 0.871

***



[Amazon Fine Food Reviews Analysis](https://github.com/zamanali23/zaman-ali.github.io/blob/master/Amazon_Fine_Food_Reviews_Analysis.ipynb)

<img src="images/amazon.png?raw=true"/>

- Applied Text Preprocessing: Stemming, stop-word removal and Lemmatization
- Trained Multinomial Naive Bayes Model
- Predicted using BOW
- Predicted using TF-Idf
- Prediction using BOW: +ve review
- Prediction using TF-Idf: -ve review

***
[Principle Component Analysis Using Housing Dataset](https://github.com/zamanali23/zaman-ali.github.io/blob/master/pca_project.ipynb)

<img src="images/pca.jpg?raw=true"/>

- Predicted the housing price
- Isolated the important features using correlation analysis
- Applied PCA for the actual prediction model
- Applied Linear Regression
- Compared the resluts with and without PCA
- Calculated the Mean Absolute Error of linear regression: 23782.177
- Calculated the Mean Square Error of linear regression: 1424296795.105
- Calculated the Root_mean_Squared Score of linear regression: 0.7083

***

[K Means model for State authorities/Policy makers](https://github.com/zamanali23/zaman-ali.github.io/blob/master/K_Means_Project.ipynb)

<img src="images/kmean.jpg?raw=true"/>

- Fitted the data to clustering model
- Calculated the number of clusters for the data
- Fitted the model with K-Means and Hierarchical clustering
- Compared the resluts for both methods
- Calculated the silhouette score of the model which was 0.5259

***
