# Dry Bean Classification Project

## a. Problem Statement
The goal of this project is to automate the classification of 7 different types of dry beans (Barbunia, Bombay, Cali, Dermason, Horoz, Seker, and Sira) based on their geometric features. This helps in quality control and agricultural automations.

## b. Dataset Description
- **Source**: UCI Dry Bean Dataset
- **Instances**: 13,611
- **Features**: 16 (Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, and ShapeFactors 1-4).
- **Target**: Multi-class categorization into 7 bean varieties.

## c. Github Repository Link
[https://github.com/2025ac05565-sulthanmd/ML_Assignment2.git](https://github.com/2025ac05565-sulthanmd/ML_Assignment2.git)

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9266 | 0.9950 | 0.9393 | 0.9369 | 0.9379 | 0.9117 |
| Decision Tree | 0.8950 | 0.9450 | 0.9080 | 0.9084 | 0.9081 | 0.8736 |
| kNN | 0.9232 | 0.9845 | 0.9395 | 0.9344 | 0.9367 | 0.9076 |
| Naive Bayes | 0.7580 | 0.9645 | 0.7630 | 0.7601 | 0.7596 | 0.7087 |
| Random Forest | 0.9258 | 0.9936 | 0.9379 | 0.9349 | 0.9363 | 0.9106 |

## Observations about model performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Excellent performance on this dataset; shows that bean features have strong linear separable characteristics after scaling. |
| Decision Tree | Decent performance but slightly lower than ensembles, likely due to overfitting on specific geometric variances. |
| kNN | Highly competitive results, showing that bean clusters are well-defined in the feature space. |
| Naive Bayes | The lowest performer, likely because some geometric features are highly correlated, violating the independence assumption. |
| Random Forest | Very stable and high performance, effectively handling the multi-class nature of the dataset. |
| **Overall Winner** | **Logistic Regression** (Highest Accuracy, AUC, and MCC). |
