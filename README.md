# Overview

As a software engineer aiming to explore the potential connections between various factors and academic success, I've created a Python-based software that analyzes a dataset on graduation rates. It employs data visualization and machine learning techniques to examine how factors like high school GPA, parental level of education, and other variables correlate to college GPA in an effort to predict a college GPA given a High School GPA.

## Dataset
The dataset used in this project includes several variables such as high school GPA, college GPA, and parental level of education. Each row represents a unique student.
[Dataset](https://www.kaggle.com/datasets/rkiattisak/graduation-rate)

The purpose of this software is to investigate if and how different variables affect college GPA. This is important for educational institutions, students, and parents to understand the factors that contribute to academic success.

[Software Demo Video](https://youtu.be/KddLLzTed38)

# Data Analysis Results

Questions answered through the data analysis include:

1. How does high school GPA correlate with college GPA?
2. Is there a relationship between parental education and college GPA?
3. How do college GPAs distribute across different high school GPA ranges?

# Development Environment

The development environment consists of:

* Python 3.12
* a custom Miniconda environment
* VSCode

Libraries used:

* Pandas for data manipulation
* Matplotlib and Seaborn for data visualization
* TensorFlow and Keras for machine learning

# Useful Websites

* [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
* [Seaborn Documentation](https://seaborn.pydata.org/)
* [Tensorflow Documentation](https://www.tensorflow.org/)

# Future Work

* Improve model accuracy by tuning hyperparameters or increasing dataset size.
* Train future models on more inputs than High School GPA to get more nuanced analysis on factors of success.
* Implement better feature importance than correlation mapping to understand which variables have the most impact on college GPA.
