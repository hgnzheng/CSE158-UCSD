# CSE 158 Recommender Systems and Web Mining

CSE 158 Recommender Systems and Web Mining taken at UC San Diego during Fall Quarter 2023.

[Fall 2023 Website](https://cseweb.ucsd.edu/classes/fa23/cse258-a/)

[Winter 2023 DSC 148 Website](https://shangjingbo1226.github.io/teaching/2023-winter-DSC148-DM)

*Note: DSC 148 Introduction to Data Mining is seen as an equivalent course for Data Science major requirement purpose. It is taught by Prof. [Jingbo Shang](https://shangjingbo1226.github.io/) during Winter Quarter each year.*

***To uphold academic integrity, please do not use any solutions of the projects as your own. MOSS will be able to tell and you will suffer from significant academic integrity violation consequences.***

Like previous offerings, the course covers basic Machine Learning concepts and various approaches for Recommender Systems. Topics include:
* Regression (Least-Squares Regression, ML basics)
* Classification (Naïve Bayes Classifier, Logistic Regression, Support Vector Machines, Model Evaluation)
*  Recommender Systems (Jaccard/Cosine/Pearson Similarity Functions, Collaborative Filtering, Latent Factor Models, One-Class Recommendation, Bayesian Personalized Ranking, Evaluation Metrics - Precision/Recall, AUC, Mean Reciprocal Rank, Cumulative Gain and NDCG, Feature-Based Recommendation)
* Text Mining (Sentiment Analysis, Bags-of-Words, TF-IDF, Stopwords, Stemming, Low-Dimensional Representations of Text)
* Content and Structure in Recommender Systems (Factorization Machines, Group and Socially Aware Recommendation, Online Advertising)
* Modeling Temporal and Sequence Data (Sliding windows and autoregression, Temporal Dynamics in Recommender Systems)
* Visual Recommendation (Complementary Item Recommendation, Fashion and Outfit Recommendation, Fit Prediction)
* Ethics and Fairness (Filter Bubbles and Recommendation Diversity, Calibration, Serendipity, and Other "Beyond Accuracy" Measures, Algorithm Fairness)

[Course outline slides](https://cseweb.ucsd.edu/classes/fa23/cse258-a/slides/intro_outline.pdf)

## Homework

### [Homework 1: Regression & Classification](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/homework1.pdf)
- Basic Logistic Regression and Classification Tasks on [GoodReads Fantasy Reviews](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/fantasy_10000.json.gz) and [Beer Reviews](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/beer_50000.json) datasets.
- Score: 8.0/8.0

### [Homework 2: Diagnostics & Rating Prediction](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/homework2.pdf)
- Implemented Logistic Regression with One-Hot Encoding, Precision@k and BER, and Similarity-based Rating Predictions on [Beer Reviews](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/beer_50000.json) and [Amazon Music Instruments](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz) datasets.
- Score: 8.0/8.0

### [Homework 3: Play Prediction & Time Played Prediction](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/homework3.pdf)
- Implemented Similarity-based Recommendation and trained a regressor to predict game playing time on [Steam dataset](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/assignment1.tar.gz) (head start of Assignment 1).
- Score: 8.0/8.0

### [Homework 4: Text Mining](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/homework4.pdf)
- Implemented basic Bag-of-Words model to find the most common words. Then, use the word set for a logistic regression to predict the genre of data from [Stem Category Data](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/steam_category.json.gz).
- Implemented TF-IDF and used the scores of 1000 most common words to train a logistic regressor to predict genre category.
- Implemented *item2vec* model on [GoodReads Young Adult Reviews](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/young_adult_20000.json.gz) dataset to find similar books based on scores from Cosine similarity function.
- Score: 8.0/8.0


## Assignments

### [Assignment 1](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/assignment1.pdf)
- A continuation of Homework 3 to optimize and fine-tuning parameters for models in Homework 3 on [Steam dataset](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/assignment1.tar.gz). 
- To predict if an user would play a game, given userID and gameID, I implemented the Bayesian Personalized Ranking Model, fine-tuned Adam optimizer learning rate and model's regularization constant, and performed an ensemble with popularity-based recommendation method.
- To predict an user's play time on a given game, I implemented the Latent Factor Model with bias terms only. Then, I fine-tuned regularization constant and performed an early stop (when validation MSE starts to increase) to avoid overfitting issues.
- I eventually achieved a reasonably good performance for both models, particularly for the play prediction task. My performance on the course Leaderboard is as follows: 

|          Task          |  Private Leaderboard Rank     |   Public Leaderboard Rank    | 
| :--------------------: |  :----------------------:     |  :----------------------:    |
|    Play Prediction     | **11/603 (Top 2% of class)**  |            20/603            |
| Time Played Prediction | 44/603 (Top 8% of class)      |            68/603            | 

*Note: If graduate students (both Master's and PhD) in CSE 258/MGTA 461 are included, my ranks are **34/1209 (Top 3%)** for Play Prediction task and 154/1209 for Time Played Prediction task.*

### [Assignment 2](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/assignment2.pdf)
- This is an open-ended project.

*My project partners are [Nate del Rosario](https://github.com/natdosan), Chuong Nguyen, and Jacob Bolano. The solution for this assignment/project is the collective efforts among us.*

## End Note
*Special thanks to Prof. [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) for his dedication in teaching the course and answering questions on Piazza. Also, I appreciate TAs' efforts to hold office hours and answer Piazza posts.*