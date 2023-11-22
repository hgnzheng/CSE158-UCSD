# CSE 158 Recommender Systems and Web Mining, UC San Diego

CSE 158 Recommender Systems and Web Mining taken at UC San Diego during Fall Quarter 2023.

[Fall 2023 Website](https://cseweb.ucsd.edu/classes/fa23/cse258-a/)

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

## Homework

### [Homework 1](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/homework1.pdf)
- Basic Logistic Regression and Classification Tasks on [GoodReads Fantasy Reviews](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/fantasy_10000.json.gz) and [Beer Reviews](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/beer_50000.json) datasets.

### [Homework 2](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/homework2.pdf)
- Implemented Logistic Regression with One-Hot Encoding, Precision @k and BER, and Similarity-based Rating Predictions on [Beer Reviews](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/beer_50000.json) and [Amazon Music Instruments](https://cseweb.ucsd.edu/classes/fa23/cse258-a/data/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz) datasets.

### [Homework 3](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/homework3.pdf)
- Implemented Similarity-based Recommendation and trained a regressor to predict game playing time on Steam data (head start of Assignment 1).

### [Homework 4](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/homework4.pdf)


## Assignments (Projects)

### [Assignment 1](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/assignment1.pdf)
- A continuation of Homework 3 to optimize and fine-tuning parameters for models in Homework 3. 
- To predict if an user would play a game, given userID and gameID, I implemented the Bayesian Personalized Ranking Model, fine-tuned Adam optimizer learning rate and model's regularization constant, and performed an ensemble with popularity-based recommendation method.
- To predict an user's play time on a given game, I implemented the Latent Factor Model with bias terms only. Then, I fine-tuned regularization constant and performed an early stop (when validation MSE starts to increase) to avoid overfitting issues.
- I eventually achieved a pretty good performance for both models, particularly for the play prediction model. My performance on the course leaderboard is as follows: 

|          Task          | Private Leaderboard Rank | Public Leaderboard Rank |
| :--------------------: | :----------------------: | :---------------------: |
|    Play Prediction     |          11/603          |         20/603          |
| Time Played Prediction |          44/603          |         68/603          |

### [Assignment 2](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/assignment2.pdf)
*My project partner is Nate del Rosario ([Github](https://github.com/natdosan)). The solutions in the repository are the collective efforts between us, with help of TAs.*

- This is an open-ended project.

*My project partners are [Nate del Rosario](https://github.com/natdosan), Chuong Nguyen, and Jacob Lin. The solution for this assignment/project is the collective efforts among us.*

## End Note
*Special thanks to Professor [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) for his dedication in teaching the course and answering questions on Piazza. Also, I appreciate TAs' efforts to hold office hours and answer Piazza posts.*