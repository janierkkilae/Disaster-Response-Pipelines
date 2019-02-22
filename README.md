# Disaster Response Pipeline Project
This project is to classify disaster response messages through machine learning. 




- process_data.py reads in the data, cleans and stores it in a SQL database. 
- train_classifier.py includes the code necessary to load data, transform it using natural language processing, run a machine learning model and train it. 
- the app folder includes all the files needed to run the Flask app and the user interface along with some visualizations. 

## An example:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
> python run.py

## Screenshots
This is the frontpage:
![Alt text](https://github.com/janierkkilae/Disaster-Response-Pipelines/blob/master/Screenshot1.PNG?raw=true "Screenshot1")

By inputting a word, you can check its category:
![Alt text](https://github.com/janierkkilae/Disaster-Response-Pipelines/blob/master/Screenshot2.PNG?raw=true "Screenshot2")

## About
This project was prepared as part of the Udacity Data Scientist nanodegree programme. The data was provided by Figure Eight. 
