# Disaster Response Pipeline Project

In this application I have analyzed disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

In the Project Workspace, I got a data set containing real messages that were sent during disaster events. I have created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Project Details

This project has 4 folders:

1. Data - This folder has the source CSV files and the python file which transforms the data and stores as sqllite db file.
2. Models - This folder has the python file in which I have built the classifier model and stored the model as pickle file.
3. Images - This folder has supporting images.
4. App - This folder has the main python app 'run.py' which hosts the site and displays visuals and message classifier screen.

Download all the files and follow below sequence:

1. Run the [ETL Pipeline Preparation.ipynb](https://github.com/Vijaiananth/Disaster-Response-Pipeline/blob/main/data/ETL%20Pipeline%20Preparation.ipynb) and you can see how the data is loaded and the list of transformations made and how the data is stored into sqllite db.
2. Run the ML Pipeline [ML Pipeline Preparation.ipynb](https://github.com/Vijaiananth/Disaster-Response-Pipeline/blob/main/models/ML%20Pipeline%20Preparation.ipynb) and you can see the ML pipelines used and the scores of the different models. At the end the model will be stored as pickle file.
3. Run [run.py](https://github.com/Vijaiananth/Disaster-Response-Pipeline/blob/main/app/run.py) which runs the hosted app and you can view the app in http://0.0.0.0:30.


Below screenshot shows overview of training data set
<img src="images/disaster-response-project1.png"
     alt="1" />

In this screenshot we can see the sample output screen
<img src="images/disaster-response-project2.png"
     alt="2" />


### Credits: [Udacity](https://www.udacity.com/), [Figure 8](https://www.figure-eight.com/)