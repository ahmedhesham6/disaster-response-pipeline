# Disaster Response Pipeline

An application that classifies emergency messages to their relevant categories based on the text input. This application can be extended to forward the message to its relevant specialized organization. The training data is provided by [Figure Eight](https://appen.com/).

## Table of contents

- [Quick Start](#quick-start)
- [Project Motivation](#project-motivation)
- [Application Description](#application-description)
- [Usage](#usage)
- [Author](#author)
- [Copyright and License](#copyright-and-license)

## Quick Start

There are 2 options available:

- Clone the repo: `git clone https://github.com/ahmedhesham6/disaster-response-pipeline.git`
- [Download the latest release](https://github.com/ahmedhesham6/disaster-response-pipeline/archive/master.zip)

Initialize Enviroment:

> conda env create -f environment.yml

Activate Eniviroment:

> conda activate disaster-response

Run Application:

> cd app

> python run.py

- Go to http://0.0.0.0:3001/

## Project Motivation

The objective of this project is to help people in disasters, when help in needed the most, by build an application where people having an emergency can input a new message which then gets classified to its relevant categories. This application can be extended to forward the message to its relevant specialized organization.

## Application Description

The application's frontend is built with Flask and Plotly, while the backend is written in Python.

The application is divided into 3 folders:

- `data`

  - Contains the data that is fed to the machine learning model after it's wrangling.

- `models`

  - Containing the machine learning model which uses a LinearSVC pipeline with GridSearchCV to classify the emergency message.

- `app`

  - Puts everything together, displays visuals for the dataset generated in the `data` folder, and provides a user interface where a person can send a message to be classified.

## Usage

After getting the repo,

1. Run the following commands in the project's root directory to set up your database and model:

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`
3. Go to `http://localhost:3001/`

## Author

### Ahmad Hesham Abdelkader

- [linkedin.com/in/ahmedhesham16/](https://www.linkedin.com/in/ahmedhesham16/)
- [github.com/ahmedhesham6](https://github.com/ahmedhesham6)

## Copyright and License

Code is released under the [MIT License](https://github.com/ahmedhesham6/disaster-response-pipeline/blob/master/LICENSE).
