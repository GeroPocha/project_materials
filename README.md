# Disaster Response Pipeline Project

## Project Overview
This project is a machine learning application designed to automatically classify disaster messages into multiple categories. This enables aid organizations to process messages more efficiently and respond faster. The application includes:

1. **ETL Pipeline**: Extract, clean, and store messages and category data in an SQLite database.
2. **ML Pipeline**: Train a model to classify messages into 36 categories and save the model as a pickle file.
3. **Web App**: A Flask web application that:
   - Classifies user input messages.
   - Displays data visualizations derived from the SQLite database.

---

## Project Structure

```
.
├── app
│   ├── templates
│   │   ├── master.html        # Main page of the web app
│   │   ├── go.html            # Results page of the web app
│   ├── run.py                 # Flask app for the web application
│
├── data
│   ├── disaster_messages.csv  # Messages dataset
│   ├── disaster_categories.csv # Categories dataset
│   ├── process_data.py        # ETL script
│   ├── DisasterResponse.db    # SQLite database with cleaned data
│
├── models
│   ├── train_classifier.py    # Model training script
│   ├── classifier.pkl         # Trained model as a pickle file
│
├── README.md                  # Project documentation
```

---

## Instructions

### 1. Run the ETL Pipeline
Cleans the data and stores it in an SQLite database.
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

### 2. Run the ML Pipeline
Trains a model and saves it as a pickle file.
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

### 3. Launch the Web App
Starts the Flask web application. Navigate to `http://localhost:3000/` to use the app.
```bash
python app/run.py
```

---

## Requirements
Install the required Python libraries:
```bash
pip install -r requirements.txt
```
Key libraries include:
- pandas
- sqlalchemy
- scikit-learn
- nltk
- flask
- plotly

---

## Important Notes
1. Ensure that the database and model pickle file are created before launching the web app.
2. Update the paths in `run.py` to ensure the web app loads the correct data and models.

---

## GitHub Usage
1. **First Commit:** Initialize the repository and add the basic files.
2. **Second Commit:** Implement the ETL and ML pipelines.
3. **Third Commit:** Integrate the web app and finalize the documentation.

---

## Final Note
This project was created as part of the Udacity Data Science Nanodegree.

