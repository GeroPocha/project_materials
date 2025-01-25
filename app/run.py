import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

database_filepath = './data/disaster_response_db.db'
# Couldnt get relative path to work

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Load data
engine = create_engine('sqlite:///'+database_filepath)
df = pd.read_sql_table('Messages', engine)

# Load model
model = joblib.load("./models/classifier.pkl")


# Index webpage displays visuals and receives user input text for the model
@app.route('/')
@app.route('/index')
def index():
    # Extract data needed for visuals
    # Visualization 1: Distribution of message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Visualization 2: Top 5 categories with the highest message counts
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    top_categories = category_counts.head(5)
    top_category_names = list(top_categories.index)

    # Visualization 3: Proportion of messages per genre
    genre_proportions = df['genre'].value_counts(normalize=True)

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_categories
                )
            ],
            'layout': {
                'title': 'Top 5 Categories with Most Messages',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        },
        {
            'data': [
                Pie(
                    labels=genre_proportions.index,
                    values=genre_proportions.values
                )
            ],
            'layout': {
                'title': 'Proportion of Messages by Genre'
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html template
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
