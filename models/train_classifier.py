import sys
import pandas as pd
import nltk
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download(['punkt', 'wordnet'])

# Preload stopwords for better performance
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def load_data(database_filepath):
    """Load data from SQLite database."""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """Tokenize and clean text data."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    clean_tokens = [
        lemmatizer.lemmatize(tok).lower().strip()
        for tok in tokens
        if tok.lower() not in STOPWORDS
    ]
    return clean_tokens

def build_model():
    """Build a machine learning pipeline with optimized settings."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split': [4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1, verbose=3, scoring='f1_weighted')
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model on test data."""
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'Category: {col}')
        print(classification_report(Y_test[col], Y_pred[:, i]))

def save_model(model, model_filepath):
    """Save the model as a pickle file."""
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database as the first argument and '
              'the filepath of the pickle file to save the model to as the second argument. \n\nExample: '
              'python train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
