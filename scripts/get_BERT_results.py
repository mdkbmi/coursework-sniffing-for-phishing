import os
import sys
import click

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
import torch 
from bertopic import BERTopic
from scipy.special import softmax
import altair as alt 
from model_evaluation import get_model_metrics

@click.command()
@click.option('--train_data', type=str, required=True, help="Path to train data")
@click.option('--results_to', type=str, required=True, help="Directory to save results in")
def main(train_data: str, results_to: str):
    for path in [train_data, results_to]:
        if not isinstance(path, str):
            raise TypeError(f"All arguments must be a string: {path}")
    
        if not os.path.exists(path):
            raise FileNotFoundError(f"File/directory not found: {path}")
        
    train_df = pd.read_parquet(train_data)
    try:
        __validate_features_df(train_df)
    except Exception as e:
        raise ValueError(f"Validation failed for {train_data}: {e}")

    sample_df = train_df.sample(100, random_state=123)


    ### sentiment analysis 

    #### define function to run model and return prediction 
    def get_sentiment(model, tokenizer, text):
        try:
            encoded_input = tokenizer(text, return_tensors='pt')
        except RuntimeError:
            return 'str too long'
        
        try:
            output = model(**encoded_input)
        except RuntimeError:
            return 'str too long'
        
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        pos = np.where(scores == max(scores))[0][0]

        if pos == 0:
            return 'negative'
        if pos == 1:
            return 'neutral'
        else:
            return 'positive'

    #### load model 
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    print('Loaded model for sentiment analysis')

    ### run 

    print('Running model for sentiment analysis...')
    toy_sentiment = sample_df[['text_preprocessed', 'target_1']]
    toy_sentiment['sentiment'] = toy_sentiment['text_preprocessed'].apply(lambda x: get_sentiment(model, tokenizer, x))

    neutral_count = len(toy_sentiment[toy_sentiment['sentiment']=='neutral'])
    pos_count = len(toy_sentiment[toy_sentiment['sentiment']=='positive'])
    neg_count = len(toy_sentiment[toy_sentiment['sentiment']=='negative'])
    parsing_error_count = len(toy_sentiment[toy_sentiment['sentiment']=='str too long'])
    sentiments_count = {
        'Positive': pos_count,
        'Neutral': neutral_count,
        'Negative': neg_count,
        'Parsing Error': parsing_error_count
    }

    sentiments_count_df = pd.DataFrame([sentiments_count])

    os.makedirs(os.path.join(results_to, "BERT"), exist_ok=True)
    sentiments_count_df.to_csv(os.path.join(results_to, "BERT/BERT_sentiment_results.csv"))

    print('Sentiment analysis complete')

    ### BERTopic 

    #### define function to run model and return prediction 
    def get_topic(model, text):
        topic, prob = model.transform(text)

        return model.topic_labels_[topic[0]]

    ### load model

    topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")

    print('Loaded model for topic modelling')

    print('Running model for topic modelling...')

    toy_topic = sample_df[['text_preprocessed', 'target_1']]
    toy_topic['topic_MaartenGr'] = toy_topic['text_preprocessed'].apply(lambda x: get_topic(topic_model, x) if x is not None else " ")


    topic_count = pd.DataFrame(toy_topic[['topic_MaartenGr']].value_counts())
    topic_count.reset_index(inplace=True)
    topic_count['label_num'] = topic_count['topic_MaartenGr'].str.split("_").str[0]

    email_count_per_topic_chart = alt.Chart(topic_count).mark_bar().encode(
        x=alt.X('label_num:N', sort='-y', title='Topic Label Number'),
        y='count'
    ).properties(
        title='Email Count per Topic'
    )

    email_count_per_topic_chart.save(
        os.path.join(results_to, "BERT/email_count_per_topic_chart.png"), 
        ppi=300
    )

    print('Topic modelling complete')

    ### BERT classification 

    #### define function to run model and return prediction 
    def BERT_prediction(model, tokenizer, email_text):

        if not isinstance(email_text, str):
            email_text = str(email_text)

        # Tokenize and preprocess the input text
        inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)

        # Make the prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        # Interpret the prediction
        result = "malicious" if predictions.item() == 1 else "benign"
        
        return result

    toy_classification = sample_df[['text_clean', 'target_1']]

    #### ElSlay/BERT-Phishing-Email-Model

    model_name = 'ElSlay/BERT-Phishing-Email-Model'

    # Load the pre-trained model and tokenizer
    model_elslay = BertForSequenceClassification.from_pretrained(model_name, cache_dir='/data/workspace/alexww14/2025-cv/notebooks/milestone2/cache')
    tokenizer_elslay = BertTokenizer.from_pretrained(model_name)

    # Ensure the model is in evaluation mode
    model_elslay.eval()

    print('Loaded ElSlay/BERT-Phishing-Email-Model')

    print('Running ElSlay...')

    toy_classification['prediction_elshay'] = toy_classification['text_clean'].apply(lambda x: BERT_prediction(model_elslay, tokenizer_elslay,x))

    elshay_evaluations = get_model_metrics(toy_classification['target_1'], toy_classification['prediction_elshay'])


    print('ElSlay complete')

    #### ealvaradob/bert-finetuned-phishing

    model_name = "ealvaradob/bert-finetuned-phishing"
    
    # Load the pre-trained model and tokenizer
    model_ealvaradob = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer_ealvaradob = BertTokenizer.from_pretrained(model_name)

    # Ensure the model is in evaluation mode
    model_ealvaradob.eval()

    print('Loaded ealvaradob/bert-finetuned-phishing')

    print('Running ealvaradob...')

    toy_classification['prediction_ealvaradob'] = toy_classification['text_clean'].apply(lambda x: BERT_prediction(model_ealvaradob, tokenizer_ealvaradob,x))

    ealvaradob_evaluations = get_model_metrics(toy_classification['target_1'], toy_classification['prediction_ealvaradob'])

    print('ealvaradob complete')

    results_df = pd.DataFrame([elshay_evaluations, ealvaradob_evaluations], index=['elslay', 'ealvaradob']).round(3)

    tidy_df = pd.DataFrame({
            'Model': results_df.index,
            'Precision': results_df['precision'],
            'Recall': results_df['recall'],
            'F1-score': results_df['f1-score'],
            'False Benign Rate / FNR': results_df['false_benign_rate'],
            'False Malicious Rate / FPR': results_df['false_malicious_rate']
        })

    tidy_df.to_csv(os.path.join(results_to, "BERT/BERT_classification_results.csv"), index=True)

if __name__ == '__main__':
    main()
