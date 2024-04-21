import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def perform_sent_analysis(text):
    # Load spaCy model for NLP tasks
    nlp = spacy.load("en_core_web_sm")

    # Basic Sentiment Analysis with TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    print(f"TextBlob - Polarity: {polarity}, Subjectivity: {subjectivity}")

    # Sentiment Intensity with VADER
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)
    print(f"VADER - Compound Score: {vader_scores['compound']}")

    # Aspect-Based Sentiment Analysis (simplified)
    doc = nlp(text)
    aspects = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            aspect_sentiment = TextBlob(token.head.text).sentiment.polarity
            aspects.append((token.text, token.head.text, aspect_sentiment))
    print("Aspect-Based Analysis:")
    for aspect in aspects:
        print(f"Aspect: {aspect[0]}, Action: {aspect[1]}, Sentiment: {aspect[2]}")

    # Temporal Sentiment Analysis (simplified)
    doc = nlp(text)
    sentences = list(doc.sents)
    temporal_sentiments = []
    for sentence in sentences:
        sentiment = TextBlob(sentence.text).sentiment.polarity
        temporal_sentiments.append(sentiment)
    print("Temporal Sentiment Analysis:")
    for i, sentiment in enumerate(temporal_sentiments, start=1):
        print(f"Sentence {i}: Sentiment {sentiment}")
    
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "vader_scores": vader_scores,
        "aspects": aspects,
        "temporal_sentiments": temporal_sentiments
    }