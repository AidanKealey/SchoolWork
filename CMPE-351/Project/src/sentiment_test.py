import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load models
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
sentiment_model_path = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)

def get_aspect_sentiment(sentence, aspect):
    inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()[0]  # returns the probabilities for negative, neutral, positive

def compute_sentiment_score(probs):
    # Sentiment weights map
    if (probs[0] < probs[2]):
        output = -1 * probs[0] + -1 * probs[1] + 1 * probs[2]
    elif (probs[0] > probs[2]):
        output = -1 * probs[0] + 1 * probs[1] + 1 * probs[2]
    else:
        output = -1 * probs[0] + 0 * probs[1] + 1 * probs[2]

    # scaled_score = 100 * output
    return output

def process_review(sentence, aspects):
    print(f"\nReview: '{sentence}'")
    review_sentiments = {aspect: {'negative': 0, 'neutral': 0, 'positive': 0} for aspect in aspects}

    for aspect in aspects:
        probs = get_aspect_sentiment(sentence, aspect)
        review_sentiments[aspect]['negative'] = probs[0]
        review_sentiments[aspect]['neutral'] = probs[1]
        review_sentiments[aspect]['positive'] = probs[2]
        print(f"Sentiment of aspect '{aspect}': Negative: {probs[0]:.4f}, Neutral: {probs[1]:.4f}, Positive: {probs[2]:.4f}")
        sentiment_score = compute_sentiment_score(probs)
        print(f"Sentiment score of aspect '{aspect}': {sentiment_score:.4f}")
    
    return review_sentiments

# Example data
reviews = [
    "The place was a very nice decoration and space, the service is super friendly and fast. The coffee is really good. "
    "It was great to have all the options in the menu with no meat.",
    "The Benedict breakfast is good, but i saw the potato fries in the next table and i was jealous. the scramble tofu with vegetables is great",
    "Food ok, service is horrible. Food took soo long to come out.",
    "Best Pizza Joint in Bethpage. We order weekly and their delivery service is fantastic and never late! The Pizza is Perfect NY style",
]
aspects = ["price", "service", "ambiance", "food"]

# Process each review individually
for review in reviews:
    review_sentiments = process_review(review, aspects)
    for aspect, sentiments in review_sentiments.items():
        print(f"Final sentiment for '{aspect}' in this review: Negative: {sentiments['negative']:.4f}, Neutral: {sentiments['neutral']:.4f}, Positive: {sentiments['positive']:.4f}")
