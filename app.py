# app.py (This will be your new Streamlit app based on plot.py)
import streamlit as st
from sent import perform_sent_analysis
import matplotlib.pyplot as plt


# make plots not as big
# Set default font sizes for plots
plt.rcParams['axes.titlesize'] = 16  # Title font size
plt.rcParams['axes.labelsize'] = 12  # Axes label font size
plt.rcParams['xtick.labelsize'] = 10 # X tick label size
plt.rcParams['ytick.labelsize'] = 10 # Y tick label size
plt.rcParams['legend.fontsize'] = 12 # Legend font size
plt.rcParams['figure.figsize'] = (8, 4)  # Default fig size - can be overridden in subplots

# Function to plot Polarity comparison for 4 texts
def plot_polarity_comparison(sent_results):
    num_texts = len(sent_results)
    polarity_scores = [sent_results[i]['polarity'] for i in range(num_texts)]
    text_labels = [f'Text {i+1}' for i in range(num_texts)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(text_labels, polarity_scores, color=['blue', 'orange', 'green', 'red'])
    ax.set_ylabel('Polarity Score')
    ax.set_title('Polarity Comparison')
    ax.set_ylim(-1, 1)  # Polarity range
    return fig

# Function to plot Subjectivity comparison for 4 texts
def plot_subjectivity_comparison(sent_results):
    num_texts = len(sent_results)
    subjectivity_scores = [sent_results[i]['subjectivity'] for i in range(num_texts)]
    text_labels = [f'Text {i+1}' for i in range(num_texts)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(text_labels, subjectivity_scores, color=['blue', 'orange', 'green', 'red'])
    ax.set_ylabel('Subjectivity Score')
    ax.set_title('Subjectivity Comparison')
    ax.set_ylim(0, 1)  # Subjectivity range
    return fig

# Function to plot VADER Compound Score comparison for 4 texts
def plot_vader_comparison(sent_results):
    num_texts = len(sent_results)
    vader_scores = [sent_results[i]['vader_scores']['compound'] for i in range(num_texts)]
    text_labels = [f'Text {i+1}' for i in range(num_texts)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(text_labels, vader_scores, color=['blue', 'orange', 'green', 'red'])
    ax.set_ylabel('VADER Compound Score')
    ax.set_title('VADER Compound Score Comparison')
    ax.set_ylim(-1, 1)  # Compound score range
    return fig

# Function to plot Aspect-Based Sentiment comparison for 4 texts
def plot_aspect_based_comparison(sent_results):
    num_texts = len(sent_results)
    aspects_scores = []
    for i in range(num_texts):
        # Debugging print statement
        print(f"Text {i+1} analysis result: {sent_results[i]}")
        
        if sent_results[i]['aspects']:
            aspects_sentiments = [aspect[2] for aspect in sent_results[i]['aspects']]
            avg_sentiment = sum(aspects_sentiments) / len(aspects_sentiments)
            aspects_scores.append(avg_sentiment)
            
            # Debugging print statements
            print(f"Text {i+1} aspects sentiments: {aspects_sentiments}")
            print(f"Text {i+1} average aspect sentiment: {avg_sentiment}")
        else:
            # Handle the case where no aspects are detected
            aspects_scores.append(0)
            print(f"Text {i+1} has no aspects detected, aspect score set to 0.")

    # Debugging print statement
    print('Aspect scores:', aspects_scores)
    
    # Calculate minimum and maximum values for the y-axis range
    min_aspect_score = min(aspects_scores)
    max_aspect_score = max(aspects_scores)
    min_val = min(min_aspect_score - 0.05, -0.1) if min_aspect_score < 0 else 0
    max_val = max(max_aspect_score + 0.05, 0.1) if max_aspect_score > 0 else 0.1
    
    # Ensure a minimum visible bar height for zero values
    min_visible_height = 0.02
    for i, score in enumerate(aspects_scores):
        if score == 0:
            # Apply a minimum height for zero value bars
            aspects_scores[i] = min_visible_height
            # Modify the max_val if necessary
            max_val = max(max_val, min_visible_height * 1.1)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    text_labels = [f"Text {i+1}" for i in range(num_texts)]
    ax.bar(text_labels, aspects_scores, color=['blue', 'orange', 'green', 'red'])
    ax.set_ylabel('Average Aspect Sentiment')
    ax.set_title('Average Aspect-Based Sentiment by Text')
    ax.set_ylim(min_val, max_val)
    
    # Debugging print statements for axes limits
    print(f"Y-axis limits set to: {min_val}, {max_val}")
    
    return fig




# Function to plot Temporal Sentiment Analysis comparison for 4 texts
def plot_temporal_sentiments(sent_results):
    num_texts = len(sent_results)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(num_texts):
        ax.plot(sent_results[i]['temporal_sentiments'], label=f'Text {i+1}')

    ax.set_ylabel('Sentiment')
    ax.set_xlabel('Sentence Number')
    ax.set_title('Temporal Sentiment Analysis')
    ax.legend()
    return fig

def interpret_polarity(polarity_scores):
    if all(score > 0 for score in polarity_scores):
        return "All texts are generally positive."
    elif all(score < 0 for score in polarity_scores):
        return "All texts are generally negative."
    else:
        return "Texts have a mix of positive and negative sentiments."

def interpret_subjectivity(subjectivity_scores):
    if all(score > 0.5 for score in subjectivity_scores):
        return "All texts are mostly subjective."
    elif all(score < 0.5 for score in subjectivity_scores):
        return "All texts are mostly objective."
    else:
        return "Texts have a mix of objectivity and subjectivity."

def interpret_vader(vader_scores):
    if all(score > 0.05 for score in vader_scores):
        return "All texts are skewed towards positive sentiment."
    elif all(score < -0.05 for score in vader_scores):
        return "All texts are skewed towards negative sentiment."
    else:
        return "Texts present a range of sentiments from positive to negative."

def interpret_aspects(aspects_scores):
    if all(score == 0 for score in aspects_scores):
        return "No significant aspect-based sentiments detected."
    else:
        positive_aspects = sum(score > 0 for score in aspects_scores)
        negative_aspects = sum(score < 0 for score in aspects_scores)
        return f"{positive_aspects} texts have positive aspect sentiments, and {negative_aspects} have negative aspect sentiments."

def interpret_temporal(temporal_sentiments):
    changes = [any(sent[i] != sent[i-1] for i in range(1, len(sent))) for sent in temporal_sentiments]
    if all(change for change in changes):
        return "Sentiment fluctuates over time in all texts."
    else:
        return "Some texts have stable sentiment, while others fluctuate."


def main():
    # Streamlit page configuration
    st.set_page_config(page_title="Text Sentiment Comparison", layout="wide")

    # Title for your app
    st.title("Text Sentiment Comparison")

    # Text input
    st.header("Input texts for sentiment analysis:")
    text_inputs = []
    for i in range(4):
        text = st.text_area(f"Text {i+1}", key=f"text_{i+1}")
        text_inputs.append(text)

    # Queue display
    st.header("Texts to be analyzed:")
    for i, text in enumerate(text_inputs):
        st.write(f"Text {i+1}:")
        st.write(text if text else "No text entered.")

    # Compare button
    if all(text_inputs):
        if st.button("Compare Now"):
            with st.spinner('Analyzing sentiments...'):
                sent_results = [perform_sent_analysis(text) for text in text_inputs]

                # Generate and display plots with insights
                fig1 = plot_polarity_comparison(sent_results)
                st.pyplot(fig1, use_container_width=False)
                st.markdown("<p style='font-size:40px'>Polarity: " + interpret_polarity([result['polarity'] for result in sent_results]) + "</p>", unsafe_allow_html=True)

                fig2 = plot_subjectivity_comparison(sent_results)
                st.pyplot(fig2, use_container_width=False)
                st.markdown("<p style='font-size:40px'>Subjectivity: " + interpret_subjectivity([result['subjectivity'] for result in sent_results]) + "</p>", unsafe_allow_html=True)

                fig3 = plot_vader_comparison(sent_results)
                st.pyplot(fig3, use_container_width=False)
                st.markdown("<p style='font-size:40px'>Vader: " + interpret_vader([result['vader_scores']['compound'] for result in sent_results]) + "</p>", unsafe_allow_html=True)

                fig4 = plot_aspect_based_comparison(sent_results)
                st.pyplot(fig4, use_container_width=False)
                st.markdown("<p style='font-size:40px'>Aspect Sentiment: " + interpret_aspects([result.get('average_aspect_sentiment', 0) for result in sent_results]) + "</p>", unsafe_allow_html=True)

                fig5 = plot_temporal_sentiments(sent_results)
                st.pyplot(fig5, use_container_width=False)
                st.markdown("<p style='font-size:40px'>Temporal: " + interpret_temporal([result['temporal_sentiments'] for result in sent_results]) + "</p>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
