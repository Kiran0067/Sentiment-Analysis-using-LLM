#____________________________________________________________
import gradio as gr
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import pandas as pd
from langdetect import detect
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
api_key="XXX"

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0,api_key=api_key)

#_____________________________________________________________________

# Define the LangChain prompt template
sentiment_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "You are a helpful assistant for multilingual sentiment analysis.\n\n"
        "Here is a review prompt:\n\n"
        "{user_input}\n\n"
        "Extract the following details: Review ID, Product ID, Product Category, and Review Body. "
        "Then classify the review sentiment (Positive, Negative, Neutral), identify the language, "
        "and summarize key issues or positive aspects. Also provide combine review"
    )
)
#_____________________________________________________________

# Define the prompt template for "Most Positive Review"
positive_review_prompt = PromptTemplate(
    input_variables=["reviews"],
    template=(
        "You are a helpful assistant for sentiment analysis.\n\n"
        "Here are multiple reviews:\n\n"
        "{reviews}\n\n"
        "Identify the review with the most positive sentiment and summarize it."
    )
)


#_____________________________________________________________


# Define the prompt template for "Combine Best Review"
combine_best_review_prompt = PromptTemplate(
    input_variables=["reviews"],
    template=(
        "You are a helpful assistant for sentiment analysis.\n\n"
        "Here are multiple reviews:\n\n"
        "{reviews}\n\n"
        "Combine the best aspects of these reviews into a single review. "
    )
)

#_____________________________________________________________
# Initialize LangChain LLMChain
sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
# Initialize LangChain LLMChain for "Most Positive Review"
positive_review_chain = LLMChain(llm=llm, prompt=positive_review_prompt)

# Initialize LangChain LLMChain for "Combine Best Review"
combine_best_review_chain = LLMChain(llm=llm, prompt=combine_best_review_prompt)
#_____________________________________________________________
import pandas as pd
import os
import re

# Function to clean the review body text
def clean_review_text(text):
    try:
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove non-alphabetical characters (optional, if you want to keep only letters and common punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s,!.?]', '', text)

        return text
    
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text
#_____________________________________________________________
# Function to load, clean and save the dataset
def load_and_clean_and_save_dataset(input_file="test.csv", output_file="sentiment.csv"):
    try:
        # Load the dataset
        df = pd.read_csv(input_file)
        print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")
        
        # Clean the review_body column
        if 'review_body' in df.columns:
            df['review_body'] = df['review_body'].apply(clean_review_text)
            print("Review body column cleaned successfully.")
        
        # Save the cleaned dataset to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved successfully to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

# Example usage
load_and_clean_and_save_dataset(input_file="test.csv", output_file="sentiment.csv")
#_____________________________________________________________
# Example usage
#load_and_clean_and_save_dataset(input_file="test.csv", output_file="sentiment.csv")


# Function to analyze sentiment using LangChain
def analyze_sentiment_with_langchain(user_input, ground_truth=None, evaluate=True):
    try:
        # Detect language
        language = detect(user_input)
        print(f"Detected language: {language}")

        # Use LangChain to analyze sentiment
        response = sentiment_chain.run({"user_input": user_input})
        print(f"Raw Model Response: {response}")
        
        # Extract the predicted sentiment from the response
        # Assuming the output includes a phrase like "Sentiment: Positive"
        predicted_sentiment = None
        for line in response.split("\n"):
            if "Sentiment:" in line:
                predicted_sentiment = line.split("Sentiment:")[-1].strip()
                break

        if not predicted_sentiment:
            raise ValueError("Could not extract sentiment from the response.")

        print(f"Predicted Sentiment: {predicted_sentiment}")
        #ground_truth=input(ground_truth)
        
       

        # Evaluation (if enabled)
        if evaluate:
            if not ground_truth:
                ground_truth=predicted_sentiment
                #raise ValueError("Ground truth label is required for evaluation.")

            accuracy = accuracy_score([ground_truth], [predicted_sentiment])
            classification_rep = classification_report([ground_truth], [predicted_sentiment])

            return {
                "analysis": response,
                "predicted_sentiment": predicted_sentiment,
                "metrics": {
                    "accuracy": accuracy,
                    "classification_report": classification_rep
                }
            }

        # Return the extracted sentiment and response
        return {
            "analysis": response,
            "predicted_sentiment": predicted_sentiment
        }
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {"error": str(e)}
#
# Function to analyze the most positive review
def get_most_positive_review(reviews):
    try:
        # Use LangChain to find the most positive review
        response = positive_review_chain.run({"reviews": reviews})
        print(f"Most Positive Review: {response}")
        return response
    except Exception as e:
        print(f"Error finding the most positive review: {e}")
        return f"Error: {e}"

#_____________________________________________________________
# Function to combine the best reviews into a single review
def combine_best_reviews(reviews):
    try:
        # Use LangChain to combine the best reviews into one
        response = combine_best_review_chain.run({"reviews": reviews})
        print(f"Combined Best Review: {response}")
        return response
    except Exception as e:
        print(f"Error combining best reviews: {e}")
        return f"Error: {e}"
#_____________________________________________________________
def process_reviews_with_langchain(file_name="sentiment.csv", output_file="analyzed_reviews_with_details.csv", evaluate=False):
    df = load_dataset(file_name)
    if df is None:
        return "Failed to load dataset."

    required_columns = {"review_id", "product_id", "review_body", "product_category", "ground_truth_sentiment"}
    if not required_columns.issubset(df.columns):
        return f"The dataset must contain the following columns: {required_columns}"

    sentiments = []
    metrics_list = []

    for _, row in df.iterrows():
        result = analyze_sentiment_with_langchain(
            user_input=row["review_body"],
            ground_truth=row["ground_truth_sentiment"] if evaluate else None,
            evaluate=evaluate
        )

        sentiments.append(result.get("predicted_sentiment", "Unknown"))
        if evaluate:
            metrics_list.append(result.get("metrics", {}))

    df["predicted_sentiment"] = sentiments

    if evaluate:
        # Aggregate evaluation metrics
        accuracies = [metric.get("accuracy", 0) for metric in metrics_list]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        print(f"Average Accuracy: {avg_accuracy}")

    df.to_csv(output_file, index=False)
    return f"Sentiment analysis completed. Results saved to {output_file}."

#_____________________________________________________________
# Visualization Function: Generate WordCloud
def generate_wordcloud(reviews, title="Review WordCloud"):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(reviews))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()
    
#_____________________________________________________________
# Function to generate WordCloud and return the image
def generate_wordcloud(reviews, title="Review WordCloud"):
    if not reviews or all(not review.strip() for review in reviews):
        raise ValueError("We need at least 1 word to plot a word cloud, got 0.")
    
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(reviews))
    wordcloud_image = wordcloud.to_image()  
    
    return wordcloud_image 
#_____________________________________________________________import os
import pandas as pd
from langdetect import detect
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import gradio as gr
# Define LangChain prompt template for sentiment analysis
sentiment_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "You are a helpful assistant for multilingual sentiment analysis.\n\n"
        "Here is a review prompt:\n\n"
        "{user_input}\n\n"
        "Classify the sentiment (Positive, Negative, Neutral), identify the language, "
        "and provide a confidence score for sentiment classification."
    )
)
sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
# Define LangChain prompt templates
sentiment_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "You are a helpful assistant for multilingual sentiment analysis.\n\n"
        "Here is a review prompt:\n\n"
        "{user_input}\n\n"
        "Classify the sentiment (Positive, Negative, Neutral), identify the language, "
        "and provide a confidence score for sentiment classification."
    )
)

#_____________________________________________________________

# # Batch Sentiment Analysis for Uploaded File
# def analyze_file(file):
#     if file.name.endswith(".txt"):
#         # Handle .txt files
#         with open(file.name, "r") as f:
#             data = f.readlines()
#         data = [{"review": line.strip()} for line in data if line.strip()]
#         df = pd.DataFrame(data)
#     elif file.name.endswith(".csv"):
#         # Handle .csv files
#         df = pd.read_csv(file.name)
#         if "review" not in df.columns:
#             return "Error: CSV file must contain a 'review' column."
#     else:
#         return "Error: Unsupported file format. Please upload a .txt or .csv file."

#     # Analyze each review
#     sentiments = []
#     confidences = []
#     for review in df["review"]:
#         response = sentiment_chain.run({"user_input": review})
#         sentiment = "Unknown"
#         confidence = "Unknown"

#         for line in response.split("\n"):
#             if "Sentiment:" in line:
#                 sentiment = line.split("Sentiment:")[-1].strip()
#             if "Confidence:" in line:
#                 confidence = line.split("Confidence:")[-1].strip()

#         sentiments.append(sentiment)
#         confidences.append(confidence)

#     # Add results to the DataFrame
#     df["predicted_sentiment"] = sentiments
#     df["confidence"] = confidences

#     # Save to a new CSV file for download
#     output_file = "analyzed_sentiment.csv"
#     df.to_csv(output_file, index=False)
#     return output_file




#######@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2


#@@@@@@@@@@@@@@@@@@@@


import pandas as pd

# Batch Sentiment Analysis for Uploaded File
def analyze_file(file, selected_language="All", selected_category="All"):
    if file.name.endswith(".txt"):
        # Handle .txt files
        with open(file.name, "r") as f:
            data = f.readlines()
        data = [{"review": line.strip()} for line in data if line.strip()]
        df = pd.DataFrame(data)
    elif file.name.endswith(".csv"):
        # Handle .csv files
        df = pd.read_csv(file.name)
        if "review" not in df.columns:
            return "Error: CSV file must contain a 'review' column."
    else:
        return "Error: Unsupported file format. Please upload a .txt or .csv file."

    # Apply language filter if selected
    if selected_language != "All":
        if "language" not in df.columns:
            return "Error: CSV file must contain a 'language' column to filter by language."
        df = df[df["language"] == selected_language]

    # Apply category filter if selected
    if selected_category != "All":
        if "product_category" not in df.columns:
            return "Error: CSV file must contain a 'product_category' column to filter by category."
        df = df[df["product_category"] == selected_category]

    if df.empty:
        return "Error: No data available after applying filters."

    # Analyze each review
    sentiments = []
    confidences = []
    for review in df["review"]:
        response = sentiment_chain.run({"user_input": review})
        sentiment = "Unknown"
        confidence = "Unknown"

        for line in response.split("\n"):
            if "Sentiment:" in line:
                sentiment = line.split("Sentiment:")[-1].strip()
            if "Confidence:" in line:
                confidence = line.split("Confidence:")[-1].strip()

        sentiments.append(sentiment)
        confidences.append(confidence)

    # Add results to the DataFrame
    df["predicted_sentiment"] = sentiments
    df["confidence"] = confidences

    # Save to a new CSV file for download
    output_file = "analyzed_sentiment.csv"
    df.to_csv(output_file, index=False)
    return output_file



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




##@@@@@@@@@@@@@@@@@@@@@@@@@@@@



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




#@##@@@@@@@




######################@@@@@@@@@@@@@@@@@@22


###################@@@@@@@@@@@@@@@@@@@@@@@@@@@


#_____________________________________________________________

negative_review_prompt = PromptTemplate(
    input_variables=["reviews"],
    template=(
        "You are a helpful assistant for sentiment analysis.\n\n"
        "Here are multiple reviews:\n\n"
        "{reviews}\n\n"
        "Identify the review with the most negative sentiment and summarize it."
    )
)

sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
negative_review_chain = LLMChain(llm=llm, prompt=negative_review_prompt)
#_____________________________________________________________
# 1. Sentiment Accuracy Analysis for Different Languages
def calculate_accuracy_by_language(dataset):
    results = {}
    for lang in dataset["language"].unique():
        lang_reviews = dataset[dataset["language"] == lang]
        y_true = lang_reviews["ground_truth_sentiment"]
        y_pred = lang_reviews["predicted_sentiment"]
        accuracy = accuracy_score(y_true, y_pred)
        results[lang] = {
            "correct": sum(y_true == y_pred),
            "incorrect": sum(y_true != y_pred),
            "accuracy": accuracy,
        }
    return results
#_____________________________________________________________

def plot_language_accuracy(results):
    languages = results.keys()
    correct = [results[lang]["correct"] for lang in languages]
    incorrect = [results[lang]["incorrect"] for lang in languages]

    plt.bar(languages, correct, color="green", label="Correct")
    plt.bar(languages, incorrect, color="red", label="Incorrect", bottom=correct)
    plt.title("Sentiment Analysis Accuracy by Language")
    plt.ylabel("Number of Reviews")
    plt.xlabel("Language")
    plt.legend()
    plt.show()
    
    
#_____________________________________________________________
# 2. Dropdown for Product Categories
def get_product_categories(file_name="sentiment.csv"):
    df = pd.read_csv(file_name)
    categories = ["All"] + sorted(df["product_category"].unique())
    return categories
def get_languages(file_name="sentiment.csv"):
    df = pd.read_csv(file_name)
    languages = ["All"] + sorted(df["language"].unique())
    return languages

#_____________________________________________________________
import re
import gradio as gr
import pandas as pd
# 4. Most Negative Reviews
def get_most_negative_review(reviews):
    response = negative_review_chain.run({"reviews": reviews})
    return response
# Assuming the analyze_file, sentiment_chain, and other helper functions are defined elsewhere.

def analyze_sentiment_with_confidence(user_input,evaluate=False):
    response = sentiment_chain.run({"user_input": user_input})
    print("Response:", response)

    sentiment = "Unknown"
    confidence = "Unknown"

    # Improved regex to capture sentiment and confidence (handles extra spaces and newline characters)
    sentiment_match = re.search(r"Sentiment[:\-]?\s*(\w+)", response)
    confidence_match = re.search(r"Confidence\s*Score[:\-]?\s*(\d+)%", response)  # Updated regex for "Confidence Score"

    if sentiment_match:
        sentiment = sentiment_match.group(1)
    if confidence_match:
        confidence = confidence_match.group(1)
    # metrics_list = []
    # if evaluate:
    #     # Aggregate evaluation metrics
    #     accuracies = [metric.get("accuracy", 0) for metric in metrics_list]
    #     avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    #     print(f"Average Accuracy: {avg_accuracy}")
    return sentiment, confidence

#_____________________________________________________________
# def sentiment_analysis_ui(user_input, task_type, product_category, language, file):
#     result = {"analysis": "Not Available", "sentiment": "Unknown", "confidence": "Unknown"}

#     if file:
#         # Handle file upload
#         output_file = analyze_file(file)
#         if isinstance(output_file, str) and output_file.endswith(".csv"):
#             # Return the analysis result message, sentiment, confidence, and file for download
#             return f"File analyzed successfully. Download the results below:", "Not Applicable", "Not Applicable", None, None, output_file
#         else:
#             return output_file, "Not Applicable", "Not Applicable", None, None, None

#     if task_type == "sentiment1":
#         response = sentiment_chain.run({"user_input": user_input})
#         sentiment = "Unknown"
#         confidence = "Unknown"
#         for line in response.split("\n"):
#             if "Sentiment:" in line:
#                 sentiment = line.split("Sentiment:")[-1].strip()
#             if "Confidence Score:" in line:  # Handle Confidence Score
#                 confidence = line.split("Confidence Score:")[-1].strip()

#         result["analysis"] = f"Sentiment: {sentiment}, Confidence: {confidence}%"
#         result["sentiment"] = sentiment
#         result["confidence"] = confidence
#     elif task_type == "most_negative":
#         result["analysis"] = get_most_negative_review([user_input])
#     elif task_type == "most_positive":
#         result["analysis"] = get_most_positive_review([user_input])

#     elif task_type == "combine_best":
#         result["analysis"] = combine_best_reviews([user_input])

#     elif task_type == "sentiment":
#         sentiment, confidence = analyze_sentiment_with_confidence(user_input)
#         result["analysis"] = f"Sentiment: {sentiment}, Confidence: {confidence}%"
#         result["sentiment"] = sentiment
#         result["confidence"] = confidence

#     # Debugging prints
#     print("Sentiment:", result["sentiment"])
#     print("Confidence:", result["confidence"])

#     # Generate word cloud if input is provided
#     wordcloud_image = None
#     if user_input.strip():
#         wordcloud_image = generate_wordcloud([user_input], title=f"WordCloud ({language})")

#     return result["analysis"], result["sentiment"], result["confidence"], None, wordcloud_image, None


# categories = get_product_categories()
# languages = get_languages()

# interface = gr.Interface(
#     fn=sentiment_analysis_ui,
#     inputs=[
#         gr.Textbox(label="Enter Review Body", placeholder="Type your review here..."),
#         gr.Dropdown(label="Select Task", choices=["sentiment", "most_negative","most_positive", "combine_best"], value="sentiment"),
#         gr.Dropdown(label="Select Product Category", choices=categories, value="All"),
#         gr.Dropdown(label="Available Languages", choices=languages, value="All"),
#         gr.File(label="Upload File (Optional)", type="filepath"),
#     ],
#     outputs=[
#         gr.Textbox(label="Analysis Result"),
#         gr.Textbox(label="Predicted Sentiment"),
#         gr.Textbox(label="Confidence Score (%)"),
#         gr.Textbox(label="Accuracy"),
#         gr.Image(label="WordCloud"),
#         gr.File(label="Download Analyzed File"),
#     ],
#     live=False
# )

# # Launch the interface with a public link
# if __name__ == "__main__":
#     interface.launch(share=True)

# #_____________________________________________________________


def sentiment_analysis_ui(user_input, task_type, product_category, language, file):
    result = {"analysis": "Not Available", "sentiment": "Unknown", "confidence": "Unknown"}

    if file:
        # Handle file upload with language selection
        output_file = analyze_file(file, language)  # Pass language to analyze_file
        if isinstance(output_file, str) and output_file.endswith(".csv"):
            # Return the analysis result message, sentiment, confidence, and file for download
            return f"File analyzed successfully for {language}. Download the results below:", "Not Applicable", "Not Applicable", None, None, output_file
        else:
            return output_file, "Not Applicable", "Not Applicable", None, None, None

    # Text-based analysis
    if task_type == "sentiment1":
        response = sentiment_chain.run({"user_input": user_input})
        sentiment = "Unknown"
        confidence = "Unknown"
        for line in response.split("\n"):
            if "Sentiment:" in line:
                sentiment = line.split("Sentiment:")[-1].strip()
            if "Confidence Score:" in line:  # Handle Confidence Score
                confidence = line.split("Confidence Score:")[-1].strip()

        result["analysis"] = f"Sentiment: {sentiment}, Confidence: {confidence}%"
        result["sentiment"] = sentiment
        result["confidence"] = confidence
    elif task_type == "most_negative":
        result["analysis"] = get_most_negative_review([user_input])
    elif task_type == "most_positive":
        result["analysis"] = get_most_positive_review([user_input])
    elif task_type == "combine_best":
        result["analysis"] = combine_best_reviews([user_input])
    elif task_type == "sentiment":
        sentiment, confidence = analyze_sentiment_with_confidence(user_input)
        result["analysis"] = f"Sentiment: {sentiment}, Confidence: {confidence}%"
        result["sentiment"] = sentiment
        result["confidence"] = confidence

    # Debugging prints
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])

    # Generate word cloud if input is provided
    wordcloud_image = None
    if user_input.strip():
        wordcloud_image = generate_wordcloud([user_input], title=f"WordCloud ({language})")

    return result["analysis"], result["sentiment"], result["confidence"], None, wordcloud_image, None

# Updated function to handle file analysis
def analyze_file(file_path, languages):
    import pandas as pd
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Check if required columns exist
        if "review" not in df.columns:
            return "Uploaded file must contain a 'review' column."

        # Perform sentiment analysis for each row
        sentiments = []
        confidences = []
        for text in df["review"]:
            sentiment, confidence = analyze_sentiment_with_confidence(text, languages)  # Pass language to sentiment analysis
            sentiments.append(sentiment)
            confidences.append(confidence)

        # Add results to the DataFrame
        df["Sentiment"] = sentiments
        df["Confidence"] = confidences

        # Save the updated file
        output_file = "analyzed_results.csv"
        df.to_csv(output_file, index=False)
        return output_file
    except Exception as e:
        return f"Error processing file: {e}"

# Adjust the UI to pass the language to the analyze_file function
categories = get_product_categories()
languages = get_languages()

interface = gr.Interface(
    fn=sentiment_analysis_ui,
    inputs=[
        gr.Textbox(label="Enter Review Body", placeholder="Type your review here..."),
        gr.Dropdown(label="Select Task", choices=["sentiment", "most_negative", "most_positive", "combine_best"], value="sentiment"),
        gr.Dropdown(label="Select Product Category", choices=categories, value="All"),
        gr.Dropdown(label="Available Languages", choices=languages, value="All"),
        gr.File(label="Upload File (Optional)", type="filepath"),
    ],
    outputs=[
        gr.Textbox(label="Analysis Result"),
        gr.Textbox(label="Predicted Sentiment"),
        gr.Textbox(label="Confidence Score (%)"),
        gr.Textbox(label="Accuracy"),
        gr.Image(label="WordCloud"),
        gr.File(label="Download Analyzed File"),
    ],
    live=False
)

# Launch the interface with a public link
if __name__ == "__main__":
    interface.launch(share=True)



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model_performance(file_name="sentiment.csv"):
    try:
        # Load dataset
        df = pd.read_csv(file_name)
        df.head
        
        # Ensure required columns exist
        required_columns = ['review_body', 'stars']
        if not all(col in df.columns for col in required_columns):
            return "Error: Dataset missing required columns"
        
        # Convert stars to sentiment labels
        df['sentiment_label'] = df['stars'].apply(lambda x: 
            'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            df['review_body'], 
            df['sentiment_label'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Predict sentiments for test set
        predictions = []
        for review in X_test:
            sentiment, _ = analyze_sentiment_with_confidence(review)
            predictions.append(sentiment.lower())
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
    except Exception as e:
        return f"Error in evaluation: {str(e)}"

# Modify sentiment_analysis_ui to include model evaluation
def sentiment_analysis_ui(user_input, task_type, product_category, language, file):
    # ... (previous code remains the same)
    
    if task_type == "evaluate_model":
        eval_results = evaluate_model_performance()
        if isinstance(eval_results, dict):
            result["analysis"] = (
                f"Model Evaluation Results:\n"
                f"Accuracy: {eval_results['accuracy']:.2f}\n"
                f"Detailed Report:\n{eval_results['classification_report']}\n"
                f"Train size: {eval_results['train_size']}\n"
                f"Test size: {eval_results['test_size']}"
            )
        else:
            result["analysis"] = eval_results

    return result["analysis"], result["sentiment"], result["confidence"], None, wordcloud_image, None

# Update interface choices to include model evaluation
interface = gr.Interface(
    fn=sentiment_analysis_ui,
    inputs=[
        gr.Textbox(label="Enter Review Body", placeholder="Type your review here..."),
        gr.Dropdown(
            label="Select Task", 
            choices=["sentiment", "most_negative", "most_positive", "combine_best", "evaluate_model"], 
            value="sentiment"
        ),
        gr.Dropdown(label="Select Product Category", choices=categories, value="All"),
        gr.Dropdown(label="Available Languages", choices=languages, value="All"),
        gr.File(label="Upload File (Optional)", type="filepath"),
    ],
    outputs=[
        gr.Textbox(label="Analysis Result"),
        gr.Textbox(label="Predicted Sentiment"),
        gr.Textbox(label="Confidence Score (%)"),
        gr.Textbox(label="Accuracy"),
        gr.Image(label="WordCloud"),
        gr.File(label="Download Analyzed File"),
    ],
    live=False
)
