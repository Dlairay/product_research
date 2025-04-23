import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
from PIL import Image
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string

# Download required NLTK resources (you only need to do this once)
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('punkt_tab')  # Add this line
except LookupError:
    nltk.download('punkt_tab')



def extract_keywords(text):
    """
    Extracts keywords from a given text using NLTK.

    Args:
        text (str): The input text.

    Returns:
        list: A list of keywords.
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in word_tokens]
    words = [w for w in stripped if w.isalpha() and w not in stop_words]
    return words



def calculate_similarity(text1, text2):
    """
    Calculates the cosine similarity between two texts.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The cosine similarity score (between 0 and 1).
    """
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]



def group_similar_terms(label_counts, similarity_threshold=0.6):
    """
    Dynamically groups similar terms in the label counts dictionary based on keyword similarity.

    Args:
        label_counts (dict): A dictionary containing labels as keys and their counts as values.
        similarity_threshold (float, optional): The threshold for grouping similarity. Defaults to 0.6.

    Returns:
        dict: A new dictionary with grouped terms and their combined counts.
    """
    grouped_counts = defaultdict(int)
    labels = list(label_counts.keys())
    grouped_labels = []

    for i, label1 in enumerate(labels):
        if i in [g[0] for g in grouped_labels]:
            continue  # Skip if already grouped

        group = [i]
        group_keywords = extract_keywords(label1)

        for j in range(i + 1, len(labels)):
            if j in [g[0] for g in grouped_labels]:
                continue # Skip if already grouped

            label2 = labels[j]
            similarity = calculate_similarity(" ".join(group_keywords), " ".join(extract_keywords(label2)))
            if similarity > similarity_threshold:
                group.append(j)

        grouped_labels.append(group)
    
    for group_indices in grouped_labels:
        group_name = " ".join([labels[i].split()[0] for i in group_indices]) #join the first words
        combined_count = sum(label_counts[labels[i]] for i in group_indices)
        grouped_counts[group_name] = combined_count
    
    
    ungrouped_labels = [i for i in range(len(labels)) if i not in [item for sublist in grouped_labels for item in sublist]]
    for i in ungrouped_labels:
        grouped_counts[labels[i]] = label_counts[labels[i]]
        
    return dict(grouped_counts)



def generate_word_cloud(label_counts, unique_websites, mask_path=None, image_color=False):
    """
    Generates a word cloud from the given label counts and unique websites.

    Args:
        label_counts (dict): A dictionary containing labels as keys and their counts as values.
        unique_websites (list): A list of unique websites.
        mask_path (str, optional): Path to a mask image. If None, no mask is used. Defaults to None.
    """
    # Group similar terms
    grouped_label_counts = group_similar_terms(label_counts)

    # Combine label counts and website information into a single dictionary
    text_data = " ".join([f"{label} " * count for label, count in grouped_label_counts.items()])
    website_text = " ".join(unique_websites)
    combined_text = text_data + " " + website_text

    # Add common words that might not be relevant to the stop words list
    additional_stopwords = ["chair", "issues", "uncomfortable", "problem", "problems", "lack", "support", "seat", "padding", "material", "quality", "durability", "design", "user", "users"]
    STOPWORDS.update(additional_stopwords)

    # Load the mask image if provided
    mask = None
    if mask_path:
        mask = np.array(Image.open(mask_path))

    # Generate word cloud
    wc = WordCloud(
        background_color="white",
        max_words=200,
        stopwords=STOPWORDS,
        mask=mask,
        contour_width=3,
        contour_color="steelblue",
        color_func=None if image_color else lambda *args, **kwargs: "steelblue"
    )
    wc.generate(combined_text)

    # If color from image is true, we update the color function.
    if image_color and mask_path:
        def color_func(word, **kwargs):
            return ImageColorGenerator(mask).get_color(word)
        wc.recolor(color_func=color_func)

    # Display the word cloud
    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()



def create_and_display_wordcloud(data):
    """
    Creates and displays a word cloud from the input data.

    Args:
        data (dict): A dictionary containing 'label_counts' and 'unique_websites'.
    """
    label_counts = data.get('label_counts', {})
    unique_websites = data.get('unique_websites', [])
    generate_word_cloud(label_counts, unique_websites)

if __name__ == '__main__':
    # Example data (replace with your actual data)
    data = {
        'label_counts': {
            'chair may hurt back': 250, 'forward tilt issue': 121, 'unadjustable lumbar support': 119,
            'rigid seat': 119, 'uncomfortable after extended use': 112, 'tilt function missing': 111,
            'padding compresses': 97, 'uncomfortable seat frame': 85, 'chair needs break-in period': 81,
            'overpriced compared to quality': 79, 'smell from chair materials': 75, 'ergonomic support lacking': 67,
            'material quality concerns': 66, 'uncomfortable seating experience': 52, 'poor quality control': 52,
            'durability issues reported': 51, 'backrest not secure': 49, 'confusing tilt function': 49,
            'durability issues': 45, 'difficulty adjusting back angle': 42, 'size not suitable for all heights': 41,
            'back problems': 35, 'limited adjustability': 35, 'assembly instructions unclear': 34,
            'bucketing backrest design': 33, 'lumbar support adjustment': 30, 'lack of footrest option': 30,
            'hard to rock backward': 29, 'durability questions': 22, 'difficult to transport': 22,
            'hard to reach adjustment knobs': 21, 'noisy movement during use': 19, 'lack of lumbar support': 19,
            'design causing discomfort': 19, 'backrest dent uncomfortable': 18, 'limited height adjustment': 16,
            'unreadable knob markings': 15, 'assembly tools needed': 13, 'issues with additional accessories': 13,
            'assembly difficulties': 13, 'missing footrest': 11, 'uncomfortable fabric': 11,
            'fake leather peeling issues': 11, 'uncomfortable for extended periods': 11, 'too bulky for some users': 8,
            'lack of lumbar support adjustment': 8, 'inconvenient headrest pillow placement': 6, 'design lacks lumbar support': 6,
            'armrests too far apart': 5, 'uncomfortable armrests': 3, 'noisy wheels': 3, 'assembly difficulties with plastic components': 3,
            'durability concerns with material cracking': 3, 'ineffective lumbar support location': 3, 'poor hydraulic mechanism packaging': 2,
            'assembly difficulty for shorter people': 1, 'uncomfortable headrest pillow': 1, 'uncomfortable bottom padding': 1
        },
        'unique_websites': ['youtube', 'tomshardware', 'tomsguide', 'windowscentral', 'wired']
    }

    # Generate and display the word cloud
    create_and_display_wordcloud(data)
