import spacy
import numpy as np
import pandas as pd

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def sentence_to_dependency_matrix(sentence):
    """
    Convert a sentence to a dependency matrix
    Args:
        sentence (str): Input sentence
    Returns:
        words (list): List of words in the sentence
        matrix (np.ndarray): Dependency matrix
    """
    # Parse the sentence
    doc = nlp(sentence)
    words = [token.text for token in doc]
    num_words = len(words)

    # Initialize a matrix with zeros
    matrix = np.zeros((num_words, num_words), dtype=int)

    # Populate the matrix with dependencies
    for token in doc:
        if token.head.i != token.i:  # Skip self-loops
            matrix[token.i][token.head.i] = 1  # Dependency from child to head
            matrix[token.head.i][token.i] = 1  # Symmetric relationship (optional)

    return words, matrix

def prepare_dependency_matrix(sentence):
    """
    Prepare a dependency matrix for a given sentence
    Args:
        sentence (str): Input sentence
    Returns: 
        df (pd.DataFrame): Dependency matrix as a DataFrame
        words (list): List of words in the
    """
    words, matrix = sentence_to_dependency_matrix(sentence)

    # Convert to DataFrame for better visualization
    df = pd.DataFrame(matrix, index=words, columns=words)
    return df, words


if  __name__ == "__main__":

    sentence = "The quick brown_fox jumps over the lazy_dog."
    df, words = prepare_dependency_matrix(sentence)
    print(df)
    print(words)

