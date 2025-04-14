import torch.nn as nn
import torch
def increment_class_labels(model, new_num_labels):
    """
    Increment the number of class labels in the classifier of the model.

    Args:
    - model: The current model (BertForSequenceClassification).
    - new_num_labels: The new total number of labels.


    - Updated model with the new classification head.
    """
    old_num_labels = model.num_labels
    if new_num_labels <= old_num_labels:
        raise ValueError("new_num_labels must be greater than the current num_labels.")

    # Expand classifier weights
    old_weights = model.classifier.weight.data
    old_bias = model.classifier.bias.data if model.classifier.bias is not None else None

    # Get hidden size from the existing classifier layer
    hidden_size = model.classifier.in_features  # Get hidden size from the existing classifier

    # Create a new classification layer with the updated number of labels
    model.classifier = nn.Linear(hidden_size, new_num_labels)  # Use the hidden size from the existing layer
    model.num_labels = new_num_labels

    # Copy old weights to the new classifier
    with torch.no_grad():
        model.classifier.weight[:old_num_labels] = old_weights
        if old_bias is not None:
            model.classifier.bias[:old_num_labels] = old_bias

    # Return the updated model
    return model # Added this line to return the updated model