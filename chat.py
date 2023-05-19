import json
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the intents from the JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']

# Load the BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intents))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the trained model weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Define the chat function
def chatbot_response(text):
    # Tokenize the input text
    tokens = tokenizer.encode(text, truncation=True, padding=True, return_tensors='pt')
    tokens = tokens.to(device)

    # Get the model predictions
    with torch.no_grad():
        outputs = model(tokens)
        _, predicted_class = torch.max(outputs.logits, dim=1)

    # Get the corresponding intent tag
    intent_tag = intents[predicted_class.item()]['tag']

    # Get a random response from the matched intent
    responses = intents[predicted_class.item()]['responses']
    response = random.choice(responses)

    return intent_tag, response

# Test the chatbot
while True:
    user_input = input("User: ")
    intent_tag, bot_response = chatbot_response(user_input)
    print("ChatBot:", bot_response)
    if intent_tag == 'goodbye':
        break
