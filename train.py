import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader

# Load the intents from the JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']

# Generate the training data
training_data = []
for intent in intents:
    for pattern in intent['patterns']:
        training_data.append((pattern, intent['tag']))

# Save the training data to a file
with open('training_data.txt', 'w') as file:
    for example in training_data:
        file.write(f'{example[0]}\t{example[1]}\n')
# Define the conversation dataset class
class ConversationDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.examples = self.load_examples(file_path)
        self.tokenizer = tokenizer
        self.classes = self.get_classes()
    
    def load_examples(self, file_path):
        examples = []
        with open(file_path, 'r') as file:
            for line in file:
                pattern, tag = line.strip().split('\t')
                examples.append((pattern, tag))
        return examples
    
    def get_classes(self):
        classes = set()
        for example in self.examples:
            classes.add(example[1])
        return list(classes)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        pattern, tag = self.examples[index]
        
        inputs = self.tokenizer.encode_plus(pattern, add_special_tokens=True, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'tag': self.classes.index(tag)}

# Set the paths and parameters
training_file = 'training_data.txt'
model_save_path = 'bert_model.pt'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch_size = 4
num_epochs = 5
learning_rate = 2e-5

# Load the dataset
dataset = ConversationDataset(training_file, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset.classes))

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tags = batch['tag'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tags)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {average_loss:.4f}')

# Save the trained model
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
