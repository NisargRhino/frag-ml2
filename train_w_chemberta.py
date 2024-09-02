# Import necessary libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Load the datasets
data_file_path = "akt1canonical.csv"  # Update this with your dataset path
predetermined_smiles_path = "unique_smile4.csv"  # Path to predetermined SMILES

# Load main dataset
df = pd.read_csv(data_file_path, nrows =100)

# Load predetermined SMILES and create a label encoder
predetermined_df = pd.read_csv(predetermined_smiles_path)
predetermined_smiles = predetermined_df['SMILES'].unique().tolist()

# Filter dataset to include only rows where fragment SMILES are in predetermined list
df_filtered = df[df['FRAG_SMILES'].isin(predetermined_smiles)].reset_index(drop=True)

print(f"Total samples after filtering: {len(df_filtered)}")

# Encode fragment SMILES to numerical labels
label_encoder = LabelEncoder()
df_filtered['label'] = label_encoder.fit_transform(df_filtered['FRAG_SMILES'])

num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')

# Custom Dataset Class
class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        drug_smiles = self.dataframe.iloc[idx]['DRUG SMILES']
        label = self.dataframe.iloc[idx]['label']

        # Tokenize the drug SMILES
        encoding = self.tokenizer(
            drug_smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        inputs = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        return inputs

# Create dataset and split into train, validation, and test sets
dataset = SMILESDataset(df_filtered, tokenizer)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42)
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Create DataLoaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Model
class ChemBERTaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ChemBERTaClassifier, self).__init__()
        self.chemberta = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.chemberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Initialize the model
model = ChemBERTaClassifier(num_classes)
model.to(device)

# Define optimizer, loss function, and scheduler
learning_rate = 2e-5
epochs = 10

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training and Validation Function
def train_epoch(model, data_loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    total_correct = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        correct = torch.sum(preds == labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_correct += correct.item()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / (len(data_loader.dataset))
    return avg_loss, accuracy

def eval_epoch(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    tanimoto_similarities = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            correct = torch.sum(preds == labels)

            total_loss += loss.item()
            total_correct += correct.item()

            # Calculate Tanimoto similarity
            for i, pred in enumerate(preds):
                pred_frag_smiles = label_encoder.inverse_transform([pred.item()])[0]
                true_frag_smiles = label_encoder.inverse_transform([labels[i].item()])[0]

                pred_mol = Chem.MolFromSmiles(pred_frag_smiles)
                true_mol = Chem.MolFromSmiles(true_frag_smiles)

                if pred_mol and true_mol:
                    pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)
                    true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
                    tanimoto_sim = DataStructs.TanimotoSimilarity(pred_fp, true_fp)
                    tanimoto_similarities.append(tanimoto_sim)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / (len(data_loader.dataset))
    avg_tanimoto = np.mean(tanimoto_similarities) if tanimoto_similarities else 0
    return avg_loss, accuracy, avg_tanimoto

# Training Loop
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
val_tanimotos = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler)
    val_loss, val_acc, val_tanimoto = eval_epoch(model, val_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_tanimotos.append(val_tanimoto)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%, Tanimoto Similarity: {val_tanimoto:.4f}")
    print("-" * 50)

# Evaluate on Test Set
test_loss, test_acc, test_tanimoto = eval_epoch(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%, Test Tanimoto Similarity: {test_tanimoto:.4f}")

# Save the trained model
model_save_path = "./chembertatrained.pkl"
with open(model_save_path, 'wb') as file:
    pickle.dump(model, file)

    # Plotting training and validation accuracy
plt.figure(figsize=(10, 5))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), [acc * 100 for acc in train_accuracies], label="Training Accuracy")
plt.plot(range(1, epochs + 1), [acc * 100 for acc in val_accuracies], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy")
plt.legend()

# Plot validation Tanimoto similarity
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), val_tanimotos, label="Validation Tanimoto Similarity", color="green")
plt.xlabel("Epoch")
plt.ylabel("Tanimoto Similarity")
plt.title("Validation Tanimoto Similarity")
plt.legend()

plt.tight_layout()
plt.show()
