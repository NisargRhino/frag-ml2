import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

model_save_path = "./chembertatrained.pkl"
with open(model_save_path, 'rb') as file:
    model = pickle.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')



data_file_path = "akt1canonical.csv" 
predetermined_smiles_path = "unique_smiles.csv"  


df = pd.read_csv(data_file_path)

predetermined_df = pd.read_csv(predetermined_smiles_path)
predetermined_smiles = predetermined_df['SMILES'].unique().tolist()

df_filtered = df[df['FRAG_SMILES'].isin(predetermined_smiles)].reset_index(drop=True)

label_encoder = LabelEncoder()
df_filtered['label'] = label_encoder.fit_transform(df_filtered['FRAG_SMILES'])

def predict_fragment(smiles):
    encoding = tokenizer(
        smiles,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
    
    predicted_class = torch.argmax(logits, dim=1).item()
    
    predicted_fragment = label_encoder.inverse_transform([predicted_class])[0]
    
    return predicted_fragment

smiles_input = "CC(=O)NCC1CN(c2ccc(-c3ccno3)c(F)c2)C(=O)O1"  
predicted_fragment = predict_fragment(smiles_input)
print(f"Input SMILES: {smiles_input}")
print(f"Predicted Fragment SMILES: {predicted_fragment}")
