#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torchinfo import summary 


# In[ ]:


import os


# In[ ]:


from torch.utils.data import Dataset, DataLoader, Subset


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchinfo import summary  # For model summary
from tqdm import tqdm  # For progress bars


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    classification_report,
    confusion_matrix
)


# In[ ]:


import pandas as pd

from data_processing import load_and_merge_data, prepare_labels


# # Import data

# In[ ]:


patient_list = [
    4096, 4098, 4101, 4102, 4103, 4104, 4105, 4114, 4116, 4118, 4121, 4125, 
    4126, 4139, 51252, 53339, 55395, 55441, 41172, 51440, 51518, 55615, 
    53595, 51551, 51625, 51632, 41410, 41411, 41412, 41420, 41488, 57887, 
    41519, 41522, 55875, 51844, 58030, 53988, 51971, 41749, 41767, 
    52062, 50027, 50044, 54144, 50086, 52146, 54197, 3000, 3003, 3004, 
    3008, 3009, 3013, 3016, 3023, 3026, 3029, 50157, 3053, 3055, 3057, 
    3060, 3062, 3064, 54265, 3069, 50175, 3072, 3073, 3071, 3075, 3076, 
    41989, 3083, 3085, 50192, 50195, 42009, 3106, 3108, 3111, 3112, 3113, 
    3114, 3115, 58420, 3125, 3126, 3128, 3130, 40012, 3157, 3161, 3169, 
    3173, 3186, 56435, 3188, 3191, 3200, 3201, 3205, 3207, 3213, 3214, 
    3215, 3216, 50319, 3218, 3219, 58510, 3221, 3227, 3237, 42164, 42171, 
    3267, 3268, 3269, 3271, 3276, 3279, 3282, 3284, 3301, 3305, 3309, 
    56558, 3314, 3316, 3318, 3320, 3321, 3323, 3325, 3330, 3333, 3350, 
    3351, 3357, 3360, 3361, 3362, 3363, 3366, 3368, 3369, 3373, 3374, 
    3376, 52530, 3380, 3383, 3387, 3389, 3390, 3400, 3404, 50509, 40273, 
    3411, 3415, 3418, 3424, 3428, 3429, 3430, 3435, 3436, 3439, 3446, 
    3452, 3453, 3454, 3457, 3458, 3460, 3462, 3464, 3467, 3468, 40338, 
    3476, 3480, 3481, 56744, 40366, 56761, 3514, 3515, 50621, 3517, 
    3519, 3523, 52678, 3527, 3541, 3543, 3544, 3556, 3558, 3563, 3564, 
    3565, 3567, 3571, 3572, 3588, 3592, 3611, 3613, 3615, 3616, 3620, 
    3624, 3625, 3630, 3632, 3633, 3634, 3636, 3637, 3650, 3651, 3654, 
    3661, 40553, 3702, 3704, 3708, 40578, 50829, 3754, 3756, 3759, 
    3765, 3776, 3778, 3780, 3794, 50901, 3800, 3803, 3804, 3805, 3806, 
    3807, 3808, 40671, 3812, 3815, 3819, 3823, 3824, 40690, 3830, 
    40694, 3832, 40703, 40704, 40707, 40709, 40713, 3850, 40714, 3852, 
    3853, 3858, 3859, 3867, 50983, 3900, 3901, 3905, 3907, 3911, 
    3917, 40781, 55124, 40806, 3950, 3952, 3963, 3967, 3969, 4004, 
    4011, 4018, 4019, 40882, 4022, 4032, 4034, 4037, 4038, 55251, 
    4071, 4074, 4075, 4077, 51186, 4090, 4091, 4092, 4093
]


# In[ ]:


features = load_and_merge_data(patient_list)
X, y = prepare_labels(features, "../PreProcessed_Data/Y.csv")

hematology_indices = list(range(0, 38))
meta1_indices = list(range(38, 38+826))
meta2_indices = list(range(38+826, 38+826+190))
protein_indices = list(range(38+826+190, 38+826+190+4785))
rnaseq_indices = list(range(38+826+190+4785, 38+826+190+4785+10652))
image_indices = list(range(38+826+190+4785+10652, 38+826+190+4785+10652+8))


# # Encoder with cross modal attention layer

# In[ ]:


num_samples = 306

dimensions = {
    'hematology': 38,
    'protein': 4785,
    'rna_seq': 10652,
    'image': 8,
    'metabolomics_part1': 826,
    'metabolomics_part2': 190
}

# Using indices for slicing
hematology_data = np.array(X.iloc[:, hematology_indices]).astype(np.float32)    #with 38 columns
metabolomics_part1_data = np.array(X.iloc[:, meta1_indices]).astype(np.float32) #with 826 columns
metabolomics_part2_data = np.array(X.iloc[:, meta2_indices]).astype(np.float32) #with 190 columns
protein_data = np.array(X.iloc[:, protein_indices]).astype(np.float32)          #with 4785 columns
rna_seq_data = np.array(X.iloc[:, rnaseq_indices]).astype(np.float32)           #with 10652 columns
image_data = np.array(X.iloc[:, image_indices]).astype(np.float32)              #with 8 columns

labels = np.array(y).astype(np.long)


# In[ ]:


# 2. Dataset Class
class ParkinsonDataset(Dataset):
    def __init__(self, hematology, protein, rna_seq, image, metabolomics_part1, metabolomics_part2, labels):
        self.hematology = hematology
        self.protein = protein
        self.rna_seq = rna_seq
        self.image = image
        self.metabolomics_part1 = metabolomics_part1
        self.metabolomics_part2 = metabolomics_part2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hematology = self.hematology[idx]
        protein = self.protein[idx]
        rna_seq = self.rna_seq[idx]
        image = self.image[idx]
        metabolomics_part1 = self.metabolomics_part1[idx]
        metabolomics_part2 = self.metabolomics_part2[idx]
        label = self.labels[idx]

        return {
            'hematology': torch.tensor(hematology, dtype=torch.float32),
            'protein': torch.tensor(protein, dtype=torch.float32),
            'rna_seq': torch.tensor(rna_seq, dtype=torch.float32),
            'image': torch.tensor(image, dtype=torch.float32),
            'metabolomics_part1': torch.tensor(metabolomics_part1, dtype=torch.float32),
            'metabolomics_part2': torch.tensor(metabolomics_part2, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


# In[ ]:





# In[ ]:


# Initialize the dataset
full_dataset = ParkinsonDataset(
    hematology=hematology_data,
    protein=protein_data,
    rna_seq=rna_seq_data,
    image=image_data,
    metabolomics_part1=metabolomics_part1_data,
    metabolomics_part2=metabolomics_part2_data,
    labels=labels
)


# In[ ]:





# In[ ]:


# 4. Cross-Attention Module
class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionModule, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        """
        query: [seq_len_query, batch_size, embed_dim]
        key: [seq_len_key, batch_size, embed_dim]
        value: [seq_len_key, batch_size, embed_dim]
        """
        attn_output, attn_weights = self.cross_attn(query, key, value)  # attn_output: [seq_len_query, batch_size, embed_dim]
        query = self.layer_norm1(attn_output + query)  # Residual connection and LayerNorm
        ffn_output = self.ffn(query)  # [seq_len_query, batch_size, embed_dim]
        out = self.layer_norm2(ffn_output + query)  # Another Residual and LayerNorm
        return out, attn_weights


# In[ ]:


class MultiModalTransformerWithCrossAttention(nn.Module):
    def __init__(self, 
                 hematology_dim=38, 
                 protein_dim=4785, 
                 rna_seq_dim=10652, 
                 image_dim=8, 
                 metabolomics_part1_dim=826, 
                 metabolomics_part2_dim=190,
                 transformer_dim=512, 
                 num_heads=8, 
                 num_cross_attn_layers=1,
                 num_transformer_layers=4, 
                 num_classes=2, 
                 dropout=0.1):
        super(MultiModalTransformerWithCrossAttention, self).__init__()
        
        # Linear layers to project each modality to transformer_dim
        self.hematology_fc = nn.Linear(hematology_dim, transformer_dim)
        self.protein_fc = nn.Linear(protein_dim, transformer_dim)
        self.rna_seq_fc = nn.Linear(rna_seq_dim, transformer_dim)
        self.image_fc = nn.Linear(image_dim, transformer_dim)
        self.metabolomics_part1_fc = nn.Linear(metabolomics_part1_dim, transformer_dim)
        self.metabolomics_part2_fc = nn.Linear(metabolomics_part2_dim, transformer_dim)
        
        # Learnable scalar weights for each modality
        self.weight_hematology = nn.Parameter(torch.ones(1))
        self.weight_protein = nn.Parameter(torch.ones(1))
        self.weight_rna_seq = nn.Parameter(torch.ones(1))
        self.weight_image = nn.Parameter(torch.ones(1))
        self.weight_metabolomics_part1 = nn.Parameter(torch.ones(1))
        self.weight_metabolomics_part2 = nn.Parameter(torch.ones(1))
        
        # Cross-Attention Modules for each modality
        self.cross_attention_hematology = nn.ModuleList([
            CrossAttentionModule(embed_dim=transformer_dim, num_heads=num_heads, dropout=dropout) 
            for _ in range(num_cross_attn_layers)
        ])
        self.cross_attention_protein = nn.ModuleList([
            CrossAttentionModule(embed_dim=transformer_dim, num_heads=num_heads, dropout=dropout) 
            for _ in range(num_cross_attn_layers)
        ])
        self.cross_attention_rna_seq = nn.ModuleList([
            CrossAttentionModule(embed_dim=transformer_dim, num_heads=num_heads, dropout=dropout) 
            for _ in range(num_cross_attn_layers)
        ])
        self.cross_attention_image = nn.ModuleList([
            CrossAttentionModule(embed_dim=transformer_dim, num_heads=num_heads, dropout=dropout) 
            for _ in range(num_cross_attn_layers)
        ])
        self.cross_attention_metabolomics_part1 = nn.ModuleList([
            CrossAttentionModule(embed_dim=transformer_dim, num_heads=num_heads, dropout=dropout) 
            for _ in range(num_cross_attn_layers)
        ])
        self.cross_attention_metabolomics_part2 = nn.ModuleList([
            CrossAttentionModule(embed_dim=transformer_dim, num_heads=num_heads, dropout=dropout) 
            for _ in range(num_cross_attn_layers)
        ])
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, hematology, protein, rna_seq, image, metabolomics_part1, metabolomics_part2):
        batch_size = hematology.size(0)
        
        # Project each modality to transformer_dim
        hematology_emb = self.hematology_fc(hematology) * self.weight_hematology  # [batch_size, transformer_dim]
        protein_emb = self.protein_fc(protein) * self.weight_protein
        rna_seq_emb = self.rna_seq_fc(rna_seq) * self.weight_rna_seq
        image_emb = self.image_fc(image) * self.weight_image
        metabolomics_part1_emb = self.metabolomics_part1_fc(metabolomics_part1) * self.weight_metabolomics_part1
        metabolomics_part2_emb = self.metabolomics_part2_fc(metabolomics_part2) * self.weight_metabolomics_part2
        
        # Reshape for Transformer: [seq_len, batch_size, embed_dim]
        # Here, seq_len=1 since each modality is represented as a single embedding
        hematology_emb = hematology_emb.unsqueeze(0)        # [1, batch_size, transformer_dim]
        protein_emb = protein_emb.unsqueeze(0)
        rna_seq_emb = rna_seq_emb.unsqueeze(0)
        image_emb = image_emb.unsqueeze(0)
        metabolomics_part1_emb = metabolomics_part1_emb.unsqueeze(0)
        metabolomics_part2_emb = metabolomics_part2_emb.unsqueeze(0)
        
        # List of all modality embeddings
        modality_embeddings = {
            'hematology': hematology_emb,
            'protein': protein_emb,
            'rna_seq': rna_seq_emb,
            'image': image_emb,
            'metabolomics_part1': metabolomics_part1_emb,
            'metabolomics_part2': metabolomics_part2_emb
        }
        
        # Apply cross-attention for each modality
        # Each modality attends to all other modalities
        for layer_idx in range(len(self.cross_attention_hematology)):
            # Hematology attends to other modalities
            other_modalities = {k: v for k, v in modality_embeddings.items() if k != 'hematology'}
            other_emb = torch.cat(list(other_modalities.values()), dim=0)  # [5, batch_size, transformer_dim]
            modality_embeddings['hematology'], _ = self.cross_attention_hematology[layer_idx](
                modality_embeddings['hematology'], other_emb, other_emb
            )
            
            # Protein attends to other modalities
            other_modalities = {k: v for k, v in modality_embeddings.items() if k != 'protein'}
            other_emb = torch.cat(list(other_modalities.values()), dim=0)
            modality_embeddings['protein'], _ = self.cross_attention_protein[layer_idx](
                modality_embeddings['protein'], other_emb, other_emb
            )
            
            # RNAseq attends to other modalities
            other_modalities = {k: v for k, v in modality_embeddings.items() if k != 'rna_seq'}
            other_emb = torch.cat(list(other_modalities.values()), dim=0)
            modality_embeddings['rna_seq'], _ = self.cross_attention_rna_seq[layer_idx](
                modality_embeddings['rna_seq'], other_emb, other_emb
            )
            
            # Image attends to other modalities
            other_modalities = {k: v for k, v in modality_embeddings.items() if k != 'image'}
            other_emb = torch.cat(list(other_modalities.values()), dim=0)
            modality_embeddings['image'], _ = self.cross_attention_image[layer_idx](
                modality_embeddings['image'], other_emb, other_emb
            )
            
            # Metabolomics Part1 attends to other modalities
            other_modalities = {k: v for k, v in modality_embeddings.items() if k != 'metabolomics_part1'}
            other_emb = torch.cat(list(other_modalities.values()), dim=0)
            modality_embeddings['metabolomics_part1'], _ = self.cross_attention_metabolomics_part1[layer_idx](
                modality_embeddings['metabolomics_part1'], other_emb, other_emb
            )
            
            # Metabolomics Part2 attends to other modalities
            other_modalities = {k: v for k, v in modality_embeddings.items() if k != 'metabolomics_part2'}
            other_emb = torch.cat(list(other_modalities.values()), dim=0)
            modality_embeddings['metabolomics_part2'], _ = self.cross_attention_metabolomics_part2[layer_idx](
                modality_embeddings['metabolomics_part2'], other_emb, other_emb
            )
        
        # Combine all cross-attended modality embeddings
        combined_emb = torch.cat(list(modality_embeddings.values()), dim=0)  # [6, batch_size, transformer_dim]
        
        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(combined_emb)  # [6, batch_size, transformer_dim]
        
        # Aggregate Transformer outputs (e.g., mean pooling)
        aggregated = transformer_output.mean(dim=0)  # [batch_size, transformer_dim]
        
        # Pass through Classification Head
        logits = self.classifier(aggregated)  # [batch_size, num_classes]
        
        return logits


# In[ ]:


class EarlyStopping:
    def __init__(self, patience=5, verbose=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose > 0:  # Print message only if verbose is greater than 0
                    print("Early stopping activated!")


# In[ ]:


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for batch in train_loader:
        # Move data to device
        hematology = batch['hematology'].to(device)
        protein = batch['protein'].to(device)
        rna_seq = batch['rna_seq'].to(device)
        image = batch['image'].to(device)
        metabolomics_part1 = batch['metabolomics_part1'].to(device)
        metabolomics_part2 = batch['metabolomics_part2'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(hematology, protein, rna_seq, image, metabolomics_part1, metabolomics_part2)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item() * hematology.size(0)
    
    # Compute average loss
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


# In[ ]:


def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    preds = []
    truths = []
    probs = []  # To store probabilities for ROC-AUC
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            hematology = batch['hematology'].to(device)
            protein = batch['protein'].to(device)
            rna_seq = batch['rna_seq'].to(device)
            image = batch['image'].to(device)
            metabolomics_part1 = batch['metabolomics_part1'].to(device)
            metabolomics_part2 = batch['metabolomics_part2'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(hematology, protein, rna_seq, image, metabolomics_part1, metabolomics_part2)
            
            # Get probabilities by applying softmax
            probabilities = nn.functional.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            probs.extend(probabilities.cpu().numpy())
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            truths.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(truths, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truths, preds, average='weighted'
    )
    roc_auc = roc_auc_score(truths, probs, average='weighted')
    
    return accuracy, precision, recall, f1, roc_auc


# In[ ]:


# Define the number of splits
num_splits = 5

# Initialize StratifiedShuffleSplit
splitter = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
roc_auc_list = []

# Directory to save the best models
os.makedirs('saved_models', exist_ok=True)


# In[ ]:


# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

for split_idx, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(full_dataset)), full_dataset.labels)):
    print(f"\n--- Split {split_idx + 1}/{num_splits} ---")
    
    # Create training and testing subsets
    train_subset = Subset(full_dataset, train_idx)
    test_subset = Subset(full_dataset, test_idx)
    
    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize the model
    model = MultiModalTransformerWithCrossAttention(
        hematology_dim=dimensions['hematology'],
        protein_dim=dimensions['protein'],
        rna_seq_dim=dimensions['rna_seq'],
        image_dim=dimensions['image'],
        metabolomics_part1_dim=dimensions['metabolomics_part1'],
        metabolomics_part2_dim=dimensions['metabolomics_part2'],
        transformer_dim=512,
        num_heads=8,
        num_cross_attn_layers=1,  # Adjust as needed
        num_transformer_layers=6,  # Adjust as needed
        num_classes=2,
        dropout=0.1
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Variables to track the best model
    best_accuracy = 0.0
    best_model_path = f'saved_models/best_model_split_{split_idx + 1}.pth'
    
    # Training loop for this split
    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    
    # Evaluation
    accuracy, precision, recall, f1, roc_auc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Weighted ROC-AUC: {roc_auc:.4f}")
    
    # Save the model if it's the best so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model for Split {split_idx + 1} saved to {best_model_path}")
    
    # Append metrics to the lists
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    roc_auc_list.append(roc_auc)
    
    # Free up memory
    del model
    torch.cuda.empty_cache()


# In[ ]:


# Calculate average metrics
avg_accuracy = np.mean(accuracy_list)
std_accuracy = np.std(accuracy_list)

avg_precision = np.mean(precision_list)
std_precision = np.std(precision_list)

avg_recall = np.mean(recall_list)
std_recall = np.std(recall_list)

avg_f1 = np.mean(f1_list)
std_f1 = np.std(f1_list)

avg_roc_auc = np.mean(roc_auc_list)
std_roc_auc = np.std(roc_auc_list)

# Print the averaged metrics
print("\n=== Average Metrics over 10 Splits ===")
print(f"Accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
print(f"Weighted Precision: {avg_precision:.3f} ± {std_precision:.3f}")
print(f"Weighted Recall: {avg_recall:.3f} ± {std_recall:.3f}")
print(f"Weighted F1 Score: {avg_f1:.3f} ± {std_f1:.3f}")
print(f"Weighted ROC-AUC: {avg_roc_auc:.3f} ± {std_roc_auc:.3f}")


# In[ ]:




