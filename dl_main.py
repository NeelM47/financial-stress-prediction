import os, random, warnings, re
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

seed_everything(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if 
                      torch.backends.mps.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")
sub = pd.read_csv("data/SampleSubmission.csv")

TARGET = "liquidity_stress_next_30d"
y = train[TARGET].values

train["profile_hash"] = train["age"].astype(str) + "_" + train["gender"].astype(str) + "_" + train["region"].astype(str)
groups = train["profile_hash"].values
cat_cols = ["gender", "region", "smartphone", "segment", "earning_pattern"]
static_num_cols = ["arpu", "age", "x_90_d_activity_rate"]

for df in [train, test]:
    for col in cat_cols:
        df[col] = df[col].fillna("Missing").astype(str)
        
    for col in static_num_cols:
        df[col] = df[col].fillna(0)

cat_dims = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    cat_dims[col] = len(le.classes_)

scaler_static = StandardScaler()
train[static_num_cols] = scaler_static.fit_transform(np.log1p(np.maximum(train[static_num_cols], 0)))
test[static_num_cols] = scaler_static.transform(np.log1p(np.maximum(test[static_num_cols], 0)))

all_cols = train.columns.tolist()
seq_cols = [c for c in all_cols if re.match(r"^m[1-6]_", c)]
base_seq_names = sorted(list(set([re.sub(r"^m[1-6]_", "", c) for c in seq_cols])))
num_seq_features = len(base_seq_names)


print(f"Found {num_seq_features} temporal features per month.")

def build_sequences(df):
    N = len(df)
    seq_data = np.zeros((N, 6, num_seq_features))
    for t_idx, month in enumerate([6, 5, 4, 3, 2, 1]):
        for f_idx, base_feat in enumerate(base_seq_names):
            col_name = f"m{month}_{base_feat}"
            if col_name in df.columns:
                seq_data[:, t_idx, f_idx] = df[col_name].fillna(0).values

    seq_data = np.log1p(np.maximum(seq_data, 0))
    return seq_data

train_seq = build_sequences(train)
test_seq = build_sequences(test)

train_seq_flat = train_seq.reshape(-1, num_seq_features)
test_seq_flat = test_seq.reshape(-1, num_seq_features)
scaler_seq = StandardScaler()
train_seq_flat = scaler_seq.fit_transform(train_seq_flat)
test_seq_flat = scaler_seq.transform(test_seq_flat)
train_seq = train_seq_flat.reshape(-1, 6, num_seq_features)
test_seq = test_seq_flat.reshape(-1, 6, num_seq_features)

X_cat_train = train[cat_cols].values
X_num_train = train[static_num_cols].values
X_cat_test = test[cat_cols].values
X_num_test = test[static_num_cols].values

class FinancialDataset(Dataset):
    def __init__(self, seq_data, cat_data, num_data, targets=None):
        self.seq_data = torch.FloatTensor(seq_data)
        self.cat_data = torch.LongTensor(cat_data)
        self.num_data = torch.FloatTensor(num_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.seq_data[idx], self.cat_data[idx], self.num_data[idx], self.targets[idx]
        return self.seq_data[idx], self.cat_data[idx], self.num_data[idx]

class FinancialSequenceModel(nn.Module):
    def __init__(self, num_seq_features, cat_dims, num_static_features):
        super(FinancialSequenceModel, self).__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, min(50, (num_classes + 1) // 2))
            for num_classes in cat_dims.values()
        ])
        total_embed_dim = sum([min(50, (c + 1) // 2) for c in cat_dims.values()])

        hidden_dim = 64
        self.rnn = nn.GRU(
                input_size=num_seq_features,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2
                )
        rnn_out_dim = hidden_dim * 2
        fc_in_dim = rnn_out_dim + total_embed_dim + num_static_features
        self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )

    def forward(self, seq_x, cat_x, num_x):
        embeds = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        embeds = torch.cat(embeds, dim=1)
        _, h_n = self.rnn(seq_x)

        rnn_out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        x = torch.cat([rnn_out, embeds, num_x], dim=1)

        out = self.fc(x)
        return out.squeeze(1)

EPOCHS = 15
BATCH_SIZE = 256
gkf = GroupKFold(n_splits=5)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

test_dataset = FinancialDataset(test_seq, X_cat_test, X_num_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_seq, y, groups)):
    print(f"\n====== FOLD {fold+1} ======")

    train_dataset = FinancialDataset(train_seq[tr_idx], X_cat_train[tr_idx], X_num_train[tr_idx], y[tr_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = FinancialDataset(train_seq[val_idx], X_cat_train[val_idx], X_num_train[val_idx], y[val_idx])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FinancialSequenceModel(num_seq_features, cat_dims, len(static_num_cols)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_loss = float('inf')
    best_model_weights = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for seq_batch, cat_batch, num_batch, y_batch in train_loader:
            seq_batch, cat_batch, num_batch, y_batch = seq_batch.to(DEVICE),cat_batch.to(DEVICE), num_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(seq_batch, cat_batch, num_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for seq_batch, cat_batch, num_batch, y_batch in val_loader:
                seq_batch, cat_batch, num_batch, y_batch = seq_batch.to(DEVICE),cat_batch.to(DEVICE), num_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(seq_batch, cat_batch, num_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_probs)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()

    model.load_state_dict(best_model_weights)
    model.eval()

    fold_oof_probs = []
    with torch.no_grad():
        for seq_batch, cat_batch, num_batch, _ in val_loader:
            seq_batch, cat_batch, num_batch = seq_batch.to(DEVICE), cat_batch.to(DEVICE), num_batch.to(DEVICE)
            probs = torch.sigmoid(model(seq_batch, cat_batch, num_batch)).cpu().numpy()
            fold_oof_probs.extend(probs)
    oof_preds[val_idx] = fold_oof_probs

    fold_test_probs = []
    with torch.no_grad():
        for seq_batch, cat_batch, num_batch in test_loader:
            seq_batch, cat_batch, num_batch = seq_batch.to(DEVICE), cat_batch.to(DEVICE), num_batch.to(DEVICE)
            probs = torch.sigmoid(model(seq_batch, cat_batch, num_batch)).cpu().numpy()
            fold_test_probs.extend(probs)
    test_preds += np.array(fold_test_probs) / gkf.n_splits

print("\n====== FINAL DL RESULTS ======")
dl_ll = log_loss(y, oof_preds)
dl_auc = roc_auc_score(y, oof_preds)
print(f"Deep Learning OOF | LogLoss: {dl_ll:.5f} | ROC-AUC: {dl_auc:.5f}")

test_preds = np.clip(test_preds, 1e-5, 1 - 1e-5)
sub["Target"] = test_preds
sub.to_csv("submissions/submission_dl.csv", index=False)
print("Saved submission_dl.csv")

