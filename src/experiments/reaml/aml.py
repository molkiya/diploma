import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Define your PyTorch model class
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load and preprocess your data, split into labeled and unlabeled sets
X, y = load_data()
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.8, stratify=y)

# Convert data to PyTorch tensors
X_labeled_tensor = torch.tensor(X_labeled, dtype=torch.float32)
y_labeled_tensor = torch.tensor(y_labeled, dtype=torch.long)
labeled_dataset = TensorDataset(X_labeled_tensor, y_labeled_tensor)
labeled_dataloader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Active Learning Loop
num_iterations = 10
batch_size = 10

for iteration in range(num_iterations):
    # Implement a query strategy (e.g., uncertainty sampling)
    model.eval()
    with torch.no_grad():
        uncertainty = model(X_unlabeled_tensor)
        uncertainty_scores = torch.max(uncertainty, dim=1)[0]
    query_indices = uncertainty_scores.argsort()[-batch_size:]

    # Label the selected instances
    labeled_instances = X_unlabeled_tensor[query_indices]
    labeled_labels = get_labels_for_instances(labeled_instances)

    # Update the labeled and unlabeled datasets
    X_labeled_tensor = torch.cat((X_labeled_tensor, labeled_instances), dim=0)
    y_labeled_tensor = torch.cat((y_labeled_tensor, labeled_labels), dim=0)
    X_unlabeled_tensor = torch.cat((X_unlabeled_tensor[:query_indices[0]], X_unlabeled_tensor[query_indices[0] + 1:]),
                                   dim=0)

    # Retrain the model on the updated labeled dataset
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in labeled_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate the model on a validation set
    validation_accuracy = evaluate_model(model, X_validation_tensor, y_validation_tensor)
    print(f"Iteration {iteration + 1}, Validation Accuracy: {validation_accuracy:.4f}")