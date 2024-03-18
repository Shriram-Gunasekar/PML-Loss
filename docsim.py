import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

# Define Document Similarity Network
class DocumentSimilarityNet(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(DocumentSimilarityNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, embedding_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization for embeddings

# Custom hybrid loss combining triplet loss and cosine similarity loss
class CustomLoss(nn.Module):
    def __init__(self, margin=0.5, scale=0.1):
        super(CustomLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.triplet_loss = losses.TripletMarginLoss(margin=margin)
        self.cosine_similarity_loss = losses.CosineSimilarityLoss()

    def forward(self, embeddings, labels):
        # Calculate triplet loss
        triplet_loss = self.triplet_loss(embeddings, labels)

        # Calculate cosine similarity loss
        cosine_similarity_loss = self.cosine_similarity_loss(embeddings, labels)

        # Total loss as a combination of triplet loss and cosine similarity loss
        total_loss = triplet_loss + self.scale * cosine_similarity_loss
        return total_loss

# Example usage in training loop
model = DocumentSimilarityNet(input_size=300, embedding_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
custom_loss = CustomLoss()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        embeddings = model(inputs)
        loss = custom_loss(embeddings, labels)
        loss.backward()
        optimizer.step()
