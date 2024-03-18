import torch
import torch.nn as nn
from pytorch_metric_learning import losses

# Define Neural Network for product recommendation
class ProductRecommendationNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ProductRecommendationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Custom hybrid loss combining contrastive loss and classification loss
class CustomLoss(nn.Module):
    def __init__(self, input_size, num_classes, margin=1.0, scale=0.1):
        super(CustomLoss, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.contrastive_loss = losses.ContrastiveLoss(margin=margin)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        # Split embeddings into anchor and positive samples (for contrastive loss)
        anchor_embeddings = embeddings[0::2]
        positive_embeddings = embeddings[1::2]

        # Calculate contrastive loss
        contrastive_loss = self.contrastive_loss(anchor_embeddings, positive_embeddings, labels)

        # Calculate classification loss
        logits = self.fc(embeddings)
        class_loss = self.ce_loss(logits, labels)

        # Total loss as a combination of contrastive loss and classification loss
        total_loss = contrastive_loss + self.scale * class_loss
        return total_loss

# Example usage in training loop
model = ProductRecommendationNet(input_size=128, num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
custom_loss = CustomLoss(input_size=128, num_classes=10)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        embeddings = model(inputs)
        loss = custom_loss(embeddings, labels)
        loss.backward()
        optimizer.step()
