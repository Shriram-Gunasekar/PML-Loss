import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.transforms import Normalize

# Define Image Style Transfer Network
class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        self.model = vgg19(pretrained=True).features
        self.model = nn.Sequential(*list(self.model.children())[:31])
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.normalization = Normalize(mean=self.normalization_mean, std=self.normalization_std)

    def forward(self, x):
        x = self.normalization(x)
        features = self.model(x)
        return features

# Custom loss combining style loss and content loss
class CustomLoss(nn.Module):
    def __init__(self, style_targets, content_targets, style_weight=1000, content_weight=1):
        super(CustomLoss, self).__init__()
        self.style_targets = style_targets
        self.content_targets = content_targets
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.mse_loss = nn.MSELoss()

    def gram_matrix(self, input):
        batch_size, channels, h, w = input.size()
        features = input.view(batch_size * channels, h * w)
        gram_matrix = torch.mm(features, features.t())
        return gram_matrix.div(batch_size * channels * h * w)

    def style_loss(self, input, target):
        input_gram = self.gram_matrix(input)
        target_gram = self.gram_matrix(target)
        return self.mse_loss(input_gram, target_gram)

    def content_loss(self, input, target):
        return self.mse_loss(input, target)

    def forward(self, output_features, style_features, content_features):
        style_loss = sum(self.style_loss(output, target) for output, target in zip(style_features, self.style_targets))
        content_loss = sum(self.content_loss(output, target) for output, target in zip(content_features, self.content_targets))
        total_loss = self.style_weight * style_loss + self.content_weight * content_loss
        return total_loss

# Example usage in training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StyleTransferNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Assuming style_targets, content_targets, and dataloader are defined
custom_loss = CustomLoss(style_targets, content_targets)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        inputs, _ = batch
        output_features = model(inputs)
        style_features = model(style_images)
        content_features = model(content_images)
        loss = custom_loss(output_features, style_features, content_features)
        loss.backward()
        optimizer.step()
