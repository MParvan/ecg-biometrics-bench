import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models_2d
from torch.autograd import Function

# =============================================================================
# 1. 1D CNN Models (Baselines)
# =============================================================================

class DeepECG(nn.Module):
    """
    PyTorch version of the DeepECG classifier.
    Updated to support both Classification (Closed-Set) and Feature Extraction.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, include_top: bool = True):
        super(DeepECG, self).__init__()
        self.include_top = include_top
        self.num_classes = num_classes

        # Feature Extractor
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Bottleneck / Embedding Layer
        self.fc1 = nn.Linear(64, 128)

        # Classifier Head
        if self.include_top:
            self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.gap(x).squeeze(-1) # (B, 64)
        embedding = F.relu(self.fc1(x)) # (B, 128)

        if self.include_top:
            return self.fc2(embedding)
        
        return embedding


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet-18 adapted for ECG.
    Typically outperforms standard CNNs on larger datasets.
    """
    def __init__(self, in_channels=1, num_classes=10, include_top=True):
        super(ResNet1D, self).__init__()
        self.include_top = include_top
        self.inplanes = 64
        
        # Initial Block
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers (ResNet-18 structure: 2, 2, 2, 2 blocks)
        self.layer1 = self._make_layer(ResNetBasicBlock, 64, 2)
        self.layer2 = self._make_layer(ResNetBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNetBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNetBasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Linear(512 * ResNetBasicBlock.expansion, 128)

        if self.include_top:
            self.fc_class = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embedding = self.relu(self.fc_embed(x)) 

        if self.include_top:
            return self.fc_class(embedding)
        return embedding


# =============================================================================
# 2. RNN & Hybrid Models
# =============================================================================

class RNN_ECG(nn.Module):
    """
    Bi-directional LSTM with a small Convolutional front-end.
    Raw ECG is too long for pure LSTM, so we downsample with Conv layers first.
    """
    def __init__(self, in_channels=1, num_classes=10, include_top=True, hidden_dim=64, num_layers=2):
        super(RNN_ECG, self).__init__()
        self.include_top = include_top
        
        # Conv Front-End
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), 
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        
        # LSTM Layers
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_dim, num_layers=num_layers, 
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        self.fc_embed = nn.Linear(hidden_dim * 2, 128)
        if self.include_top:
            self.fc_class = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x) # (B, 64, L/4)
        x = x.permute(0, 2, 1) # (B, L, C)
        
        out, (h_n, c_n) = self.lstm(x)
        
        # Concatenate the last forward and backward hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        x = torch.cat((h_forward, h_backward), dim=1) # (B, 2*hidden)
        
        embedding = self.fc_embed(x) 
        if self.include_top:
            return self.fc_class(embedding)
        return embedding


class HybridCNNLSTM(nn.Module):
    """
    Classic Hybrid architecture: 1D CNN for feature extraction -> LSTM for temporal dynamics.
    Common baseline in literature (e.g., BioECG).
    """
    def __init__(self, in_channels=1, num_classes=10, include_top=True):
        super(HybridCNNLSTM, self).__init__()
        self.include_top = include_top
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64*2, num_classes)

    def forward(self, x):
        features = self.cnn(x).permute(0, 2, 1)
        out, _ = self.lstm(features)
        embedding = torch.mean(out, dim=1) # Global Average Pooling
        
        if self.include_top:
            return self.fc(embedding)
        return embedding


# =============================================================================
# 3. Transformers & Attention
# =============================================================================

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class ECGTransformer(nn.Module):
    """
    Pure Transformer Encoder for ECG.
    Ref: Section IV-E of your survey.
    """
    def __init__(self, in_channels=1, num_classes=10, include_top=True, seq_len=5000, embed_dim=128, num_heads=4, num_layers=2):
        super(ECGTransformer, self).__init__()
        self.include_top = include_top
        
        self.conv_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=7, stride=1, padding=3)
        self.pos_enc = LearnablePositionalEncoding(seq_len, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, embed_dim*2, 0.1) for _ in range(num_layers)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        if self.include_top:
            self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Embed: (B, C, L) -> (B, Embed, L) -> (B, L, Embed)
        x = self.conv_embed(x).permute(0, 2, 1)
        
        # Add Pos Encoding (truncate to current len)
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        for block in self.transformer:
            x = block(x)
            
        x = x.permute(0, 2, 1) # Back to (B, C, L) for pooling
        emb = self.pool(x).squeeze(-1)
        
        if self.include_top:
            return self.fc(emb)
        return emb


# =============================================================================
# 4. 2D Models (For Spectrograms/Scalograms)
# =============================================================================

class ResNet18_2D(nn.Module):
    """
    Standard ResNet18 for 2D Inputs.
    Ref: Section IV-B of your survey.
    """
    def __init__(self, in_channels=1, num_classes=10, include_top=True):
        super(ResNet18_2D, self).__init__()
        self.include_top = include_top
        
        resnet = models_2d.resnet18(weights=None)
        
        # Modify first layer for 1 channel input
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.include_top:
            return self.fc(x)
        return x


# =============================================================================
# 5. Specialized Wrappers (Siamese & DANN)
# =============================================================================

class SiameseNetwork(nn.Module):
    """
    Siamese Wrapper. Takes two inputs (x1, x2).
    Ref: Section IV-F of your survey.
    """
    def __init__(self, backbone):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        self.backbone.include_top = False # Ensure we get embeddings

    def forward(self, x1, x2=None):
        emb1 = self.backbone(x1)
        if x2 is not None:
            emb2 = self.backbone(x2)
            return emb1, emb2
        return emb1

# --- Domain Adversarial Components (For Paper 2) ---

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

class DANN_ECG(nn.Module):
    """
    Domain Adversarial Neural Network for Cross-Session Adaptation.
    """
    def __init__(self, backbone, num_classes, num_domains=2, embed_dim=128):
        super(DANN_ECG, self).__init__()
        self.feature_extractor = backbone
        self.feature_extractor.include_top = False
        
        # 1. Identity Head
        self.identity_classifier = nn.Linear(embed_dim, num_classes)
        
        # 2. Domain Head (Adversary)
        self.domain_classifier = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        id_logits = self.identity_classifier(features)
        domain_logits = self.domain_classifier(features)
        return id_logits, domain_logits


# =============================================================================
# 6. Loss Functions
# =============================================================================

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.sqrt(torch.clamp(distances, min=1e-16))

        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        hardest_positive_dist = torch.max(distances * pos_mask.float(), dim=1)[0]
        
        max_dist = torch.max(distances)
        hardest_negative_dist = torch.min(distances + max_dist * pos_mask.float(), dim=1)[0]

        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        return triplet_loss.mean()