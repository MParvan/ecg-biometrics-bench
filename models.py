import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models_2d
from torch.autograd import Function

# =============================================================================
# 1. 1D CNN Models (Baselines)
# =============================================================================

class CNN(nn.Module):
    """
    Standard 1D CNN Classifier (Previously named DeepECG).
    Supports both Classification (Closed-Set) and Feature Extraction.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, include_top: bool = True):
        super(CNN, self).__init__()
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


class DeepECG(nn.Module):
    """
    Deep-ECG CNN architecture proposed by Labati et al. (2019).
    Ref: "Deep-ECG: Convolutional Neural Networks for ECG biometric recognition"
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, include_top: bool = True):
        super(DeepECG, self).__init__()
        self.include_top = include_top

        # Layer 1: Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Layer 2: Conv -> ReLU -> LRN -> MaxPool
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        # LRN Params: D=5, k=1, alpha=2e-4, beta=0.75
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=2e-4, beta=0.75, k=1.0)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Layer 3: Conv -> ReLU -> MaxPool
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Layer 4: Conv -> ReLU -> LRN
        self.conv4 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        self.lrn4 = nn.LocalResponseNorm(size=5, alpha=2e-4, beta=0.75, k=1.0)

        # Layer 5: Conv -> ReLU
        self.conv5 = nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3)

        # Layer 6: Conv -> ReLU -> LRN -> Dropout
        self.conv6 = nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=4) 
        self.lrn6 = nn.LocalResponseNorm(size=5, alpha=2e-4, beta=0.75, k=1.0)
        self.drop = nn.Dropout(0.5)

        # Global Average Pooling to ensure sequence length independence 
        # (Safely extracts 256-dimensional real features regardless of input length)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully Connected (For Closed-Set Identification / Training)
        if self.include_top:
            self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(self.lrn2(F.relu(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.lrn4(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.drop(self.lrn6(F.relu(self.conv6(x))))

        embedding = self.gap(x).squeeze(-1) # Output shape: (Batch, 256)

        if self.include_top:
            return self.fc(embedding)
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

# --- Domain Adversarial Components ---

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












# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models_2d
# from torch.autograd import Function

# # =============================================================================
# # 1. 1D CNN Models (Baselines)
# # =============================================================================

# class CNN(nn.Module):
#     """
#     Standard 1D CNN Classifier (Previously named DeepECG).
#     Supports both Classification (Closed-Set) and Feature Extraction.
#     """
#     def __init__(self, in_channels: int = 1, num_classes: int = 10, include_top: bool = True):
#         super(CNN, self).__init__()
#         self.include_top = include_top
#         self.num_classes = num_classes

#         # Feature Extractor
#         self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
#         self.bn1 = nn.BatchNorm1d(16)
#         self.pool1 = nn.MaxPool1d(kernel_size=2)

#         self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.pool2 = nn.MaxPool1d(kernel_size=2)

#         self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(64)

#         # Global Average Pooling
#         self.gap = nn.AdaptiveAvgPool1d(1)

#         # Bottleneck / Embedding Layer
#         self.fc1 = nn.Linear(64, 128)

#         # Classifier Head
#         if self.include_top:
#             self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#         x = F.relu(self.bn3(self.conv3(x)))

#         x = self.gap(x).squeeze(-1) # (B, 64)
#         embedding = F.relu(self.fc1(x)) # (B, 128)

#         if self.include_top:
#             return self.fc2(embedding)
        
#         return embedding


# class DeepECG(nn.Module):
#     """
#     Deep-ECG CNN architecture proposed by Labati et al. (2019).
#     Ref: "Deep-ECG: Convolutional Neural Networks for ECG biometric recognition"
#     """
#     def __init__(self, in_channels: int = 1, num_classes: int = 10, include_top: bool = True):
#         super(DeepECG, self).__init__()
#         self.include_top = include_top

#         # Layer 1: Conv -> ReLU -> MaxPool
#         self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3)
#         self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # Layer 2: Conv -> ReLU -> LRN -> MaxPool
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
#         # LRN Params: D=5, k=1, alpha=2e-4, beta=0.75
#         self.lrn2 = nn.LocalResponseNorm(size=5, alpha=2e-4, beta=0.75, k=1.0)
#         self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # Layer 3: Conv -> ReLU -> MaxPool
#         self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
#         self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # Layer 4: Conv -> ReLU -> LRN
#         self.conv4 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
#         self.lrn4 = nn.LocalResponseNorm(size=5, alpha=2e-4, beta=0.75, k=1.0)

#         # Layer 5: Conv -> ReLU
#         self.conv5 = nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3)

#         # Layer 6: Conv -> ReLU -> LRN -> Dropout
#         self.conv6 = nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=4) 
#         self.lrn6 = nn.LocalResponseNorm(size=5, alpha=2e-4, beta=0.75, k=1.0)
#         self.drop = nn.Dropout(0.5)

#         # Global Average Pooling to ensure sequence length independence 
#         self.gap = nn.AdaptiveAvgPool1d(1)

#         # Fully Connected
#         if self.include_top:
#             self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(self.lrn2(F.relu(self.conv2(x))))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.lrn4(F.relu(self.conv4(x)))
#         x = F.relu(self.conv5(x))
#         x = self.drop(self.lrn6(F.relu(self.conv6(x))))

#         embedding = self.gap(x).squeeze(-1) # Output shape: (Batch, 256)

#         if self.include_top:
#             return self.fc(embedding)
#         return embedding


# class Hammad_CNN(nn.Module):
#     """
#     Architecture 5 from Hammad et al. (2020) - "ResNet-Attention model for human authentication using ECG signals"
#     """
#     def __init__(self, in_channels: int = 1, num_classes: int = 10, include_top: bool = True):
#         super(Hammad_CNN, self).__init__()
#         self.include_top = include_top
        
#         self.features = nn.Sequential(
#             # Layer 1 & 2
#             nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Dropout(0.2),
            
#             # Layer 5 & 6
#             nn.Conv1d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Dropout(0.2)
#         )
        
#         # GAP ensures dynamic length adaptation
#         self.gap = nn.AdaptiveAvgPool1d(1)
        
#         self.fc1 = nn.Linear(128, 512)
#         self.fc_drop = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(512, 448)
        
#         if self.include_top:
#             self.classifier = nn.Linear(448, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.gap(x).squeeze(-1) 
        
#         x = F.relu(self.fc1(x))
#         x = self.fc_drop(x)
#         emb = F.relu(self.fc2(x))
        
#         if self.include_top:
#             return self.classifier(emb)
#         return emb


# # =============================================================================
# # 2. ResNet & Attention Models
# # =============================================================================

# class ResNetBasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResNetBasicBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += identity
#         out = self.relu(out)
#         return out


# class ResNet1D(nn.Module):
#     """
#     1D ResNet-18 adapted for ECG.
#     """
#     def __init__(self, in_channels=1, num_classes=10, include_top=True):
#         super(ResNet1D, self).__init__()
#         self.include_top = include_top
#         self.inplanes = 64
        
#         self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(ResNetBasicBlock, 64, 2)
#         self.layer2 = self._make_layer(ResNetBasicBlock, 128, 2, stride=2)
#         self.layer3 = self._make_layer(ResNetBasicBlock, 256, 2, stride=2)
#         self.layer4 = self._make_layer(ResNetBasicBlock, 512, 2, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc_embed = nn.Linear(512 * ResNetBasicBlock.expansion, 128)

#         if self.include_top:
#             self.fc_class = nn.Linear(128, num_classes)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         embedding = self.relu(self.fc_embed(x)) 

#         if self.include_top:
#             return self.fc_class(embedding)
#         return embedding


# class ResNetAttentionBlock(nn.Module):
#     def __init__(self, channels, kernel_size):
#         super().__init__()
#         # Pre-activation Residual Module as defined by Hammad et al.
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.relu1 = nn.ReLU()
#         self.drop1 = nn.Dropout(0.2)
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        
#         self.bn2 = nn.BatchNorm1d(channels)
#         self.relu2 = nn.ReLU()
#         self.drop2 = nn.Dropout(0.2)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')

#     def forward(self, x):
#         residual = x
        
#         out = self.bn1(x)
#         out = self.relu1(out)
#         out = self.drop1(out)
#         out = self.conv1(out)
        
#         out = self.bn2(out)
#         out = self.relu2(out)
#         out = self.drop2(out)
#         out = self.conv2(out)
        
#         return out + residual

# class TemporalAttention(nn.Module):
#     def __init__(self, feature_dim):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim // 2),
#             nn.Tanh(),
#             nn.Linear(feature_dim // 2, 1)
#         )

#     def forward(self, x):
#         # x shape: (B, C, L) -> (B, L, C)
#         x = x.permute(0, 2, 1)
#         attn_weights = F.softmax(self.attention(x), dim=1) # (B, L, 1)
#         # Weighted sum over temporal dimension
#         out = torch.sum(x * attn_weights, dim=1) # (B, C)
#         return out

# class ResNet_Attention(nn.Module):
#     """
#     ResNet-Attention architecture proposed by Hammad et al. (2020).
#     Ref: "ResNet-Attention model for human authentication using ECG signals"
#     """
#     def __init__(self, in_channels: int = 1, num_classes: int = 10, include_top: bool = True, base_filters: int = 64):
#         super(ResNet_Attention, self).__init__()
#         self.include_top = include_top
        
#         # Initial Local Feature Learning Layer
#         self.init_conv = nn.Sequential(
#             nn.Conv1d(in_channels, base_filters, kernel_size=32, padding='same'),
#             nn.BatchNorm1d(base_filters),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Conv1d(base_filters, base_filters, kernel_size=32, padding='same')
#         )
        
#         # Substructures (MaxPool + Residual Module) with halving kernel sizes
#         kernels = [32, 16, 8, 4, 2]
#         self.substructures = nn.ModuleList()
#         for k in kernels:
#             self.substructures.append(nn.Sequential(
#                 nn.MaxPool1d(2),
#                 ResNetAttentionBlock(base_filters, kernel_size=k)
#             ))
            
#         self.final_bn = nn.BatchNorm1d(base_filters)
#         self.final_relu = nn.ReLU()
        
#         # Global Feature Learning
#         self.attention = TemporalAttention(base_filters)
        
#         # Authentication Part
#         self.fc1 = nn.Linear(base_filters, 128)
#         self.relu_fc = nn.ReLU()
        
#         if self.include_top:
#             self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.init_conv(x)
#         for block in self.substructures:
#             x = block(x)
#         x = self.final_relu(self.final_bn(x))
        
#         emb = self.attention(x)
#         emb = self.relu_fc(self.fc1(emb))
        
#         if self.include_top:
#             return self.fc2(emb)
#         return emb


# # =============================================================================
# # 3. RNN & Hybrid Models
# # =============================================================================

# class RNN_ECG(nn.Module):
#     """
#     Bi-directional LSTM with a small Convolutional front-end.
#     Raw ECG is too long for pure LSTM, so we downsample with Conv layers first.
#     """
#     def __init__(self, in_channels=1, num_classes=10, include_top=True, hidden_dim=64, num_layers=2):
#         super(RNN_ECG, self).__init__()
#         self.include_top = include_top
        
#         # Conv Front-End
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm1d(32), nn.ReLU(),
#             nn.MaxPool1d(2), 
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(64), nn.ReLU(),
#         )
        
#         # LSTM Layers
#         self.lstm = nn.LSTM(
#             input_size=64, hidden_size=hidden_dim, num_layers=num_layers, 
#             batch_first=True, bidirectional=True, dropout=0.2
#         )
        
#         self.fc_embed = nn.Linear(hidden_dim * 2, 128)
#         if self.include_top:
#             self.fc_class = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.conv(x) # (B, 64, L/4)
#         x = x.permute(0, 2, 1) # (B, L, C)
        
#         out, (h_n, c_n) = self.lstm(x)
        
#         # Concatenate the last forward and backward hidden states
#         h_forward = h_n[-2, :, :]
#         h_backward = h_n[-1, :, :]
#         x = torch.cat((h_forward, h_backward), dim=1) # (B, 2*hidden)
        
#         embedding = self.fc_embed(x) 
#         if self.include_top:
#             return self.fc_class(embedding)
#         return embedding


# class HybridCNNLSTM(nn.Module):
#     """
#     Classic Hybrid architecture: 1D CNN for feature extraction -> LSTM for temporal dynamics.
#     Common baseline in literature (e.g., BioECG).
#     """
#     def __init__(self, in_channels=1, num_classes=10, include_top=True):
#         super(HybridCNNLSTM, self).__init__()
#         self.include_top = include_top
        
#         self.cnn = nn.Sequential(
#             nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
#             nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
#             nn.Conv1d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
#             nn.Conv1d(128, 64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64), nn.ReLU()
#         )
        
#         self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(64*2, num_classes)

#     def forward(self, x):
#         features = self.cnn(x).permute(0, 2, 1)
#         out, _ = self.lstm(features)
#         embedding = torch.mean(out, dim=1) # Global Average Pooling
        
#         if self.include_top:
#             return self.fc(embedding)
#         return embedding


# # =============================================================================
# # 4. Transformers & Attention
# # =============================================================================

# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, seq_len: int, d_model: int):
#         super().__init__()
#         self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

#     def forward(self, x):
#         return x + self.pos_embedding[:, :x.size(1), :]

# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, d_model, n_heads, dim_ff=256, dropout=0.1):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model)
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         attn_out, _ = self.attn(x, x, x)
#         x = self.norm1(x + self.dropout(attn_out))
#         ff_out = self.ff(x)
#         x = self.norm2(x + self.dropout(ff_out))
#         return x

# class ECGTransformer(nn.Module):
#     """
#     Pure Transformer Encoder for ECG.
#     """
#     def __init__(self, in_channels=1, num_classes=10, include_top=True, seq_len=5000, embed_dim=128, num_heads=4, num_layers=2):
#         super(ECGTransformer, self).__init__()
#         self.include_top = include_top
        
#         self.conv_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=7, stride=1, padding=3)
#         self.pos_enc = LearnablePositionalEncoding(seq_len, embed_dim)
#         self.dropout = nn.Dropout(0.1)
        
#         self.transformer = nn.ModuleList([
#             TransformerEncoderBlock(embed_dim, num_heads, embed_dim*2, 0.1) for _ in range(num_layers)
#         ])
        
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         if self.include_top:
#             self.fc = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         # Embed: (B, C, L) -> (B, Embed, L) -> (B, L, Embed)
#         x = self.conv_embed(x).permute(0, 2, 1)
        
#         # Add Pos Encoding (truncate to current len)
#         x = self.pos_enc(x)
#         x = self.dropout(x)
        
#         for block in self.transformer:
#             x = block(x)
            
#         x = x.permute(0, 2, 1) # Back to (B, C, L) for pooling
#         emb = self.pool(x).squeeze(-1)
        
#         if self.include_top:
#             return self.fc(emb)
#         return emb


# # =============================================================================
# # 5. 2D Models (For Spectrograms/Scalograms)
# # =============================================================================

# class ResNet18_2D(nn.Module):
#     """
#     Standard ResNet18 for 2D Inputs.
#     """
#     def __init__(self, in_channels=1, num_classes=10, include_top=True):
#         super(ResNet18_2D, self).__init__()
#         self.include_top = include_top
        
#         resnet = models_2d.resnet18(weights=None)
        
#         # Modify first layer for 1 channel input
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             resnet.bn1, resnet.relu, resnet.maxpool,
#             resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         if self.include_top:
#             return self.fc(x)
#         return x


# # =============================================================================
# # 6. Specialized Wrappers (Siamese & DANN)
# # =============================================================================

# class SiameseNetwork(nn.Module):
#     """
#     Siamese Wrapper. Takes two inputs (x1, x2).
#     """
#     def __init__(self, backbone):
#         super(SiameseNetwork, self).__init__()
#         self.backbone = backbone
#         self.backbone.include_top = False # Ensure we get embeddings

#     def forward(self, x1, x2=None):
#         emb1 = self.backbone(x1)
#         if x2 is not None:
#             emb2 = self.backbone(x2)
#             return emb1, emb2
#         return emb1

# # --- Domain Adversarial Components ---

# class GradientReversalFn(Function):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#         return x.view_as(x)
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.neg() * ctx.alpha, None

# class GradientReversal(nn.Module):
#     def __init__(self, alpha=1.0):
#         super().__init__()
#         self.alpha = alpha
#     def forward(self, x):
#         return GradientReversalFn.apply(x, self.alpha)

# class DANN_ECG(nn.Module):
#     """
#     Domain Adversarial Neural Network for Cross-Session Adaptation.
#     """
#     def __init__(self, backbone, num_classes, num_domains=2, embed_dim=128):
#         super(DANN_ECG, self).__init__()
#         self.feature_extractor = backbone
#         self.feature_extractor.include_top = False
        
#         # 1. Identity Head
#         self.identity_classifier = nn.Linear(embed_dim, num_classes)
        
#         # 2. Domain Head (Adversary)
#         self.domain_classifier = nn.Sequential(
#             GradientReversal(alpha=1.0),
#             nn.Linear(embed_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_domains)
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         id_logits = self.identity_classifier(features)
#         domain_logits = self.domain_classifier(features)
#         return id_logits, domain_logits


# # =============================================================================
# # 7. Loss Functions
# # =============================================================================

# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.3):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, embeddings, labels):
#         dot_product = torch.matmul(embeddings, embeddings.t())
#         square_norm = torch.diag(dot_product)
#         distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
#         distances = torch.sqrt(torch.clamp(distances, min=1e-16))

#         pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        
#         hardest_positive_dist = torch.max(distances * pos_mask.float(), dim=1)[0]
        
#         max_dist = torch.max(distances)
#         hardest_negative_dist = torch.min(distances + max_dist * pos_mask.float(), dim=1)[0]

#         triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
#         return triplet_loss.mean()

# =============================================================================
# 7. Literature Baselines
# =============================================================================
# Re-implementations of published ECG biometric architectures, added so the
# benchmark can compare methods rather than only architectural families.
#
# Each is written from the architecture described in its source publication.
# These are re-implementations, not the authors' released code, and two
# adaptations are applied uniformly so every model can be evaluated under the
# identical protocol the framework enforces:
#
#   1. Global average pooling replaces any flatten-then-dense stage. Beat
#      length varies from 76 samples (NSRDB at 128 Hz) to 600 (PTB at 1 kHz),
#      and a fixed flatten would restrict each model to a single dataset.
#   2. The classifier head is separable via include_top, so one trained
#      network supplies both closed-set logits and verification embeddings.
#
# Reported numbers for these baselines therefore characterise the architecture
# under this benchmark's protocol. They are not claims about the originally
# published results, which used different preprocessing and different splits.


class _DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable 1D convolution.

    A per-channel spatial filter followed by a pointwise channel mix. Shared
    by the MobileNet and separable-residual baselines.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(_DepthwiseSeparableConv1d, self).__init__()

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.bn_depthwise = nn.BatchNorm1d(in_channels)

        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn_pointwise = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn_depthwise(self.depthwise(x)))
        x = F.relu(self.bn_pointwise(self.pointwise(x)))

        return x


class ECGXtractor(nn.Module):
    """
    Autoencoder-style convolutional feature extractor.

    Ref: P. Melzi, R. Tolosana, R. Vera-Rodriguez, "ECG biometric
    recognition: Review, system proposal, and benchmark evaluation",
    IEEE Access 11 (2023) 15555-15566.

    The published system pre-trains a convolutional autoencoder on unlabelled
    heartbeats and reuses its encoder as the biometric feature extractor. This
    implementation provides that encoder together with the classification head
    used for supervised training. The decode() method exposes the
    reconstruction branch so the autoencoder objective can be reproduced
    separately; only the encoder participates in biometric matching, which
    matches the role the published system assigns it.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 include_top: bool = True, embedding_dim: int = 128,
                 include_decoder: bool = False):
        super(ECGXtractor, self).__init__()

        self.include_top = include_top
        self.embedding_dim = embedding_dim
        self.include_decoder = include_decoder

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Built only when asked for. Supervised training never propagates
        # gradient through the decoder, so constructing it unconditionally
        # would add optimizer state that is never updated and would inflate
        # the parameter count reported in the computational profile.
        self.decoder = None

        if include_decoder:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(32, in_channels, kernel_size=4, stride=2,
                                   padding=1),
            )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Linear(128, embedding_dim)

        if self.include_top:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x):
        """
        Return the pooled encoder embedding used for biometric matching.
        """
        features = self.encoder(x)
        pooled = self.gap(features).squeeze(-1)

        return F.relu(self.embedding(pooled))

    def decode(self, x):
        """
        Reconstruct the input, for optional autoencoder pre-training.

        Requires the model to have been built with include_decoder=True.
        """
        if self.decoder is None:
            raise RuntimeError(
                "This ECGXtractor was built without a decoder. Construct it "
                "with include_decoder=True to use the reconstruction branch."
            )

        return self.decoder(self.encoder(x))

    def forward(self, x):
        embedding = self.encode(x)

        if self.include_top:
            return self.classifier(embedding)

        return embedding


class MobileNetGRU(nn.Module):
    """
    Lightweight depthwise-separable CNN followed by a recurrent stage.

    Ref: D. H. Rai, S. Kafley, "Lightweight MobileNetV1+GRU for ECG biometric
    authentication: Federated and adversarial evaluation", arXiv preprint
    arXiv:2509.20382 (2025).

    The convolutional trunk follows the MobileNetV1 pattern of one standard
    convolution followed by stacked depthwise separable blocks, and the GRU
    models the sequential structure that survives pooling. Channel widths are
    scaled by width_multiplier, as in MobileNetV1.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 include_top: bool = True, width_multiplier: float = 1.0,
                 gru_hidden: int = 64):
        super(MobileNetGRU, self).__init__()

        self.include_top = include_top

        def width(channels):
            return max(8, int(channels * width_multiplier))

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width(32), kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(width(32)),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            _DepthwiseSeparableConv1d(width(32), width(64)),
            _DepthwiseSeparableConv1d(width(64), width(128), stride=2),
            _DepthwiseSeparableConv1d(width(128), width(128)),
            _DepthwiseSeparableConv1d(width(128), width(256), stride=2),
        )

        self.gru = nn.GRU(
            input_size=width(256),
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.embedding_dim = gru_hidden * 2

        if self.include_top:
            self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))

        # (B, C, T) -> (B, T, C) for the recurrent stage.
        sequence = x.transpose(1, 2)
        outputs, _ = self.gru(sequence)

        # Mean over time keeps the embedding independent of beat length.
        embedding = outputs.mean(dim=1)

        if self.include_top:
            return self.classifier(embedding)

        return embedding


class MultiScaleCNN(nn.Module):
    """
    Parallel multi-scale 1D CNN.

    Ref: Y. Chu, H. Shen, K. Huang, "ECG authentication method based on
    parallel multi-scale one-dimensional residual network with center and
    margin loss", IEEE Access 7 (2019) 51598-51607.

    Parallel branches with different receptive fields capture the sharp QRS
    complex and the slower P and T waves simultaneously, rather than forcing a
    single kernel size to serve both. Branch outputs are concatenated before
    the shared trunk.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 include_top: bool = True, branch_channels: int = 32,
                 kernel_sizes=(3, 7, 15)):
        super(MultiScaleCNN, self).__init__()

        self.include_top = include_top

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(branch_channels, branch_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for kernel_size in kernel_sizes
            ]
        )

        fused_channels = branch_channels * len(kernel_sizes)

        self.trunk = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(fused_channels, 128, kernel_size=5, padding=2,
                      bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Linear(256, 128)
        self.embedding_dim = 128

        if self.include_top:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        fused = torch.cat(
            [branch(x) for branch in self.branches],
            dim=1,
        )

        features = self.trunk(fused)
        pooled = self.gap(features).squeeze(-1)
        embedding = F.relu(self.embedding(pooled))

        if self.include_top:
            return self.classifier(embedding)

        return embedding


class _SeparableResidualBlock(nn.Module):
    """
    Residual block built from depthwise separable convolutions.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(_SeparableResidualBlock, self).__init__()

        self.conv1 = _DepthwiseSeparableConv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
        )
        self.conv2 = _DepthwiseSeparableConv1d(
            out_channels,
            out_channels,
            kernel_size=3,
        )

        self.shortcut = None

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.conv2(self.conv1(x))

        return F.relu(out + identity)


class SeparableResNet(nn.Module):
    """
    Residual depthwise-separable CNN for low-latency authentication.

    Ref: N. Ihsanto, K. Ramli, D. Sudiana, T. S. Gunawan, "Fast and accurate
    algorithm for ECG authentication using residual depthwise separable
    convolutional neural networks", Applied Sciences 10 (9) (2020) 3304.

    Depthwise separable convolutions cut the parameter count by roughly an
    order of magnitude relative to a standard residual network of the same
    depth, which is what makes the architecture attractive for on-device
    authentication.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 include_top: bool = True, base_channels: int = 32):
        super(SeparableResNet, self).__init__()

        self.include_top = include_top

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.stage1 = _SeparableResidualBlock(
            base_channels,
            base_channels * 2,
            stride=2,
        )
        self.stage2 = _SeparableResidualBlock(
            base_channels * 2,
            base_channels * 4,
            stride=2,
        )
        self.stage3 = _SeparableResidualBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = base_channels * 8

        if self.include_top:
            self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage3(self.stage2(self.stage1(x)))

        embedding = self.gap(x).squeeze(-1)

        if self.include_top:
            return self.classifier(embedding)

        return embedding
