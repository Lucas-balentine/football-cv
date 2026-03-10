"""
Team classification via CNN embeddings + clustering.

For each player crop, extract a feature embedding from a pretrained
ResNet-18 (ImageNet weights). Then cluster all embeddings into 2 teams
using K-Means. The CNN captures texture, pattern, color, and shape —
much richer than raw color alone.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.cluster import KMeans


# Preprocessing to match ImageNet expectations
_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Lazy-loaded model
_model = None


def _get_model() -> torch.nn.Module:
    """Load ResNet-18 with ImageNet weights, remove the classification head."""
    global _model
    if _model is None:
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        # Remove the final FC layer — we want the 512-d embedding
        _model = torch.nn.Sequential(*list(resnet.children())[:-1])
        _model.eval()
    return _model


def extract_embeddings(crops: list[np.ndarray]) -> np.ndarray:
    """Extract 512-d feature embeddings for a list of BGR crops.

    Args:
        crops: list of BGR numpy arrays (any size).

    Returns:
        (N, 512) float32 array of L2-normalized embeddings.
    """
    model = _get_model()

    # Batch all crops together
    tensors = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(_preprocess(rgb))

    batch = torch.stack(tensors)

    with torch.no_grad():
        features = model(batch)  # (N, 512, 1, 1)

    embeddings = features.squeeze(-1).squeeze(-1).numpy()  # (N, 512)

    # L2 normalize so K-Means uses cosine-like distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings


def classify_teams(
    crops: list[np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """Classify a list of player crops into two teams using CNN embeddings.

    Returns:
        team_labels: list of team assignments (0 or 1) per player.
        embeddings: (N, 512) array of feature embeddings.
    """
    if len(crops) < 2:
        return [-1] * len(crops), np.zeros((len(crops), 512))

    embeddings = extract_embeddings(crops)

    # K-Means into 2 teams
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    kmeans.fit(embeddings)

    team_labels = [int(l) for l in kmeans.labels_]

    return team_labels, embeddings
