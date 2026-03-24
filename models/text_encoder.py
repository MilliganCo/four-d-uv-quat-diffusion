# models/text_encoder.py - Простой текстовый энкодер для условной генерации
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

# CIFAR-10 классы
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR10_CLASS_DESCRIPTIONS = {
    'airplane': 'an airplane flying in the sky',
    'automobile': 'a car or automobile on the road',
    'bird': 'a bird perched or flying',
    'cat': 'a cat sitting or standing',
    'deer': 'a deer in nature',
    'dog': 'a dog sitting or standing',
    'frog': 'a frog on a surface',
    'horse': 'a horse standing or running',
    'ship': 'a ship on water',
    'truck': 'a truck on the road'
}


class ClassLabelEncoder(nn.Module):
    """
    Простой энкодер для меток классов CIFAR-10
    Использует learnable embeddings для каждого класса
    """
    def __init__(self, num_classes=10, embed_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        labels: [B] - метки классов (0-9)
        returns: [B, embed_dim] - текстовые эмбеддинги
        """
        emb = self.embedding(labels)  # [B, embed_dim]
        return self.mlp(emb)


class SimpleTextEncoder(nn.Module):
    """
    Простой текстовый энкодер для текстовых промптов
    Использует bag-of-words подход с learnable embeddings
    """
    def __init__(self, vocab_size=1000, embed_dim=128, max_length=77):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        text_ids: [B, seq_len] - токенизированные тексты
        returns: [B, embed_dim] - текстовые эмбеддинги
        """
        B, seq_len = text_ids.shape
        positions = torch.arange(seq_len, device=text_ids.device).unsqueeze(0).expand(B, -1)
        
        x = self.embedding(text_ids) + self.pos_embedding(positions)
        x = self.transformer(x)
        # Усредняем по последовательности
        x = x.mean(dim=1)  # [B, embed_dim]
        return self.proj(x)


def get_text_encoder(encoder_type: str = "class_label", embed_dim: int = 128):
    """
    Фабрика для создания текстового энкодера
    
    Args:
        encoder_type: "class_label" для меток классов, "simple" для текстов
        embed_dim: размерность эмбеддингов
    """
    if encoder_type == "class_label":
        return ClassLabelEncoder(num_classes=10, embed_dim=embed_dim)
    elif encoder_type == "simple":
        return SimpleTextEncoder(vocab_size=1000, embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

