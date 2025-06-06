import torch
from torch import nn
from math import sqrt

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        # 使用 Conv2d(kernel=patch_size, stride=patch_size) 是 ViT 中经典的“卷积替代 patch MLP”的技巧，本质是工程层面简化
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B = x.size(0)
        '''
        注: 原版ViT是'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 然后所有块的像素经过mlp作为输入的embed的。
        这儿通过Conv2d 占位式实现 MLP over patch, 是 ViT 在 PyTorch 里的“工程加速实现”。实际是等价的。
        '''
        x = self.proj(x)  # shape: [B, embed_dim, H', W']
        '''
        Tensor.flatten(start_dim=0, end_dim=-1)
        '''
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # prepend cls token
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, mask=None):
        query, key, value = self.q(query), self.k(key), self.v(value)
        scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float("inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        x = torch.cat([h(query, key, value, mask) for h in self.heads], dim=-1)
        return self.output_linear(x)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        print(x.shape)
        for layer in self.encoder_layers:
            x = layer(x)
        cls_token = x[:, 0]  # Extract class token
        return self.head(self.norm(cls_token))

# 示例：用随机图像测试 ViT
if __name__ == '__main__':
    model = ViT(img_size=224, patch_size=16, in_channels=3,
                embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, num_classes=1000)
    dummy_input = torch.randn(1, 3, 224, 224)
    # ViT 对 NLP Transformer 的唯一核心改造点就是输入的“表示方式”，也就是 Patch Embedding。 我们只是把“词”换成了“图像块”，Transformer 本身并没有大改。这说明 Transformer 是一种非常强大、通用的架构。
    output = model(dummy_input)
    print(output.shape)  # torch.Size([1, 1000])
