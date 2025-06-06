import torch
from torch import nn
import torch.nn.functional as F

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x

def get_attn_mask(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)
    cnt = 0
    '''
    这部分代码只在初始化时执行一次, 用来生成静态的 attn mask, 不会影响模型的每轮前向传播速度。
    '''
    # slice(start, stop, step) 等价于 start:stop:step 的切片操作
    # 给区域打上标签cnt
    for h in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
        for w in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)  # [num_windows, N]
    # 区域标签不同的位置才 mask。
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x B_ x num_heads x N x head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # num_windows = mask.shape[0]
            # attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            print(f"attn: {attn.shape}, mask: {mask.shape}")
            attn = attn + mask.unsqueeze(1)#.unsqueeze(0)
            # attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        if shift_size > 0:
            H, W = input_resolution
            # mask来处理 Shifted Window 的跨界问题，防止信息泄露, 主要出现在图像边缘
            attn_mask = get_attn_mask(H, W, window_size, shift_size, device="cpu")  # device 后续自动转移
            '''
            register_buffer(name, tensor) 用来注册一个 不会训练的张量，但会随着模型迁移到 GPU/CPU, 且能被保存和加载。
            '''
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            '''
            这儿完成cyclic shift, 论文里配图是按window_size=4来的, 实际window_size=7用得更多。
            '''
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        '''
        只在窗口内进行self-attention, 减少复杂度。
        '''
        attn_windows = self.attn(self.norm1(x_windows), mask=self.attn_mask)
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, L, C)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        '''
        不再使用显示的pos_embedding, 以致输入分辨率被限制。
        而是用相对偏移自动建模 patch 间位置关系。
        '''
        x = self.proj(x)  # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x

class SwinStage(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                )
            )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, img_size=56, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2], num_heads=[3, 6], window_size=7, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_chans, embed_dim)
        H = W = img_size // patch_size
        self.stage1 = SwinStage(embed_dim, (H, W), depths[0], num_heads[0], window_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)

# 测试 Swin Transformer
if __name__ == '__main__':
    '''
    Swin Transformer 是 ViT 的本地化、分层版，通过 Window Attention + Shift + Downsample 设计，大幅降低了计算量，使 Transformer 真正适配高分辨率视觉任务。
    '''
    model = SwinTransformer()
    # Swin Transformer 把 ViT 的“全局注意力”换成了“局部滑窗注意力”，还引入了层级结构，让 Transformer 更像 CNN，能高效处理高分辨率图像。
    dummy = torch.randn(1, 3, 56, 56)
    out = model(dummy)
    print(out.shape)  # torch.Size([1, 1000])
