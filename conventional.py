# This is impl of ViT Model with basic building blocks hacked together from timm's implementation
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


# Patch Embedding
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        img_size = (img_size, img_size)  # Ensure tuple format
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        # Calculate number of patches
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Convolutional layer for patch embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                

    def forward(self, x):
        x=self.proj(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x

# MLP (Feed-forward network)
class Mlp(nn.Module):
    """Multilayer Perceptron"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# DropPath (Stochastic Depth)
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        # Implement stochastic depth
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with any dimensionality
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output

# Attention Module
class Attention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'Dimension must be divisible by num_heads.'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor

        # Layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
 
    def forward(self, x):
        #x = x.transpose(1,2) # Swap before attention
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)  # Each has shape [B, num_heads, N, head_dim]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        #print( f'shapeeeeee after attention:{x.shape}')
        return x


# Transformer Block
class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        # Layers
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                        act_layer=act_layer, drop=drop)

    def forward(self, x):
        
       #_______________________________________ Original
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Feed-forward network with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# Vision Transformer
class VisionTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads= num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize parameters with trunc_normal_
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_vit_weights)

    @staticmethod
    def _init_vit_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        #print(f"Input shape: {x.shape}")
        B = x.shape[0]

        # Embed patches
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        #print(f"After patch embedding: {x.shape}")

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        #print(f"After adding class token: {x.shape}")

        x = x + self.pos_embed
        #print(f"After adding positional embedding: {x.shape}")
        x = self.pos_drop(x)

        # Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            #print(f"After block {i + 1}: {x.shape}")

        # Final normalization
        x = self.norm(x)
        #print(f"After normalization: {x.shape}")
        #print(f"classification token: {x[:, 0].shape}")
        return x[:, 0]  # Return the class token

    def forward(self, x):
        x = self.forward_features(x)
        #print(f"classification token shape: {x.shape}")
        x = self.head(x)
        return x
    
    def get_last_selfattention(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks[:-1]:
            x = blk(x)

        # Last block attention extraction
        last_blk = self.blocks[-1]
        x_norm = last_blk.norm1(x)
        B, N, C = x_norm.shape
        qkv = last_blk.attn.qkv(x_norm).reshape(B, N, 3, last_blk.attn.num_heads, last_blk.attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        attn_scores = (q @ k.transpose(-2, -1)) * last_blk.attn.scale
        attn_probs = attn_scores.softmax(dim=-1)

        return attn_probs



# Example usage
if __name__ == "__main__":
    # Create a ViT model with a single head in the first block
    model = VisionTransformer(
        img_size=224, patch_size=16, in_chans=3, num_classes=10,
        embed_dim=196, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, norm_layer=nn.LayerNorm
    )
    dummy_input = torch.randn(1, 3, 224, 224)
    print(model)
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}")  # Output shape: [1, 10]
