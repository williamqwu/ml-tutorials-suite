# =============================================================================
# ViTForClassfication: A ViT model for image classification
# -----------------------------------------------------------------------------
# Summary: Implements a scalable ViT model used for image classification
#          on realworld data.
# Author: Tin Nguyen (https://github.com/tintn/vision-transformer-from-scratch)
# Modified by: Q.WU
# =============================================================================

import math
import torch
from torch import nn

#######################################################################
# Sample config
#######################################################################

"""
Sample config:
    config = {
        "patch_size": 4,  
        "hidden_size": 48,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 4 * 48, # 4 * hidden_size
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "image_size": 32,
        "num_classes": 10, # num_classes of CIFAR10
        "num_channels": 3,
        "qkv_bias": True,
        "use_faster_attention": True,
    }
    
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    assert config['intermediate_size'] == 4 * config['hidden_size']
    assert config['image_size'] % config['patch_size'] == 0

Explanation: 
- `patch_size`: [Embedding] width/height of each square patch.
    - Example value: 4
    - Input image size: 32x32 -> 8x8 patches
- `hidden_size`: [Embedding] size of the vector after image patch being converted into embeddings.
    - Example value: 48
- `num_hidden_layers`: [Attention] number of Transformer encoder blocks.
    - Default value: 4
    - Higher values increase model depth/capacity, while adding compute cost.
- `num_attention_heads`: [Attention] control the attention head size.
    - Default value: 4
    - NOTE: num_attention_heads <= hidden_size; see assertion below.
    - Higher value: more perspectives/specialization for each head; each head capture less and increase memory usage; for simple feature, more heads are redundant.
- `intermediate_size`: [Attention] intermediate layer size for FFN inside each transformer block.
    - Default value: 4*48
    - NOTE: typically set as 4*hidden_size. Can reduced to 2x for efficiency / increased to 6x/8x for large models.
    - Higher value: more neurons -> handle more complex transformations, improve accuracy; slow down training/inference.
- `hidden_dropout_prob`: [Global] dropout probability for all three components.
    - Default value: 0.0
    - Higher value: useful to reduce overfitting in small datasets / overparameterized models; could slower convergence and potential underfitting.
- `attention_probs_dropout_prob`: [Attention] dropout applied to attention weights
    - Default value: 0.0
    - Higher value: prevents attention from being overly "confident", useful in small model; could leads to underfitting
- `initializer_range`: [Initialization] a stdev value controlling the random weight initialization scale for linear and convolutional layers (and embeddings).
    - Default value: 0.02
    - Higher value: promotes stronger gradient signals early in training, "jump-start"; risk of exploding gradients.
- `image_size`: [Embedding] width (or height) of the input image
    - Example value: 32
    - NOTE: it assumes that the image is square, e.g. 32x32. Must match dataset.
- `num_classes`: [Classifier] specifies number of output categories for classification.
    - Example value: 10
    - NOTE: must match dataset.
- `num_channels`: [Embedding] number of image color channels
    - Example value: 3
    - NOTE: must match dataset.
- `qkv_bias`: [Attention] a boolean flag that determines whether to include a bias term in the linear layers used to compute the QKV.
    - Default value: True
    - by setting True: more flexible representations, slightly better empirical performance; small increase (~0.1%) of more parameters.
- `use_faster_attention`: [Attention] controls whether to use optimizer, fused multi-head attention, or standard per-head version.
    - Default value: True
    - by setting True: faster training/inference, lower memory overhead
"""

#######################################################################
# General blocks
#######################################################################


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


#######################################################################
# ViT Part.1: Transform Images into Embeddings
#######################################################################


class PatchEmbeddings(nn.Module):
    """Part 1.1 of ViT
    TL;DR:
        - A layer that converts the input image into patches and then project them into a vector space.
    Inheritance/Composition:
        - A component (member) of `Embeddings`.
    Expected Input: a batch of images.
        - Shape: (batch_size, num_channels, image_size (H), image_size (W))
        - Example shape: (64, 3, 32, 32)
        - Constraint: `image_size` must be divisible by `patch_size`.
    Expected Output: a batch of image patch sequence.
        - Shape: (batch_size, num_patches, hidden_size)
        - Example shape: (64, 64, 48) given patch_size = 4, hidden_size = 48.
        - In the example above, each patch represented as a 48-dimensional vector.
    Behavior during training:
        - Since it has a `nn.Conv2d` layer, this projection is learnable (containing weights and biases).
            - nn.Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P)
            - Weights: A 4D tensor of shape (D, C, P, P)
            - Biases: A 1D tensor of shape (D,)
        - Intuitively, this layer learns how best to represent each patch as an embedding, not just a raw flattened patch.
    Behavior during evaluation:
        - The behavior is deterministic (since this module doesn't use Dropout or BatchNorm).
    Behavior during initialization: TODO
    """

    def __init__(self, config):
        # Call the base class constructor `nn.Module` to properly initialize the module.
        super().__init__()
        # Extract relevant hyperparameters from the input config.
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the total number of patches.
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a learnable projection applied to every patch.
        #   1. Divides the image into non-overlapping patches (that's why we config stride = kernel_size = `patch_size`).
        #   2. Projects each patch (e.g., 4×4×3) into a vector of size hidden_size.
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, hidden_size, H_patch, W_patch)
        x = self.projection(x)
        # `.flatten(2)` flattens the last two spatial dimensions: (B, hidden_size, H', W') -> (B, hidden_size, num_patches)
        # `.transpose(1, 2)` swaps the content between dim1 and dim2
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """Part 1.2 of ViT
    TL;DR:
        - A full component handling embedding-related functionality, which combines the patch embeddings with the class token and position embeddings.
    Inheritance/Composition:
        - A component (member) of `ViTForClassfication`.
    Expected Input: a batch of images.
        - Shape: (batch_size, num_channels, image_size (H), image_size (W))
        - Example shape: (64, 3, 32, 32)
        - Constraint: `image_size` must be divisible by `patch_size`.
    Expected Output: a batch of embedding sequences for transformer input.
        - Shape: (batch_size, num_patches+1, hidden_size)
        - `+1` accounts for the prepended [CLS] token.
    Behavior during training:
        - If `hidden_dropout_prob` >0 from config, dropout is active
        - Parameters (patch_embeddings/cls_token/position_embeddings) will be updated via backpropagation.
    Behavior during evaluation:
        - Output is deterministic (dropout is disabled if setting `model.eval()`).
    Behavior during initialization: TODO
    """

    def __init__(self, config):
        # Call the base class constructor `nn.Module` to properly initialize the module.
        super().__init__()
        # Save the config (for what?).
        self.config = config
        # Initialize a `PatchEmbeddings` instance.
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] (~classification) token.
        #   Similar to BERT, the [CLS] token is added to the beginning of the
        #   input sequence and is used to classify the entire sequence.
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create position embeddings for the [CLS] token and the patch embeddings.
        #   Add 1 to the sequence length for the [CLS] token.
        #   Shape: (1, num_patches + 1, hidden_size)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"])
        )
        # Apply dropout in final embeddings to alleviate overfitting.
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


#######################################################################
# ViT Part.2: Encoder w/ Attention + FFN
#######################################################################


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias,
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat(
            [attention_output for attention_output, _ in attention_outputs], dim=-1
        )
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack(
                [attention_probs for _, attention_probs in attention_outputs], dim=1
            )
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(
            self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias
        )
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = self.attention(
            self.layernorm_1(x), output_attentions=output_attentions
        )
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


#######################################################################
# ViT: Whole structure at high-level
#######################################################################


class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(
            embedding_output, output_attentions=output_attentions
        )
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config["initializer_range"]
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
