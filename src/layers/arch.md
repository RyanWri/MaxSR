# Architecture of MaxSR

<p>The MaxSR network is designed to perform super-resolution, enhancing the resolution of low-resolution images. Let's break down the architecture, transformations, and the flow of tensors through the network blocks.</p>

<ol>
<li> Input
Dimensions: A low-resolution image of size 𝐻 × 𝑊 with 3 color channels (RGB), 
<br/> resulting in a tensor of shape [batch_size, in_channels, H, W].
</li>
<li> Shallow Feature Extraction Block (SFEB)
<br/> Input: Tensor of shape [batch_size, in_channels, H, W].
<br/> Transformation:
Convolution with out_channels filters, kernel size 3, padding 1.
ReLU activation.
<br/> Output: Feature tensor of shape [batch_size, out_channels, H, W].
</li>
<li> Multiple Cascaded Adaptive MaxViT Blocks
<br/> Number of Blocks: As specified in the configuration.
<br/> Each Block: Adaptive Grid Self-Attention:
<br/> Input: Feature tensor of shape [batch_size, hidden_dim, H, W].
<br/> Transformation:
<br/> Query, Key, Value convolutions with reshaping and attention mechanism.
<br/> Softmax and matrix multiplication for attention.
Output reshaped back to [batch_size, hidden_dim, H, W].
<br/> Output: Tensor of shape [batch_size, hidden_dim, H, W].
<br/> Feed-Forward Network (FFN):
<br/> Input: Flattened tensor [batch_size, hidden_dim, H * W].
<br/> Transformation:
Linear layers with ReLU and Dropout.
Layer normalization.
Output: Tensor reshaped back to [batch_size, hidden_dim, H, W].
<br/> Overall Output: Tensor of shape [batch_size, hidden_dim, H, W].
</li>
<li> Hierarchical Feature Fusion Block (HFFB)
<br/> Input: Tensor of shape [batch_size, hidden_dim, H, W].
<br/> Transformation:
Two convolutional layers with ReLU.
<br/> Output: Tensor of shape [batch_size, hidden_dim, H, W].
</li>
<li> Reconstruction Block (RB)
<br/> Input: Tensor of shape [batch_size, hidden_dim, H, W].
<br/> Transformation:
<br/> Convolutional layer to produce out_channels * (upscale_factor^2) filters.
<br/> Pixel shuffle to upscale the image.
<br/> ReLU activation.
<br/> Output: High-resolution image tensor of shape [batch_size, out_channels, H * upscale_factor, W * upscale_factor].
</li>
</ol>