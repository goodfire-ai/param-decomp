# Convolutional Layer Decomposition in VPD

## Summary

This document explains how we extend adVersarial Parameter Decomposition (VPD) to convolutional layers. The convolution decomposition works by:

- Viewing the convolution weight as a linear transformation on flattened patches. This linear transformation is given by a weight matrix of (`out_channels`, `in_channels * kH * kW`).
- Factorizing this linear transformation into $V$ (patch features to components) and $U$ (components to outputs).
- Reshaping $U$ and $V$ back into `conv2d` operations for efficient implementation.

This allows us to analyze convolutional networks using the same VPD framework we use for fully connected networks.

## Convolutions as Linear Operations

A 2D convolution can be understood as a linear operation applied at each spatial location. Consider a convolutional layer with:

- Input: (`batch`, `in_channels`, `H`, `W`)
- Kernel size: (`kH`, `kW`)
- Output channels: `out_channels`

At each spatial position, the convolution extracts a patch of size (`in_channels`, `kH`, `kW`), flattens it to a vector of size (`in_channels * kH * kW`), and applies a linear transformation to produce `out_channels` outputs.

So conceptually, the conv weight of shape (`out_channels`, `in_channels`, `kH`, `kW`) can be viewed as a matrix of shape (`out_channels`, `in_channels * kH * kW`) that operates on flattened patches.

## The Decomposition

We decompose this "flattened" view of the convolution weight:

$$W_{\text{flat}} = V\,U.$$

> **Note for readers (transcriber's caveat):** Taken literally, this equation is dimensionally inconsistent with the shapes given below: `W_flat` is (`out_channels`, `in_channels * kH * kW`), whereas `V @ U` with the stated shapes of $V$ and $U$ is its transpose, (`in_channels * kH * kW`, `out_channels`). The reconstruction step later does apply a `.T` (`(V @ U).T.reshape(...)`), which is consistent with the shapes. So the intended convention is *most likely* $W_{\text{flat}} = (V\,U)^\top$, with the line above being informal shorthand — but I'm flagging this as a probable source inconsistency rather than something I can confirm from the document alone.

Where:

- $V$ has shape (`in_channels * kH * kW`, `C`) and maps input patches to $C$ component activations.
- $U$ has shape (`C`, `out_channels`) and maps component activations to output channels.

Interpretation:

- $V$ learns to detect features in input patches and produce a component activation for each.
- $U$ learns how to combine these component activations into the final output channels.

## Efficient Implementation

Rather than explicitly extracting patches and doing matrix multiplication (which would be slow), we implement this using native convolution operations.

### Step 1: Compute component activations via convolution

We reshape $V$ from (`in_channels * kH * kW`, `C`) into $C$ convolutional filters of shape (`C`, `in_channels`, `kH`, `kW`). Applying these filters to the input gives component activations at each spatial location:

```python
component_acts = conv2d(input, V_as_filters)
# Result: (batch, C, H_out, W_out)
```

### Step 2: Apply $U$ via $1 \times 1$ convolution

We reshape $U$ from (`C`, `out_channels`) into $1 \times 1$ convolutional filters of shape (`out_channels`, `C`, `1`, `1`). Applying this to the component activations:

```python
output = conv2d(component_acts, U_as_1x1)
# Result: (batch, out_channels, H_out, W_out)
```

A $1 \times 1$ convolution is equivalent to applying a linear transformation independently at each spatial location, exactly what we need to map the $C$ component activations to `out_channels` outputs.

## Reconstructing the Original Weight

To verify the decomposition or to use the decomposed model without masking, we can reconstruct the original convolution weight:

```python
W_reconstructed = (V @ U).T.reshape(out_channels, in_channels, kH, kW)
```

