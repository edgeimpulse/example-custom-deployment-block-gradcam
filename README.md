# Custom Deployment Block Gradcam

This custom deployment block example exports your test datasets and apply a gradcam overlay based on your trained model.

![Example of a “car vs not car” model, highlighting the “features” of a car](/assets/grad-cam-example.png)

*Note: This currently works for image classification and visual regression models where at least one 2D convolution layer is present in your architecture.*

## Grad-CAM details

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that helps interpret the predictions of a convolutional neural network (CNN) by visualizing the regions of the input image that most influenced the model's decision. This is achieved by:

1. Computing gradients of the target output with respect to the feature maps of the last convolutional layer.
2. Weighting the feature maps by the importance of these gradients.
3. Generating a heatmap that highlights the areas of interest.

This script extends Grad-CAM for:
- **Classification Models**: Highlights areas contributing to the predicted class.
- **Visual Regression Models**: Highlights areas contributing to the numerical regression output.

If you want more information on the Grad-CAM technique, we invite you to read this paper [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391).

## What does this custom deployment block do?

1. **Dataset export**:
   - Exports the test dataset from an Edge Impulse project.
   - Automatically retries until the dataset is ready for download.

2. **Model download**:
   - Retrieves the trained model (`.h5` format) for your Edge Impulse project.
   - Ensures compatibility with models containing 2D convolutional layers.

3. **Grad-CAM visualization**:
   - Applies the Grad-CAM technique to visualize the regions of the input image that contributed most to the model's predictions.
   - Works for:
     - **Classification models**: Highlights regions associated with the predicted class.
     - **Regression models**: Highlights regions contributing to the predicted regression value.

4. **Output generation**:
   - Saves Grad-CAM heatmaps overlaid on the original images.
   - Separates correctly predicted and incorrectly predicted samples into separate directories for easy analysis.

## Google Colab option

To test the functionality without setting up locally, use this [Google Colab](https://colab.research.google.com/drive/1UE8LUE6X8M1COk98Jj7n3XS5YjwGUOnE?usp=sharing). It comes pre-configured to run in a browser with no local setup required.

## Custom deployment block

Clone this repository:

```bash
git clone git@github.com:edgeimpulse/example-custom-deployment-block-gradcam.git
```

Initialize the block:

```bash
edge-impulse-blocks init
Edge Impulse Blocks v1.30.3
? In which organization do you want to create this block? Developer Relations
Attaching block to organization 'Developer Relations'
? Choose an option Create a new block
? Do you have an integration URL (shown after deployment, e.g. your docs page), leave empty to skip https://github
.com/edgeimpulse/example-custom-deployment-block-gradcam

Your new block has been created in '/Users/luisomoreau/workspace/ei/custom-deployment-gradcam'.
When you have finished building your block, run 'edge-impulse-blocks push' to update the block in Edge Impulse.
```

Push the deployment block to your organization:

```bash
edge-impulse-block push
```

## Testing locally with Docker

**Build**:

```
docker build -t gradcam-deployment-block .
```

**Run**:

```
docker run --rm -it gradcam-deployment-block --api-key ei_123456789abcdef
```

## Fine-tuning the visualization

You can adjust **three parameters** to fine-tune the visualization: `alpha`, `pooling-gradients`, and `heatmap-normalization`.

### 1. **Alpha (`--alpha`)**

The `alpha` parameter controls the **transparency** of the Grad-CAM overlay when it is superimposed on the original image.

**Default Value**: `0.4`

**Range**:

- A value between `0` and `1`.
  - `0`: Fully transparent (only the original image is visible).
  - `1`: Fully opaque (only the Grad-CAM overlay is visible).

**How to Choose**:

- **Recommended Default (`0.4`)**:
  Provides a balance between showing the original image and highlighting the Grad-CAM heatmap.
- **Higher Alpha (`> 0.5`)**:
  Use this if you want the Grad-CAM heatmap to dominate the visualization.
- **Lower Alpha (`< 0.4`)**:
  Use this if you want the original image to be more prominent.

### 2. **Pooling Gradients (`--pooling-gradients`)**

The `pooling-gradients` parameter determines how gradients (importance signals) are combined across the spatial dimensions of the last convolutional layer to generate the heatmap.

**Options**:

- `mean` (default):
  - Averages the gradients across the spatial dimensions.
  - Smoothens the heatmap, providing a general overview of important regions.
- `sum_abs`:
  - Takes the sum of the absolute values of the gradients.
  - Highlights areas with strong activations, which can sometimes create sharper heatmaps.

**How to Choose**:

- **Classification Models**:
  - Use `mean` for a smoother and more generalized heatmap.
  - Use `sum_abs` if you want to emphasize the most critical regions (e.g., sharp object boundaries).
- **Regression Models**:
  - `sum_abs` is often more useful for regression tasks, as it highlights features contributing to extreme values.
- **Experiment**:
  Try both options to see which one provides more meaningful visualizations for your model and dataset.

### 3. **Heatmap Normalization (`--heatmap-normalization`)**

The `heatmap-normalization` parameter determines how the heatmap values are scaled for visualization.

**Options**:

- `percentile` (default):
  - Ensures the heatmap values are scaled between `0` and `1` based on their maximum value.
  - Best for emphasizing high-activation areas while normalizing the output globally.
- `simple`:
  - Normalizes the heatmap by dividing by the global maximum.
  - A simpler approach that may work well for datasets where most activations are relevant.

**How to Choose**:

- **Default (`percentile`)**:
  - Works well in most cases, especially if you expect only certain areas of the image to be significant.
- **Use `simple`**:
  - Suitable for models where the activations are uniformly spread, and all areas are equally important.

## Limitations

1. This script assumes the presence of at least one 2D convolutional layer in the model architecture.
2. It is designed for image classification and visual regression tasks.
3. For regression models, the script uses a threshold to determine correctness; adjust this threshold (`threshold = 0.1`) as needed for your use case.