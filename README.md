# Custom Deployment Block Gradcam

This custom deployment block example exports your test datasets and apply a gradcam overlay based on your trained model.

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

## Setup

```
edge-impulse-blocks init
Edge Impulse Blocks v1.29.5
? In which organization do you want to create this block? Developer Relations
Attaching block to organization 'Developer Relations'
? Choose an option Create a new block
? Do you have an integration URL (shown after deployment, e.g. your docs page), leave empty to skip https://github
.com/edgeimpulse/example-custom-deployment-block-gradcam

Your new block has been created in '/Users/luisomoreau/workspace/ei/custom-deployment-gradcam'.
When you have finished building your block, run 'edge-impulse-blocks push' to update the block in Edge Impulse.
```

## Limitations

1. This script assumes the presence of at least one 2D convolutional layer in the model architecture.
2. It is designed for image classification and visual regression tasks.
3. For regression models, the script uses a threshold to determine correctness; adjust this threshold (`threshold = 0.1`) as needed for your use case.