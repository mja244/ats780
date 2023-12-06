# Apply integrated gradients XAI
# technique to precip CNN model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

model = tf.keras.models.load_model('../HW2/small_cnn_model.keras')

Xtrain = np.load('../HW2/data/Xtrain_pr.npy')
Xval = np.load('../HW2/data/Xval_pr.npy')
#Xtest = np.load('data/Xtest_pr.npy')

Ytrain = np.load('../HW2/data/Ytrain_pr.npy')
Yval = np.load('../HW2/data/Yval_pr.npy')
#Ytest = np.load('data/Ytest_pr.npy')

def get_gradients(inputs, top_pred_idx=None):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        inputs: 2D/3D/4D matrix of samples
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    inputs = tf.cast(inputs, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        
        # Run the forward pass of the layer and record operations
        # on GradientTape.
        preds = model(inputs, training=False)  
        
        # For classification, grab the top class
        if top_pred_idx is not None:
            preds = preds[:, top_pred_idx]
        
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.        
    grads = tape.gradient(preds, inputs)
    return grads


def get_integrated_gradients(inputs, baseline=None, num_steps=50, top_pred_idx=None):
    """Computes Integrated Gradients for a prediction.

    Args:
        inputs (ndarray): 2D/3D/4D matrix of samples
        baseline (ndarray): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.            

    Returns:
        Integrated gradients w.r.t input image
    """
    # If baseline is not provided, start with zeros
    # having same size as the input image.
    if baseline is None:
        input_size = np.shape(inputs)[1:]
        baseline = np.zeros(input_size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # 1. Do interpolation.
    inputs = inputs.astype(np.float32)
    interpolated_inputs = [
        baseline + (step / num_steps) * (inputs - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_inputs = np.array(interpolated_inputs).astype(np.float32)

    # 3. Get the gradients
    grads = []
    for i, x_data in enumerate(interpolated_inputs):
        grad = get_gradients(x_data, top_pred_idx=top_pred_idx)
        grads.append(grad)
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    # 5. Calculate integrated gradients and return
    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads


hi = get_gradients(Xtrain)
print(hi.shape)

ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))

contours = np.arange(268, 305, 4)

cb = plt.contourf(lons, lats, tos_hist_base, contours,
    transform=ccrs.PlateCarree(), cmap=get_cmap('inferno'))

ax.coastlines()

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
    linewidth=1, color='dimgray', alpha=0.5, linestyle='--')

gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabels_bottom = False
gl.ylabels_left = False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([0, 90, 180, -90, 0])
gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
#gl.xlabel_style = {'size': 10, 'color': 'gray'}
#gl.ylabel_style = {'size': 10, 'color': 'gray'}

plt.subplots_adjust(bottom = 0.1,hspace=0.3,wspace=0.0)

cbar = plt.colorbar(cb, orientation="horizontal")
cbar.set_label(r'Precip GF sensitivity',fontsize=12)

plt.show()
#plt.savefig('../figures/hist_tos_base.png', dpi=300)
#plt.close()
