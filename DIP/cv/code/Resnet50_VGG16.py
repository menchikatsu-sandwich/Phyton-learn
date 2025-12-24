# -*- coding: utf-8 -*-
"""
Converted to PyTorch: Grad-CAM and feature-map visualizer for VGG16 and ResNet50
"""

import os
# Workaround for OpenMP duplicate-runtime error seen on some Windows/conda setups.
# This allows the process to continue but is an unsafe workaround. Prefer fixing
# conflicting OpenMP runtimes in the environment if possible.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# ==================================================
# Config
# ==================================================
img_path = r"C:\Users\santo\Documents\coding\PY\DIP\cv\train\healthy\IMG_20230630_194505_result.jpg"  # ganti sesuai lokasi gambar kamu
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])


# ==================================================
# Helpers: get module by dotted name
# ==================================================
def get_module(model, target_layer_name):
    # model.get_submodule is available in recent torch; fallback to manual search
    try:
        return model.get_submodule(target_layer_name)
    except Exception:
        # walk named_modules
        for name, mod in model.named_modules():
            if name == target_layer_name:
                return mod
    raise ValueError(f"Layer '{target_layer_name}' not found in model")


# ==================================================
# Grad-CAM implementation for PyTorch
# ==================================================
def make_gradcam_heatmap(img_tensor, model, target_layer_name):
    model.eval()
    activations = None
    gradients = []

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()
        # attach a hook on the activation tensor to capture its gradient
        def save_grad(grad):
            gradients.append(grad.detach())
        out.register_hook(save_grad)

    target_module = get_module(model, target_layer_name)
    fh = target_module.register_forward_hook(forward_hook)

    # forward
    output = model(img_tensor)
    preds = output

    class_idx = preds.argmax(dim=1).item()

    # backward on the predicted class score
    loss = preds[0, class_idx]
    model.zero_grad()
    loss.backward(retain_graph=True)

    # remove forward hook
    fh.remove()

    if len(gradients) == 0:
        raise RuntimeError("Gradients were not captured for the target layer; check layer name and that model is on the same device as input")

    grad = gradients[0]
    # pooled gradients across spatial dims (H,W)
    pooled_grads = torch.mean(grad, dim=(2, 3))  # shape [batch, channels]

    # weight the channels by corresponding gradients
    activ = activations[0]
    pooled = pooled_grads[0][:, None, None]
    weighted = activ * pooled
    heatmap = weighted.sum(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-10)
    return heatmap


def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # heatmap: 2D numpy in [0,1], original_img: HxW*3 uint8
    hmap = np.uint8(255 * heatmap)
    hmap = cv2.resize(hmap, (original_img.shape[1], original_img.shape[0]))
    colored = cv2.applyColorMap(hmap, colormap)
    overlayed = colored.astype(float) * alpha + original_img.astype(float)
    overlayed = overlayed / np.max(overlayed)
    overlayed = np.clip(overlayed * 255, 0, 255).astype('uint8')
    return overlayed


# ==================================================
# Feature map visualizer
# ==================================================
def get_feature_map(model, img_tensor, target_layer_name):
    feature = None

    def forward_hook(module, inp, out):
        nonlocal feature
        feature = out.detach()

    target_module = get_module(model, target_layer_name)
    fh = target_module.register_forward_hook(forward_hook)
    _ = model(img_tensor)
    fh.remove()
    fmap = feature[0].cpu().numpy()
    return fmap


def visualize_feature_maps_grid(fmap, max_filters=16):
    # fmap shape: H x W x C
    H, W, C = fmap.shape
    n = min(C, max_filters)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.zeros((rows * H, cols * W), dtype=np.uint8)
    for i in range(n):
        r = i // cols
        c = i % cols
        f = fmap[:, :, i]
        f = f - f.mean()
        f = f / (f.std() + 1e-6)
        f = f * 64 + 128
        f = np.clip(f, 0, 255).astype('uint8')
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = f
    return grid


# ==================================================
# Model layer mappings (PyTorch-friendly)
# ==================================================
vgg_layers = {
    "Low (block1_conv2)": "features.3",
    "Mid (block3_conv3)": "features.16",
    "High (block5_conv3)": "features.28",
}

resnet_layers = {
    "Low (layer1)": "layer1",
    "Mid (layer2)": "layer2",
    "High (layer4)": "layer4",
}


def load_image(path):
    img = Image.open(path).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return img, img_resized, tensor


def main():
    # load image
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    orig_img, vis_img, img_tensor = load_image(img_path)
    vis_img_np = np.array(vis_img)  # HxWx3 uint8

    # load models
    vgg_model = models.vgg16(pretrained=True).to(DEVICE)
    resnet_model = models.resnet50(pretrained=True).to(DEVICE)

    # Prepare figure
    plt.figure(figsize=(12, 10))

    models_dict = {
        "VGG16": (vgg_model, vis_img_np, vgg_layers),
        "ResNet50": (resnet_model, vis_img_np, resnet_layers),
    }

    col = 0
    for title, (model, vis_np, layers) in models_dict.items():
        # Grad-CAM for each level
        for i, (level_name, layer_name) in enumerate(layers.items()):
            heatmap = make_gradcam_heatmap(img_tensor, model, layer_name)
            overlayed = overlay_heatmap(heatmap, vis_np)
            plt.subplot(4, 3, i + 1 + (3 * col))
            plt.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
            plt.title(f"{title} {level_name}\nGrad-CAM")
            plt.axis('off')

        # Feature maps
        for i, (level_name, layer_name) in enumerate(layers.items()):
            fmap = get_feature_map(model, img_tensor, layer_name)  # C,H,W
            # convert to H,W,C
            if fmap.ndim == 3:
                fmap_hw_c = np.transpose(fmap, (1, 2, 0))
            else:
                # fallback
                fmap_hw_c = fmap
            grid = visualize_feature_maps_grid(fmap_hw_c)
            plt.subplot(4, 3, i + 4 + (3 * col))
            plt.imshow(grid, cmap='viridis')
            plt.title(f"{title} {level_name}\nFeature Maps")
            plt.axis('off')

        col += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

