import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import argparse
import os
import string
import validators
from models.rsmtd import RSMTD  # Assumes RSMTD model class is defined in shrink.py
from utils.infer_utils import TokenLabelConverter, ViTSTRFeatureExtractor  # Assumes these are from the ViTSTR codebase
from models.vision_transformer import *  # Import ViTSTR class (adjust path as necessary)
import matplotlib.pyplot as plt

def get_attention_hook(attention_maps):
    """Returns a hook function that computes and stores attention weights."""
    def hook(module, input, output):
        x = input[0]  # Input to the attention module
        B, N, C = x.shape  # Batch size, sequence length, channels
        # Compute Q, K, V from the qkv layer
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Queries, Keys, Values
        # Compute attention weights
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)  # Softmax over the last dimension
        attention_maps.append(attn)  # Store the attention map
    return hook
'''
def visualize_attention(image, attention_maps, output_path):
    """Visualize attention maps for a given image."""
    fig, axes = plt.subplots(1, len(attention_maps) + 1, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    for i, attn in enumerate(attention_maps):
        # attn shape: [B, num_heads, N, N], assume B=1
        attn = attn.mean(dim=1).squeeze(0)  # Average over heads: [N, N]
        attn = attn[0]  # Attention from first token: [N]
        grid_size = int(np.sqrt(attn.size(0)))
        attn = attn.view(grid_size, grid_size).cpu().numpy()
        axes[i+1].imshow(attn, cmap='hot')
        axes[i+1].set_title(f"Layer {i}")
    plt.savefig(output_path)
    plt.close()
'''
def visualize_attention(cropped_image, attention_maps, output_dir, crop_idx):
    """Visualize attention maps for a single cropped image and save overlays"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PIL image to numpy array for OpenCV processing
    original_np = np.array(cropped_image)
    original_cv = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    h, w = original_cv.shape[:2]
    
    # Create a subdirectory for this crop
    crop_dir = os.path.join(output_dir, f"crop_{crop_idx}")
    os.makedirs(crop_dir, exist_ok=True)
    
    # Save original cropped image
    original_path = os.path.join(crop_dir, "original.png")
    cv2.imwrite(original_path, original_cv)
    
    for layer_idx, attn in enumerate(attention_maps):
        # Process attention map
        attn_avg = attn.mean(dim=1).squeeze(0)  # Average over heads
        cls_attention = attn_avg[0, 1:]  # CLS token attention to patches
        grid_size = int(np.sqrt(cls_attention.size(0)))
        attn_grid = cls_attention.view(grid_size, grid_size).cpu().numpy()
        
        # 1. Save attention heatmap
        plt.figure(figsize=(5, 5))
        plt.imshow(attn_grid, cmap='hot')
        plt.axis('off')
        heatmap_path = os.path.join(crop_dir, f"layer_{layer_idx}_heatmap.png")
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 2. Create and save attention overlay
        attn_resized = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize and convert to heatmap
        attn_normalized = (attn_resized - attn_resized.min()) 
        attn_normalized /= attn_resized.max() - attn_resized.min()
        attn_uint8 = (attn_normalized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
        
        # Blend with original image
        overlay = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)
        overlay_path = os.path.join(crop_dir, f"layer_{layer_idx}_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        
        # 3. Save raw attention data
        np.save(os.path.join(crop_dir, f"layer_{layer_idx}_attn.npy"), attn_grid)
# --- Utility Functions ---

def preprocess_image(image_path, device):
    """Preprocess the input image for the RSMTD model."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((640, 640))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def post_process(shrink_mask, offsets, threshold=0.4):
    """Post-process the shrink mask to extract contours of text regions."""
    shrink_mask = torch.sigmoid(shrink_mask[0, 0]).cpu().numpy()
    binary_mask = (shrink_mask > threshold).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    kernel_size = 32
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    reconstructed_contours = []
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        dilated = cv2.dilate(component_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reconstructed_contours.extend(contours)
    return filter_and_merge_contours(reconstructed_contours)

def crop_text_regions(image, contours, output_dir=None):
    cropped_images = []
    file_paths = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image[y:y+h, x:x+w]
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        cropped_images.append(cropped_pil)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"cropped_image_{i}.png")
            cropped_pil.save(file_path)
            file_paths.append(file_path)
    return cropped_images, file_paths

def img2text(model, images, converter, attention_maps=None):
    """Recognize text in the given images using the ViTSTR model and optionally collect attention maps."""
    pred_strs = []
    attention_maps_all = [] if attention_maps is not None else None
    with torch.no_grad():
        for img in images:
            if attention_maps is not None:
                attention_maps.clear()  # Clear attention maps for this image
            pred = model(img, seqlen=converter.batch_max_length)
            _, pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
            pred_index = pred_index.view(-1, converter.batch_max_length)
            length_for_pred = torch.IntTensor([converter.batch_max_length - 1])
            pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
            pred_EOS = pred_str[0].find('[s]')
            pred_str = pred_str[0][:pred_EOS] if pred_EOS != -1 else pred_str[0]
            pred_strs.append(pred_str)
            if attention_maps is not None:
                attention_maps_all.append([attn.clone().detach() for attn in attention_maps])
    return pred_strs, attention_maps_all

def filter_and_merge_contours(contours):
    filtered_contours = []
    for c1 in contours:
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        is_subset = False
        for c2 in contours:
            if c1 is c2:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(c2)
            if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                is_subset = True
                break
        if not is_subset:
            filtered_contours.append(c1)
    return filtered_contours

def visualize_results(image_path, contours, output_path=None):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    mask = np.zeros((640, 640), dtype=np.uint8)
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 1] = mask  # Green channel
    alpha = 0.3
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    if output_path:
        cv2.imwrite(output_path, blended)
        print(f"Result saved to {output_path}")
    else:
        cv2.imshow('Detected Text Mask', blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description='Integrated Text Detection and Recognition with RSMTD and ViTSTR')
    parser.add_argument('--shrink_model_path', type=str, required=True, help='Path to the trained RSMTD model weights')
    parser.add_argument('--vitstr_model', type=str, required=True, help='Path or URL to the ViTSTR model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--quantized', action='store_true', help='Use quantized ViTSTR model')
    parser.add_argument('--rpi', action='store_true', help='Use settings optimized for Raspberry Pi')
    parser.add_argument('--save_cropped', action='store_true', help='Save cropped text regions to output_dir')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")

    # Load RSMTD model
    shrink_model = RSMTD().to(device)
    shrink_model.load_state_dict(torch.load(args.shrink_model_path, map_location=device))
    shrink_model.eval()

    # Set up ViTSTR converter
    character = string.printable[:-6]
    vitstr_args = argparse.Namespace(character=character, batch_max_length=25)
    converter = TokenLabelConverter(vitstr_args)

    # Load ViTSTR model
    if validators.url(args.vitstr_model):
        checkpoint = args.vitstr_model.rsplit('/', 1)[-1]
        torch.hub.download_url_to_file(args.vitstr_model, checkpoint)
    else:
        checkpoint = args.vitstr_model

    if args.quantized:
        backend = "qnnpack" if args.rpi else "fbgemm"
        torch.backends.quantized.engine = backend
        model = torch.jit.load(checkpoint)
    else:
        num_classes = len(converter.character)
        model = TextRecognition(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=num_classes,
            in_chans=1
        )
        state_dict = torch.load(checkpoint, map_location=device)
        adjusted_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'module.vitstr.' prefix if present
            new_key = key.replace('module.vitstr.', '')
            adjusted_state_dict[new_key] = value
        
        # Load the adjusted state dictionary into the model
        model.load_state_dict(adjusted_state_dict)
    if args.gpu:
        model.to(device)
    model.eval()

    # Register hooks for attention maps
    if not args.quantized and hasattr(model, 'blocks'):
        attention_maps = []
        for block in model.blocks:  # Adjust if attribute name differs
            if hasattr(block, 'attn'):
                block.attn.register_forward_hook(get_attention_hook(attention_maps))
    else:
        attention_maps = None
        if args.quantized:
            print("Attention maps not supported for quantized models.")
        else:
            print("Model does not have 'blocks' attribute; attention maps not supported.")

    # Preprocess input image for RSMTD
    image_tensor = preprocess_image(args.image_path, device)

    # Detect text regions with RSMTD
    with torch.no_grad():
        shrink_mask, offsets, _ = shrink_model(image_tensor)

    # Extract contours
    contours = post_process(shrink_mask, offsets)
    print(f"Detected {len(contours)} text regions")
    if len(contours) == 0:
        print("No text regions detected.")
        return

    # Crop text regions
    original_image = cv2.imread(args.image_path)
    original_image = cv2.resize(original_image, (640, 640))
    cropped_pil_images, cropped_pil_images_paths = crop_text_regions(original_image, contours, args.output_dir if args.save_cropped else None)

    visualize_results(args.image_path, contours, os.path.join(args.output_dir, "visualized_result.png"))

    # Prepare images for ViTSTR
    extractor = ViTSTRFeatureExtractor()
    images = [extractor(img) for img in cropped_pil_images_paths]
    if args.gpu:
        images = [img.to(device) for img in images]

    # Recognize text and collect attention maps
    pred_strs, attention_maps_all = img2text(model, images, converter, attention_maps)

    # Output results
    print("\nRecognized Texts:")
    for i, text in enumerate(pred_strs):
        print(f"Text region {i}: {text}")

    # Visualize attention maps
    if attention_maps_all is not None:
        for i, (img, maps) in enumerate(zip(cropped_pil_images, attention_maps_all)):
            output_path = os.path.join(args.output_dir, f"attention_image_{i}.png")
            visualize_attention(img, maps, output_path, i)
            print(f"Attention maps for image {i} saved to {output_path}")

if __name__ == "__main__":
    main()