import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from shapely.geometry import Polygon
from shrink import RSMTD  # Replace with your actual module

# -------------------------------
# Utility Functions
# -------------------------------

def preprocess_image(image_path):
    """Preprocess the input image for the model."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((640, 640))  # Match training input size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image  # Return both tensor and PIL image

def post_process(shrink_mask, offsets, shrink_threshold=0.3, dilation_size=32):
    """
    Post-process the shrink_mask to generate predicted text contours.
    Uses dilation and connected component analysis.
    """
    shrink_mask = torch.sigmoid(shrink_mask[0, 0]).cpu().numpy()
    binary_mask = (shrink_mask >= shrink_threshold).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    
    reconstructed_contours = []
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        dilated = cv2.dilate(component_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            reconstructed_contours.extend(contours)
    
    return reconstructed_contours


'''
def post_process(shrink_mask, offsets, shrink_threshold=0.5):
    """
    Post-process the shrink-mask and offsets to reconstruct full text contours.
    Applies offsets along the contour normal direction.
    """
    # Convert shrink-mask to binary
    shrink_mask = torch.sigmoid(shrink_mask[0, 0]).cpu().numpy()
    binary_mask = (shrink_mask >= shrink_threshold).astype(np.uint8) * 255
    
    # Extract contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract offset map
    offset_map = offsets[0, 0].cpu().numpy()  # Shape: (640, 640)
    
    # Debugging: Print offset statistics
    print(f"Offset map min: {offset_map.min():.2f}, max: {offset_map.max():.2f}, mean: {offset_map.mean():.2f}")
    
    reconstructed_contours = []
    for contour in contours:
        if len(contour) < 3:  # Skip invalid contours
            continue
        
        new_contour = []
        contour = contour.squeeze()  # Shape: (N, 2)
        num_points = len(contour)
        
        for i in range(num_points):
            x, y = contour[i]
            
            # Estimate tangent using neighboring points (finite difference)
            prev_idx = (i - 1) % num_points
            next_idx = (i + 1) % num_points
            prev_x, prev_y = contour[prev_idx]
            next_x, next_y = contour[next_idx]
            
            # Tangent vector
            tx = next_x - prev_x
            ty = next_y - prev_y
            tangent_len = np.sqrt(tx**2 + ty**2)
            if tangent_len == 0:
                new_contour.append([float(x), float(y)])
                continue
            
            # Normalize tangent
            tx /= tangent_len
            ty /= tangent_len
            
            # Normal vector (perpendicular to tangent, outward direction depends on contour orientation)
            # Assuming clockwise contours; swap nx, ny signs if counterclockwise
            nx = -ty
            ny = tx
            
            # Get offset value with bounds checking
            offset_val = offset_map[int(y), int(x)] if (0 <= int(y) < offset_map.shape[0] and 
                                                        0 <= int(x) < offset_map.shape[1]) else 0
            
            # Adjust for negative offsets (inward) or scale if needed
            offset_val = max(offset_val, 0)  # Ensure outward extension (adjust based on model output)
            
            # Move point along normal by offset value
            new_x = x + offset_val * nx
            new_y = y + offset_val * ny
            new_contour.append([new_x, new_y])
        
        reconstructed_contours.append(np.array(new_contour, dtype=np.float32))
    
    return reconstructed_contours
'''

def load_and_scale_gt(gt_path, scale_w, scale_h):
    """
    Load ground truth polygons from a text file and scale them to 640x640.
    Expected file format: x1,y1,x2,y2,x3,y3,x4,y4,label
    """
    polygons = []
    with open(gt_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 9 and parts[8] != '###':
                coords = list(map(float, parts[:8]))
                x = [int(coords[i] * scale_w) for i in range(0, 8, 2)]
                y = [int(coords[i] * scale_h) for i in range(1, 8, 2)]
                polygon = np.array(list(zip(x, y)), dtype=np.int32)
                polygons.append(polygon)
    return polygons

def generate_mask_from_polygons(polygons, size=(640, 640)):
    """Generate a binary mask from a list of polygons."""
    mask = np.zeros(size, dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly], 255)
    return mask

def visualize_results(image_path, pred_mask, gt_mask, output_path=None):
    """
    Overlay the predicted and groundtruth masks on the original image.
    Predicted mask is drawn in green and groundtruth in red.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    
    # Create colored masks
    pred_colored = np.zeros_like(image)
    pred_colored[:, :, 1] = pred_mask  # Green for predicted
    
    gt_colored = np.zeros_like(image)
    gt_colored[:, :, 2] = gt_mask      # Red for ground truth
    
    # Blend the image with the colored masks
    alpha = 0.3
    blended = cv2.addWeighted(image, 1, pred_colored, alpha, 0)
    blended = cv2.addWeighted(blended, 1, gt_colored, alpha, 0)
    
    if output_path:
        cv2.imwrite(output_path, blended)
        # print(f"Visualization saved to {output_path}")
    else:
        cv2.imshow('Visualization', blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# -------------------------------
# Metric Computation Functions
# -------------------------------

def compute_iou(contour, polygon):
    """Compute IoU between a detected contour and a groundtruth polygon."""
    if len(contour) < 3:
        return 0.0
    det_poly = Polygon(contour.squeeze())
    gt_poly = Polygon(polygon)
    if not det_poly.is_valid or not gt_poly.is_valid:
        return 0.0
    intersection = det_poly.intersection(gt_poly).area
    union = det_poly.union(gt_poly).area
    return intersection / union if union > 0 else 0.0

def compute_metrics(detected_contours, gt_polygons, iou_threshold=0.5):
    matched_gt = set()
    tp = 0
    for contour in detected_contours:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_poly in enumerate(gt_polygons):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(contour, gt_poly)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
    fp = len(detected_contours) - tp
    fn = len(gt_polygons) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f_measure, tp, fp, fn

# -------------------------------
# Main Evaluation and Visualization
# -------------------------------

def evaluate_dataset(model, images_dir, gt_dir, device, iou_threshold=0.5):
    """
    Evaluate the model on the entire dataset and compute overall metrics.
    Assumes that ground truth files are named with prefix 'gt_' matching image filenames.
    """
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    total_tp = total_fp = total_fn = total_gt = 0
    
    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        gt_name = f"gt_{os.path.splitext(image_name)[0]}.txt"
        gt_path = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path):
            print(f"GT file {gt_path} not found. Skipping {image_name}.")
            continue
        
        # Load image and get scaling factors
        image = Image.open(image_path).convert('RGB')
        original_w, original_h = image.size
        scale_w = 640 / original_w
        scale_h = 640 / original_h
        
        # Preprocess image
        input_tensor, _ = preprocess_image(image_path)
        input_tensor = input_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            shrink_mask, offsets, _ = model(input_tensor)
            # print(f"Offset size: {offsets.shape}")
        
        # Get predicted contours
        predicted_contours = post_process(shrink_mask, offsets)
        
        # Load and scale ground truth polygons
        gt_polygons = load_and_scale_gt(gt_path, scale_w, scale_h)

        total_gt += len(gt_polygons)
        
        # Compute metrics for this image
        single_precision, single_recall, single_f_measure, tp, fp, fn = compute_metrics(predicted_contours, gt_polygons, iou_threshold=iou_threshold)
        print(f'Image: {image_name}', end=', ')
        print(f"Precision: {single_precision}, Recall: {single_recall}, f-measure: {single_f_measure}", end=", ")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        total_tp += tp
        total_fp += fp
        total_fn += fn

        predicted_mask = np.zeros((640, 640), dtype=np.uint8)
        for contour in predicted_contours:
            if len(contour) >= 3:
                int_contour = np.round(contour).astype(np.int32)
                cv2.fillPoly(predicted_mask, [int_contour], 255)
        
        # Generate ground truth mask from polygons
        gt_mask = generate_mask_from_polygons(gt_polygons, size=(640, 640))

        output_vis_path = os.path.join('./experiments/shrink_exp2/', f'vis_{image_name}')
        visualize_results(image_path, predicted_mask, gt_mask, output_path=output_vis_path)

    # Compute overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f_measure = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    print("==== Overall Metrics ====")
    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}, Total Ground truth: {total_gt}")
    print(f"Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F-measure: {overall_f_measure:.4f}")

def visualize_single_image(model, image_path, gt_path, device, output_path=None):
    """
    Visualize predicted text regions overlaid on the original image alongside the GT mask.
    """
    # Load image and compute scaling factors
    image = Image.open(image_path).convert('RGB')
    original_w, original_h = image.size
    scale_w = 640 / original_w
    scale_h = 640 / original_h

    # Preprocess image
    input_tensor, _ = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        shrink_mask, offsets, _ = model(input_tensor)
    
    # Get predicted contours and build predicted mask
    predicted_contours = post_process(shrink_mask, offsets)
    predicted_mask = np.zeros((640, 640), dtype=np.uint8)
    for contour in predicted_contours:
        cv2.fillPoly(predicted_mask, [contour], 255)
    
    # Load and scale ground truth polygons, then generate GT mask
    gt_polygons = load_and_scale_gt(gt_path, scale_w, scale_h)
    gt_mask = generate_mask_from_polygons(gt_polygons, size=(640, 640))
    
    # Visualize overlay (predicted in green, ground truth in red)
    visualize_results(image_path, predicted_mask, gt_mask, output_path)

# -------------------------------
# Main Script
# -------------------------------

if __name__ == '__main__':
    # Define dataset and model paths (update these to your actual file locations)
    images_dir = r"D:\Projects\RPP\Sem 7\RISTE\data\raw\icdar_og\ch4_test_images - Copy"
    gt_dir = r"D:\Projects\RPP\Sem 7\RISTE\data\raw\icdar_og\Challenge4_Test_Task4_GT"
    model_path = r'.\checkpoints\models\rsmtd_finetune_final.pth'
    
    # Set device and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RSMTD().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path} on device {device}")
    
    # Evaluate overall metrics on the dataset
    evaluate_dataset(model, images_dir, gt_dir, device, iou_threshold=0.15)
    
    # Visualize one test image (update filename as needed)
    test_image = os.path.join(images_dir, 'img_94.jpg')
    test_gt = os.path.join(gt_dir, 'gt_img_94.txt')
    visualize_single_image(model, test_image, test_gt, device, output_path='visualization_img_94.png')
