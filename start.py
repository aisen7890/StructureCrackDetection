import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
from scipy.spatial import distance
from skimage.draw import line as draw_line  # Added import
import base64
from openai import OpenAI
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import transforms


    

# Initialize models and APIs for crack analysis pipeline:
# - CNN for initial crack detection
# - SAM2 for precise segmentation
# - GPT-4V for depth estimation

# OpenAI setup for depth estimation
client = OpenAI(api_key='sk-proj-lf5goAiCA7w91BVNUW5MT3BlbkFJZKVjgCM9hQjuqAPTr6uQ')



# SAM2 Model setup for precise segmentation
sam2_checkpoint = "checkpoints\\sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
model_SAM2 = build_sam2(model_cfg, sam2_checkpoint)
predictor = SAM2ImagePredictor(model_SAM2)




# CNN Model setup for initial crack detection
checkpoint_path = "checkpoints\\model_epoch_30.pth"


# CNN architecture for binary crack segmentation
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(3, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256)
        )
        self.decoder = nn.Sequential(
            self.upconv_block(256, 128),
            self.upconv_block(128, 64),
            self.upconv_block(64, 32)
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return self.sigmoid(x)
    
# Load model function
def load_model(checkpoint_path):
    model = SimpleCNN().cuda()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

model_CNN = load_model(checkpoint_path)




# Directory containing images
image_dir = "image"
output_dir = "output_image"



# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Initialize three-point prompting for SAM2
points = []
labels = np.array([1, 1, 1], dtype=np.int32)  # Three foreground points for better coverage


# Enhanced function to handle text placement and avoid overlaps
def place_text_avoiding_overlap(text_width, text_height, image_width, image_height, occupied_rects,
                                preferred_position=None, margin=10, boundary_margin=10):
    positions = []

    # Adjust image boundaries to account for boundary_margin
    adjusted_image_width = image_width - boundary_margin * 2
    adjusted_image_height = image_height - boundary_margin * 2

    if preferred_position is not None:
        positions.append(preferred_position)

        # Try positions in a spiral pattern around the preferred position
        pref_x, pref_y = preferred_position
        max_distance = max(image_width, image_height)
        for distance in range(margin, int(max_distance), margin):
            for angle in np.linspace(0, 2*np.pi, num=36, endpoint=False):
                x = pref_x + distance * np.cos(angle)
                y = pref_y + distance * np.sin(angle)
                x = int(x)
                y = int(y)
                # Adjust positions if they are outside the image boundaries
                x = max(boundary_margin, min(x, image_width - text_width - boundary_margin))
                y = max(boundary_margin, min(y, image_height - text_height - boundary_margin))
                positions.append((x, y))
    else:
        # If no preferred position, generate positions over the whole image
        for y in range(boundary_margin, image_height - text_height - boundary_margin, margin):
            for x in range(boundary_margin, image_width - text_width - boundary_margin, margin):
                positions.append((x, y))

    # Remove duplicates
    positions = list(dict.fromkeys(positions))

    # Try each position and check for overlaps
    for x, y in positions:
        overlap = False
        for rect in occupied_rects:
            rect_x, rect_y, rect_w, rect_h = rect
            if (x < rect_x + rect_w and x + text_width > rect_x and
                y < rect_y + rect_h and y + text_height > rect_y):
                overlap = True
                break
        if not overlap:
            return x, y  # Found a suitable position

    # If no suitable position found, return None
    return None




# Main processing loop for crack analysis
for image_file in os.listdir(image_dir):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, image_file)

        #########################
        # CNN-based binary image#
        #########################
        transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        ])

        image_for_CNN = Image.open(image_path).convert('RGB')
        img_tensor = transform(image_for_CNN).unsqueeze(0).cuda()
    
        with torch.no_grad():
            output = model_CNN(img_tensor)
    
        output_img_raw = (output.squeeze().cpu().numpy() > 0.5).astype('uint8') * 255  # Binary mask
        

        # Step 2: Extract and filter significant contours (>5% area)
        contours, _ = cv2.findContours(output_img_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the total area of all contours
        total_contour_area = sum(cv2.contourArea(cnt) for cnt in contours)

        # Determine the area threshold (5% of total contour area)
        threshold_area = 0.05 * total_contour_area #마스킹 영역이 5% 이상인 경우만 계산에 포함 (균열의 일부로 보기 힘든 작은 마스킹 부분들을 필터링하는 용도)

        # Filter contours to include only those larger than the threshold area
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

        # Sort the filtered contours by area in descending order
        sorted_contours = sorted(large_contours, key=cv2.contourArea, reverse=True)

        # Ensure there are contours to process
        if len(sorted_contours) == 0:
            print(f"No contours found in {image_file}. Skipping image.")
            continue  # Skip to the next image

        # Calculate the area of the largest contour
        largest_contour_area = cv2.contourArea(sorted_contours[0])

        # Calculate the ratio of the largest contour area to the total contour area
        if total_contour_area > 0:
            largest_contour_ratio = largest_contour_area / total_contour_area
        else:
            largest_contour_ratio = 0  # Avoid division by zero

        # Check if the largest contour occupies more than 85% of the total contour area
        if largest_contour_ratio > 0.85:
            # Only use the largest contour
            contours_to_process = [sorted_contours[0]]
        else:
            # Use up to two largest contours
            contours_to_process = sorted_contours[:2]

        # Step 3: Select three strategic points for SAM2 segmentation
        # Points are chosen to maximize coverage of the crack area
        coordinates = []
        height, width = output_img_raw.shape
        valid_contours = []
        points_found = 0

        # Process each selected contour
        for contour in contours_to_process:
            # Create an empty mask for the image
            mask = np.zeros_like(output_img_raw)

            # Draw and fill the contour on the mask
            cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

            # Get the coordinates of all pixels inside the contour
            ys, xs = np.where(mask == 255)
            contour_pixels = list(zip(xs, ys))

            # Filter pixels that are at least 10 pixels away from the boundary
            valid_pixels = []
            for x, y in contour_pixels:
                # Calculate distances to the four edges
                distance_to_left = x
                distance_to_right = width - x - 1
                distance_to_top = y
                distance_to_bottom = height - y - 1
                # Minimum distance to any edge
                distance_from_edge = min(distance_to_left, distance_to_right, distance_to_top, distance_to_bottom)
                # Only consider points at least 10 pixels away from the boundary
                if distance_from_edge >= 10:
                    valid_pixels.append((x, y))

            if valid_pixels:
                # Select the point closest to the boundary
                min_distance = float('inf')
                closest_point = None

                for x, y in valid_pixels:
                    # Calculate distances to the four edges
                    distance_to_left = x
                    distance_to_right = width - x - 1
                    distance_to_top = y
                    distance_to_bottom = height - y - 1
                    # Minimum distance to any edge
                    distance_from_edge = min(distance_to_left, distance_to_right, distance_to_top, distance_to_bottom)
                    # Update if this point is closer to the boundary
                    if distance_from_edge < min_distance:
                        min_distance = distance_from_edge
                        closest_point = [int(x), int(y)]

                # Append the closest point to the list
                coordinates.append(closest_point)
                valid_contours.append((valid_pixels, closest_point))
                points_found += 1
            else:
                # No valid point found in this contour
                pass

        # Handle cases where less than three points were found
        if points_found == 3:
            # We have three points, proceed
            pass
        elif points_found == 2:
            # Two points found, need one more
            valid_pixels, first_point = valid_contours[0]
            
            # Find the point that maximizes the minimum distance to both existing points
            max_min_distance = -1
            third_point = None
            
            for x, y in valid_pixels:
                # Calculate distances to both existing points
                dist1 = np.sqrt((x - coordinates[0][0])**2 + (y - coordinates[0][1])**2)
                dist2 = np.sqrt((x - coordinates[1][0])**2 + (y - coordinates[1][1])**2)
                min_dist = min(dist1, dist2)
                
                if min_dist > max_min_distance:
                    max_min_distance = min_dist
                    third_point = [int(x), int(y)]
            
            if third_point:
                coordinates.append(third_point)
                points_found += 1
            else:
                print(f"No suitable third point found in contour for {image_file}. Skipping image.")
                continue
        elif points_found == 1:
            # Only one point found, need two more
            valid_pixels, first_point = valid_contours[0]
            
            # Find the point farthest from the first point
            max_distance = -1
            second_point = None
            
            for x, y in valid_pixels:
                distance = np.sqrt((x - first_point[0])**2 + (y - first_point[1])**2)
                if distance > max_distance:
                    max_distance = distance
                    second_point = [int(x), int(y)]
            
            if second_point:
                coordinates.append(second_point)
                points_found += 1
                
                # Find the third point that maximizes the minimum distance to both existing points
                max_min_distance = -1
                third_point = None
                
                for x, y in valid_pixels:
                    dist1 = np.sqrt((x - coordinates[0][0])**2 + (y - coordinates[0][1])**2)
                    dist2 = np.sqrt((x - coordinates[1][0])**2 + (y - coordinates[1][1])**2)
                    min_dist = min(dist1, dist2)
                    
                    if min_dist > max_min_distance:
                        max_min_distance = min_dist
                        third_point = [int(x), int(y)]
                
                if third_point:
                    coordinates.append(third_point)
                    points_found += 1
                else:
                    print(f"No suitable third point found in contour for {image_file}. Skipping image.")
                    continue
            else:
                print(f"No suitable second point found in contour for {image_file}. Skipping image.")
                continue
        else:
            # No valid points found
            print(f"No valid points found in contours for {image_file}. Skipping image.")
            continue

        # Now coordinates is a list of three [x, y] lists
        points_np = coordinates  # Format: [[x1, y1], [x2, y2], [x3, y3]]
        


        #########################
        #SAM2-based segmentation#
        #########################
        # Step 4: Precise segmentation with SAM2 using three points
        input_prompts = {'points': points_np, 'labels': labels}

        image_pil = Image.open(image_path)
        image_rgb = np.array(image_pil)
        image_rgb = cv2.resize(image_rgb, (300, 300), interpolation=cv2.INTER_LINEAR)

        # Load the image
        predictor.set_image(image_rgb)

        # Perform segmentation using the selected points
        with torch.inference_mode():
            masks, _, _ = predictor.predict(point_coords=points_np, point_labels=labels)

        # Get the mask as a numpy array
        mask = (torch.tensor(masks[0]) > 0.0).cpu().numpy()

        # Load the image as a NumPy array
        image_array = np.array(Image.open(image_path))

        # Ensure the mask is the same size as the image
        if mask.shape != image_array.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        # Create an RGBA version of the image
        if image_array.shape[2] == 3:
            # Add an alpha channel
            image_array_rgba = np.concatenate([image_array, 255 * np.ones((*image_array.shape[:2], 1), dtype=np.uint8)], axis=2)
        else:
            image_array_rgba = image_array.copy()

        # Define the mask color (e.g., red with 50% transparency)
        mask_color = np.array([255, 0, 0, 127], dtype=np.uint8)  # Red color with transparency

        # Apply the mask color where the mask is True
        image_array_rgba[mask] = mask_color

        result_image = Image.fromarray(image_array_rgba)

        draw_im = ImageDraw.Draw(result_image)

        font_size = 24
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)  # Adjust the path if needed
        except IOError:
            print("Using default font")
            font = ImageFont.load_default()

        occupied_rects = []

        # Get image dimensions
        image_width, image_height = result_image.size

        # Compute brightness for the segmented area
        masked_indices = np.argwhere(mask)
        if len(masked_indices) >= 30:
            selected_indices = masked_indices[np.random.choice(len(masked_indices), size=5, replace=False)]
        elif len(masked_indices) > 10:
            selected_indices = masked_indices
        else:
            print(f"No segmented pixels found in {image_file}.")
            continue  # Skip to the next image

        # Get the pixel values at these indices
        selected_pixels = image_array[selected_indices[:, 0], selected_indices[:, 1], :]  # shape (N, 3)

        # Compute brightness using luminance formula
        brightness_values = 0.2126 * selected_pixels[:, 0] + 0.7152 * selected_pixels[:, 1] + 0.0722 * selected_pixels[:, 2]

        # Calculate average brightness
        average_brightness = brightness_values.mean()

        # Preferred position for Brightness label (centroid of the mask)
        centroid_y, centroid_x = masked_indices.mean(axis=0)
        brightness_text = f"Crack_B: {average_brightness:.2f}"
        brightness_text_width, brightness_text_height = font.getsize(brightness_text)

        # Adjust position slightly below the centroid
        brightness_preferred_position = (centroid_x - brightness_text_width / 2, centroid_y + 10)

        # Position for Brightness label
        brightness_position = place_text_avoiding_overlap(
            brightness_text_width, brightness_text_height, image_width, image_height,
            occupied_rects, preferred_position=brightness_preferred_position, boundary_margin=10)

        if brightness_position is not None:
            brightness_x, brightness_y = brightness_position
            # Draw the red text for Brightness
            draw_im.text((brightness_x, brightness_y), brightness_text, fill=(255, 0, 0), font=font)
            occupied_rects.append((brightness_x, brightness_y, brightness_text_width, brightness_text_height))
        else:
            print(f"Could not find a suitable position for Brightness label in {image_file}.")

        # Compute brightness for the background
        background_indices = np.argwhere(~mask)
        if len(background_indices) >= 100:
            bg_selected_indices = background_indices[np.random.choice(len(background_indices), size=10, replace=False)]
        elif len(background_indices) > 10:
            bg_selected_indices = background_indices
        else:
            print(f"No background pixels found in {image_file}.")
            continue  # Skip to the next image

        # Get the pixel values at these indices
        bg_selected_pixels = image_array[bg_selected_indices[:, 0], bg_selected_indices[:, 1], :]  # shape (N, 3)

        # Compute brightness using luminance formula
        bg_brightness_values = 0.2126 * bg_selected_pixels[:, 0] + 0.7152 * bg_selected_pixels[:, 1] + 0.0722 * bg_selected_pixels[:, 2]

        # Calculate average background brightness
        bg_average_brightness = bg_brightness_values.mean()

        # Preferred position for Background Brightness label (somewhere else to avoid overlap)
        bg_brightness_text = f"Surface_B: {bg_average_brightness:.2f}"
        bg_brightness_text_width, bg_brightness_text_height = font.getsize(bg_brightness_text)

        # Try to place the background brightness label at the top-left corner
        bg_brightness_preferred_position = (10, 10)

        # Position for Background Brightness label
        bg_brightness_position = place_text_avoiding_overlap(
            bg_brightness_text_width, bg_brightness_text_height, image_width, image_height,
            occupied_rects, preferred_position=bg_brightness_preferred_position, boundary_margin=10)

        if bg_brightness_position is not None:
            bg_brightness_x, bg_brightness_y = bg_brightness_position
            # Draw the green text for Background Brightness
            draw_im.text((bg_brightness_x, bg_brightness_y), bg_brightness_text, fill=(0, 128, 0), font=font)
            occupied_rects.append((bg_brightness_x, bg_brightness_y, bg_brightness_text_width, bg_brightness_text_height))
        else:
            print(f"Could not find a suitable position for Background Brightness label in {image_file}.")



        # Save the first image as PNG (밝기만 표시되어 있는 균열 사진 저장)
        output_image_path1 = os.path.join(output_dir, f"output_segmented_{os.path.splitext(image_file)[0]}_1.png")
        result_image.save(output_image_path1)


        #########################
        #   Depth calculation   #
        #########################

        #이미지를 gpt 형식으로 변환
        img_byte_arr = BytesIO()
        result_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        img_encoded = base64.b64encode(img_byte_arr).decode('utf-8')


        # Now proceed to compute length and thickness
        # Get the contours from the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the total area of all contours
        total_contour_area = sum(cv2.contourArea(cnt) for cnt in contours)

        # Determine the area threshold (10% of total contour area)
        threshold_area = 0.05 * total_contour_area #마스킹 영역이 5% 이상인 경우만 계산에 포함 (균열의 일부로 보기 힘든 작은 마스킹 부분들을 필터링하는 용도)

        # Filter contours to include only those larger than the threshold area
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

        if len(large_contours) > 0:
            # Initialize an empty list to hold contour points from significant contours
            contour_points = []
            
            for contour in large_contours:
                # Append all points of the current contour to contour_points
                contour_points.extend(contour[:, 0, :])  # Extract the (x, y) coordinates

            # Proceed with your existing code using contour_points
            # Compute the pairwise distance between all points on the contour
            distances = distance.cdist(contour_points, contour_points, 'euclidean')
            
            # Find the maximum distance (length of the segmented area)
            max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
            max_distance = distances[max_distance_idx]
            point1 = contour_points[max_distance_idx[0]]
            point2 = contour_points[max_distance_idx[1]]
            
            # Compute the blue line vector and its unit vector
            blue_line_vector = point2 - point1
            line_length = np.linalg.norm(blue_line_vector)
            line_direction = blue_line_vector / line_length
            
            # Compute the perpendicular unit vector
            perp_direction = np.array([-line_direction[1], line_direction[0]])
            
            # Sample points along the blue line
            num_samples = 100
            t_values = np.linspace(0, 1, num_samples)
            line_points = point1 + t_values[:, None] * blue_line_vector
            
            max_thickness = 0
            thickest_line_coords = None
            
            # Ensure mask is a boolean array
            mask_bool = mask.astype(bool)
            
            for lp in line_points:
                max_extent = max(mask.shape)
                end_point_pos = lp + max_extent * perp_direction
                end_point_neg = lp - max_extent * perp_direction
                
                # Convert to integer pixel coordinates
                y0, x0 = int(lp[1]), int(lp[0])
                y1, x1 = int(end_point_neg[1]), int(end_point_neg[0])
                y2, x2 = int(end_point_pos[1]), int(end_point_pos[0])
                
                # Get coordinates of the line from end_point_neg to end_point_pos
                rr, cc = draw_line(y1, x1, y2, x2)
                
                # Clip coordinates to image boundaries
                rr = np.clip(rr, 0, mask.shape[0] - 1)
                cc = np.clip(cc, 0, mask.shape[1] - 1)
                
                # Get mask values along the line
                mask_values = mask_bool[rr, cc]
                
                # Find indices where mask is True
                true_indices = np.where(mask_values)[0]
                
                if len(true_indices) >= 2:
                    first_true_idx = true_indices[0]
                    last_true_idx = true_indices[-1]
                    
                    # Get the coordinates of the intersection points
                    y_start, x_start = rr[first_true_idx], cc[first_true_idx]
                    y_end, x_end = rr[last_true_idx], cc[last_true_idx]
                    
                    # Calculate thickness (distance between the two points)
                    thickness = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
                    
                    # Update maximum thickness if current thickness is greater
                    if thickness > max_thickness:
                        max_thickness = thickness
                        thickest_line_coords = ((x_start, y_start), (x_end, y_end))
                else:
                    continue  # No valid thickness found along this line
        else:
            print(f"No contours larger than 10% of total area in {image_file}.")
            continue  # Skip to the next image

        # Now draw the blue line for the length
        draw_im.line([tuple(point1), tuple(point2)], fill=(0, 0, 255, 255), width=2)

        # Draw the orange line for the thickness
        if thickest_line_coords:
            draw_im.line(thickest_line_coords[0] + thickest_line_coords[1], fill=(255, 165, 0, 255), width=2)

        # Calculate positions and draw labels for Length and Thickness

        # Preferred position for Length label (midpoint of length line)
        length_midpoint = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
        length_text = f"Length: {max_distance:.2f}px"
        length_text_width, length_text_height = font.getsize(length_text)

        # Adjust position slightly above the line to avoid overlap
        length_preferred_position = (length_midpoint[0] - length_text_width / 2, length_midpoint[1] - length_text_height - 10)

        # Position for Length label
        length_position = place_text_avoiding_overlap(
            length_text_width, length_text_height, image_width, image_height,
            occupied_rects, preferred_position=length_preferred_position, boundary_margin=10)

        if length_position is not None:
            length_x, length_y = length_position
            # Draw the blue text for Length
            draw_im.text((length_x, length_y), length_text, fill=(0, 0, 255), font=font)
            occupied_rects.append((length_x, length_y, length_text_width, length_text_height))
        else:
            print(f"Could not find a suitable position for Length label in {image_file}.")

        # Preferred position for Thickness label (midpoint of thickness line)
        if thickest_line_coords:
            thickness_midpoint = ((thickest_line_coords[0][0] + thickest_line_coords[1][0]) / 2,
                                    (thickest_line_coords[0][1] + thickest_line_coords[1][1]) / 2)
        else:
            thickness_midpoint = length_midpoint  # Fallback to length midpoint if thickness line not found

        thickness_text = f"Thickness: {max_thickness:.2f}px"
        thickness_text_width, thickness_text_height = font.getsize(thickness_text)
        # Adjust position slightly above the line
        thickness_preferred_position = (thickness_midpoint[0] - thickness_text_width / 2, thickness_midpoint[1] - thickness_text_height - 10)

        # Position for Thickness label
        thickness_position = place_text_avoiding_overlap(
            thickness_text_width, thickness_text_height, image_width, image_height,
            occupied_rects, preferred_position=thickness_preferred_position, boundary_margin=10)

        if thickness_position is not None:
            thickness_x, thickness_y = thickness_position
            # Draw the orange text for Thickness
            draw_im.text((thickness_x, thickness_y), thickness_text, fill=(255, 165, 0), font=font)
            occupied_rects.append((thickness_x, thickness_y, thickness_text_width, thickness_text_height))
        else:
            print(f"Could not find a suitable position for Thickness label in {image_file}.")



        # Save the second image as PNG (밝기 + 치수가 표시되어 있는 균열 사진 저장)
        output_image_path2 = os.path.join(output_dir, f"output_segmented_{os.path.splitext(image_file)[0]}_2.png")
        result_image.save(output_image_path2)


        #균열 깊이 추정
        # Calculate normalized brightness values
        highest_brightness = max(average_brightness, bg_average_brightness)
        normalized_crack_brightness = average_brightness / highest_brightness
        normalized_surface_brightness = bg_average_brightness / highest_brightness

        # Save the original image
        original_image_path = os.path.join(output_dir, f"original_{os.path.splitext(image_file)[0]}.png")
        Image.open(image_path).save(original_image_path)

        # Save the masked image (just the mask)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image_path = os.path.join(output_dir, f"mask_{os.path.splitext(image_file)[0]}.png")
        mask_image.save(mask_image_path)

        # Save the annotated image (with measurements)
        annotated_image_path = os.path.join(output_dir, f"annotated_{os.path.splitext(image_file)[0]}.png")
        result_image.save(annotated_image_path)

        # Convert all images to base64
        def image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        original_img_encoded = image_to_base64(original_image_path)
        mask_img_encoded = image_to_base64(mask_image_path)
        annotated_img_encoded = image_to_base64(annotated_image_path)

        # Step 7: Depth estimation using GPT-4V
        # Analyzes images and normalized brightness values
        response_img = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """
                Role: You are a construction inspector tasked with assessing the depth of cracks on a concrete surface based on the provided images.

                Objective: Your goal is to estimate the depth of the cracks in millimeters (mm) using the provided images and normalized brightness values.

                Instructions:
                1. Analyze the three provided images:
                   - Original image: Shows the actual concrete surface
                   - Mask image: Shows the detected crack area
                   - Annotated image: Shows measurements and brightness values

                2. Use the following normalized brightness values:
                   - Crack brightness: {normalized_crack_brightness:.2f}

                3. Depth Estimation Guidelines:
                   - If normalized crack brightness < 0.6: Deep crack (>40mm)
                   - If 0.6 ≤ normalized crack brightness < 0.75: Medium crack (20-39mm)
                   - If normalized crack brightness ≥ 0.75: Shallow crack (0-19mm)

                4. Output: Provide your depth estimation in Korean using the following format:
                   '''
                   - 균열 깊이: n mm ~ n mm (표면/얕은 균열/중간 균열/깊은 균열)
                   - 근거: [간단한 설명]
                   '''
                """},
                {"role": "user", "content": [
                    {"type": "text", "text": "Please estimate the depth of the cracks using the provided images and brightness values."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_img_encoded}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{mask_img_encoded}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{annotated_img_encoded}"}}
                ]}
            ],
        )
        img_exp = response_img.choices[0].message.content.strip()

        # Save results to file
        file_path = f'{os.path.join(output_dir, f"output_segmented_{os.path.splitext(image_file)[0]}_3.txt")}'
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(f"Surface Brightness: {bg_average_brightness:.2f}\n")
            file.write(f"Crack Brightness: {average_brightness:.2f}\n")
            file.write(f"Normalized Surface Brightness: {normalized_surface_brightness:.2f}\n")
            file.write(f"Normalized Crack Brightness: {normalized_crack_brightness:.2f}\n")
            file.write("Estimated depth: " + img_exp + '\n')
            file.write(length_text + '\n')
            file.write(thickness_text + '\n')

        # Clean up temporary files
        os.remove(original_image_path)
        os.remove(mask_image_path)


















