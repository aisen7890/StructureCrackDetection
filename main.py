import streamlit as st
import os
from openai import OpenAI
import json
from PIL import Image, ImageGrab
import base64
import io
import tkinter as tk
from io import BytesIO
from datetime import datetime

import os
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import supervision as sv
import cv2
import numpy as np



#necessary SAM2 parameters

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# sam2_hiera_large.pt
# sam2_hiera_base_plus.pt
# sam2_hiera_small.pt
# sam2_hiera_tiny.pt
sam2_checkpoint = "checkpoints\\sam2_hiera_large.pt"


# sam2_hiera_l.yaml
# sam2_hiera_b+.yaml
# sam2_hiera_s.yaml
# sam2_hiera_t.yaml
model_cfg = "sam2_hiera_l.yaml"
image_dir = "image"
output_dir = "output_image"



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam2 = build_sam2(model_cfg, sam2_checkpoint, device =DEVICE, apply_postprocessing=False)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize the SAM2 predictor
#mask_generator  = SAM2AutomaticMaskGenerator(sam2)







# Set page config to wide mode
st.set_page_config(layout="wide")

# Set your OpenAI API key
client = OpenAI(api_key="{your_api_key}")

# Token pricing (in USD)
INPUT_TOKEN_PRICE = 0.00015  # $0.15 per 1000 input tokens
OUTPUT_TOKEN_PRICE = 0.0006  # $0.60 per 1000 output tokens

# Image pricing (in USD)
IMAGE_PRICE_PER_PIXEL = 0.000000001275  # $0.0001275 per 100x100 pixels

# Exchange rate
USD_TO_KRW = 1400  # 1 USD = 1400 KRW

# Maximum image size
MAX_IMAGE_SIZE = (1024, 1024)

def read_instruction_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    image.thumbnail(max_size, Image.LANCZOS)
    return image

def optimize_image(image, quality=85):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_image_cost(width, height):
    return width * height * IMAGE_PRICE_PER_PIXEL

def analyze_image(image, instruction):
    base64_image = optimize_image(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": "Please analyze the following image."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
        
        analysis_text = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        input_cost_usd = (prompt_tokens / 1000) * INPUT_TOKEN_PRICE
        output_cost_usd = (completion_tokens / 1000) * OUTPUT_TOKEN_PRICE
        
        # Calculate image cost
        width, height = image.size
        image_cost_usd = calculate_image_cost(width, height)
        
        total_cost_usd = input_cost_usd + output_cost_usd + image_cost_usd
        total_cost_krw = total_cost_usd * USD_TO_KRW
        
        return {
            "analysis": analysis_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "image_width": width,
            "image_height": height,
            "image_cost_usd": image_cost_usd,
            "total_cost_usd": total_cost_usd,
            "total_cost_krw": total_cost_krw
        }
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def get_image_from_clipboard():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    try:
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            return resize_image(img)
        elif img is None:
            return None
        else:
            st.warning("Clipboard contains data, but it's not an image.")
            return None
    except Exception as e:
        st.error(f"Error accessing clipboard: {e}")
        return None
    finally:
        root.destroy()

# Streamlit UI
st.title("구조물 이미지 진단 APP")
st.write("-필요한 기능 디코에 남겨주시면 반영하겠습니다.")
# Create two columns with custom widths
left_column, right_column = st.columns([1, 2])  # Adjust the ratio as needed

# Left column for image upload
with left_column:
    # Button to paste image from clipboard
    if st.button("Paste Image from Clipboard"):
        image = get_image_from_clipboard()
        if image is not None:
            st.session_state['image'] = image
            st.success("Image pasted from clipboard and resized if necessary!")
        else:
            st.warning("No image found in clipboard. Please copy an image first.")

    # File uploader
    uploaded_file = st.file_uploader("Or choose an image file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = resize_image(image)
        st.session_state['image'] = image
        st.success("Image uploaded and resized if necessary!")

    # Display the image if it exists in session state
    if 'image' in st.session_state:
        st.image(st.session_state['image'], caption='Uploaded/Pasted Image (Resized)', use_column_width=True)
        st.write(f"Image size: {st.session_state['image'].size[0]}x{st.session_state['image'].size[1]} pixels")

# Right column for analysis button and results
with right_column:
    # Get the instruction
    instruction = read_instruction_file("instruct.txt")

    # Analyze button
    analyze_button = st.button('Analyze Image')

    if analyze_button:
        if 'image' in st.session_state:
            with st.spinner('Analyzing...'):
                json_data = analyze_image(st.session_state['image'], instruction)
                
                if json_data is not None:
                    # Display the analysis result
                    st.subheader("Analysis Result:")
                    st.write(json_data["analysis"])
                    
                    # Display token usage and cost
                    st.subheader("Token Usage and Cost:")
                    st.write(f"Input tokens: {json_data['prompt_tokens']}")
                    st.write(f"Output tokens: {json_data['completion_tokens']}")
                    st.write(f"Total tokens: {json_data['total_tokens']}")
                    st.write(f"Image size: {json_data['image_width']}x{json_data['image_height']} pixels")
                    st.write(f"Image cost (USD): ${json_data['image_cost_usd']:.6f}")
                    st.write(f"Total cost (USD): ${json_data['total_cost_usd']:.6f}")
                    st.write(f"Total cost (KRW): ₩{json_data['total_cost_krw']:.2f}")
                    
                    # Add timestamp to JSON data
                    json_data['timestamp'] = datetime.now().isoformat()

                    # Create 'log' folder if it doesn't exist
                    os.makedirs('log', exist_ok=True)

                    # Save JSON to file in 'log' folder
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_json_path = f"log/image_analysis_{timestamp}.json"
                    with open(output_json_path, 'w', encoding='utf-8') as json_file:
                        json.dump(json_data, json_file, indent=4, ensure_ascii=False)
                    st.success(f"JSON file saved to: {output_json_path}")



                    # SAM2 segmentation
                    uploaded_img = st.session_state['image']
                    if uploaded_img.mode == 'RGBA':
                        uploaded_img2 = uploaded_img.convert('RGB')

                    image_rgb = np.array(uploaded_img2)
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                    mask_generator  = SAM2AutomaticMaskGenerator(
                        model=sam2,
                        # points_per_batch=128, #한번에 몇개의 점을 검토할지 결정
                        # points_per_side=2, #한 이미지 내에서 몇군데나 점을 찍어서 요소를 파악할지 결정(높은 숫자=더욱 촘촘하게 확인)
                        #pred_iou_thresh=0.7, #낮을수록 미세한 요소들도 잡아냄.
                        # stability_score_offset=0.9, #따로따로 segmentation이 분할되지 않고 일관성 있게 유지되게 하는 파라메터
                        # stability_score_thresh=0.92, #몇 % 확실하면 segmentation 하는지 결정하는 파라메터
                        # crop_n_layers=1, #이미지를 분할해서 인식하는 파라메터.
                        # box_nms_thresh=0., #많이 겹치는 영역을 하나로 인식. 낮을수록 하나로 합쳐짐.


                        points_per_batch=64,        # Increase points to evaluate more potential cracks
                        pred_iou_thresh=0.5,         # Lower IoU threshold to capture finer cracks
                        stability_score_offset=0.9,  # Increase offset to emphasize stable crack detection
                        stability_score_thresh=0.85, # Lower threshold to retain more crack masks
                        #crop_n_layers=1,             # Increase layers to focus on smaller areas where cracks may be
                        box_nms_thresh=0.4,          # Lower threshold to avoid missing small but important cracks
                        )
                    
                    masks = mask_generator.generate(image_rgb)

                    # Extract and sort masks by area
                    areas = [mask['area'] for mask in masks]
                    sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
                    top_masks = [masks[i] for i in sorted_indices[:5]] #상위 n개 표시

                    # Annotate image with mask areas
                    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
                    detections = sv.Detections.from_sam(sam_result=top_masks)
                    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

                    # Define text properties
                    font_scale = 0.6
                    font_thickness = 2
                    font_color = (255, 0, 0)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Get image dimensions
                    img_height, img_width = image_bgr.shape[:2]

                    used_positions = []

                    # Function to check for overlap with existing text
                    def adjust_position(x, y, text_width, text_height):
                        # Adjust positions to avoid overlapping
                        for ux, uy, uw, uh in used_positions:
                            if (x < ux + uw and x + text_width > ux) and (y < uy + uh and y + text_height > uy):
                                # Overlapping detected, adjust the y position to move down
                                y = uy + uh + 10  # Move down by the height of the overlapping text plus some padding
                                if y + text_height > img_height:  # If new position goes out of bounds, move it up
                                    y = uy - text_height - 10
                                break
                        # Add the final position to used_positions
                        used_positions.append((x, y, text_width, text_height))
                        return x, y


                    # Convert the image to grayscale for brightness calculation
                    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)



                                # Function to calculate average brightness for a list of points using grayscale image
                    brightness_values = []

                    def calculate_average_brightness(points):
                        for point in points:
                            x, y = int(point[0]), int(point[1])
                            # Ensure the point is within image bounds
                            if 0 <= x < img_width and 0 <= y < img_height:
                                # Use the grayscale value directly as brightness
                                brightness = image_gray[y, x]
                                brightness_values.append(brightness)
                        # Return the average brightness
                        if brightness_values:
                            return np.mean(brightness_values)
                        else:
                            return 0

                    # Add average brightness text within each mask
                    for i, mask in enumerate(top_masks):
                        # Randomly select 10 points within the mask
                        selected_coords = mask['point_coords']
                        random_indices = np.random.choice(len(selected_coords), min(10, len(selected_coords)), replace=False)
                        random_points = [selected_coords[idx] for idx in random_indices]

                        # Calculate average brightness (명도)
                        avg_brightness = calculate_average_brightness(random_points)
                        brightness_text = f"{avg_brightness:.2f}"

                        # Get the first point as the base position for text
                        x, y = random_points[0]

                        # Calculate text size
                        text_size, _ = cv2.getTextSize(brightness_text, font, font_scale, font_thickness)
                        text_width, text_height = text_size

                        # Adjust text position to ensure visibility within image boundary
                        if x + text_width > img_width:
                            x = img_width - text_width - 10  # Shift left
                        if y - text_height < 0:
                            y = text_height + 10  # Shift down

                        # Adjust position to avoid overlap
                        x, y = adjust_position(x, y, text_width, text_height)

                        cv2.putText(annotated_image, brightness_text, (int(x), int(y)), font, font_scale, font_color, font_thickness)

                    image_png = Image.fromarray(annotated_image)

                    # Display segmented image with annotated average brightness (명도)
                    st.image(image_png, caption='Segmented Image with Average Brightness (명도)', use_column_width=True)


                    used_positions.clear()
                    brightness_values.clear()
                    masks.clear()


                    #이미지를 gpt 형식으로 변환
                    img_byte_arr = BytesIO()
                    image_png.save(img_byte_arr, format="PNG")
                    img_byte_arr = img_byte_arr.getvalue()
                    img_encoded = base64.b64encode(img_byte_arr).decode('utf-8')




                    #균열 깊이 추정
                    response_img = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                            {"role": "system", "content": """
                            Role: You are a construction inspector tasked with assessing the depth of cracks on a concrete surface based on an image provided.

                            Objective: Your goal is to determine the depth of the cracks in millimeters (mm) using the brightness values given for each area of the image, normalized against the highest brightness value.

                            Instructions:

                            1. Brightness Interpretation and Normalization: 
                             - Normalization: Divide each brightness value by the highest brightness value in the image. This will result in a normalized brightness value between 0 and 1.
                                a) A normalized value of 1 represents the highest brightness (surface level, 0 mm depth).
                                b) Values closer to 0 represent deeper areas.
                             - The normalized brightness value will be used to determine the relative depth.

                            2. Surface Variation Consideration: 
                             - After normalization, slight variations in brightness (normalized brightness difference of 0.1 or less) might represent the same depth. Therefore, small differences should not be interpreted as differences in depth.

                            3. Depth Calculation: 
                             - Normalized Brightness to Depth: Use the normalized brightness value to estimate the depth of the cracks.
                             - If the normalized brightness value is below 0.6 (indicating original brightness below 60 percent of the highest), the crack should be considered deeper than 30 mm.
                             - For other normalized values, depth should be inferred based on the relative difference from the surface (normalized value 1). Larger differences indicate greater depth.

                            4. Output: Provide the estimated depth range for each area in millimeters (mm). Do not include any explanations or details about your reasoning or calculations. Simply provide the final answers. Your response must be in Korean, using the following format to report the depth range for each area:
                            
                             '''
                             - 밝기 nn: n mm ~ n mm(깊은 균열 or 얕은 균열 or 표면) 
                             '''

                            """
                            },  
                            {"role": "user", "content": [
                                {"type": "text", "text": "Evaluate depth of the cracks with a given image."},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/png;base64,{img_encoded}"}
                                }
                            ]}
                        ],
                    )
                    img_exp = response_img.choices[0].message.content.strip().lower()

                    st.write(f"{img_exp}")










            
                    # #all mask data is saved here
                    # areas = []
                    # selected_coords = []
                    # predicted_iou = []
                    # stability_score = []
                    # bbox = []
                    # crop_box = []
                    # segmentation = []

                    # for mask in masks:
                    #     areas.append(mask['area'])
                    #     selected_coords.append(mask['point_coords'])
                    #     predicted_iou.append(mask['predicted_iou'])
                    #     stability_score.append(mask['stability_score'])
                    #     bbox.append(mask['bbox'])
                    #     crop_box.append(mask['crop_box'])
                    #     segmentation.append(mask['segmentation'])





                    # # Display area for each mask
                    # st.write("Mask Boundaries and Areas:")
                    # i = 0
                    # for area in areas:
                    #     st.write(f"Mask {i + 1}:")
                    #     st.write(f" - Area: {area:.2f} pixels")
                    #     i += 1
                                        
                    




















                else:
                    st.error("Failed to generate analysis for the image.")
        else:
            st.warning("Please upload an image or paste from clipboard first.")