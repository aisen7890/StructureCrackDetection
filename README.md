# CrackSAM: Concrete Crack Analysis System

A comprehensive system for detecting, measuring, and analyzing cracks in concrete surfaces using computer vision and AI.

## Features

- Initial crack detection using CNN
- Precise segmentation using SAM2 (Segment Anything Model 2)
- Crack measurements (length, thickness)
- Brightness analysis for crack and surface
- Depth estimation using GPT-4V
- Visual annotation of results

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CrackSAM.git
cd CrackSAM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model checkpoints:
- Place SAM2 checkpoint at `checkpoints/sam2_hiera_large.pt`
- Place CNN checkpoint at `checkpoints/model_epoch_30.pth`

4. Set up your OpenAI API key:
- Replace the API key in `start.py` with your own

## Directory Structure

```
CrackSAM/
├── checkpoints/           # Model checkpoints
│   ├── sam2_hiera_large.pt
│   └── model_epoch_30.pth
├── image/                 # Input images
├── output_image/          # Output results
├── start.py              # Main script
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Usage

1. Place your concrete surface images in the `image/` directory
2. Run the main script:
```bash
python start.py
```

3. Results will be saved in `output_image/`:
- `output_segmented_[filename]_1.png`: Image with brightness values
- `output_segmented_[filename]_2.png`: Image with measurements
- `output_segmented_[filename]_3.txt`: Detailed measurements and depth estimation

## Output Format

The system generates three types of outputs for each image:

1. **Brightness Analysis Image**:
   - Red overlay showing detected crack
   - Crack brightness value
   - Surface brightness value

2. **Measurement Image**:
   - Blue line showing crack length
   - Orange line showing crack thickness
   - Annotated measurements

3. **Text File**:
   - Surface brightness
   - Crack brightness
   - Normalized brightness values
   - Estimated depth
   - Length and thickness measurements

## Models Used

1. **CNN Model**:
   - Architecture: Encoder-decoder with skip connections
   - Purpose: Initial crack detection
   - Input: 300x300 RGB image
   - Output: Binary mask

2. **SAM2 Model**:
   - Purpose: Precise crack segmentation
   - Uses three strategic points for better accuracy
   - Generates detailed mask of crack area

3. **GPT-4V**:
   - Purpose: Depth estimation
   - Uses normalized brightness values
   - Analyzes three images (original, mask, annotated)

## Notes

- The system requires at least 5% of the image area to be detected as a crack for processing
- For best results, use high-resolution images with good lighting
- Depth estimation is based on brightness analysis and may vary with lighting conditions

## License

[Your chosen license]

## Acknowledgments

- SAM2 model by Meta AI
- OpenAI GPT-4V 