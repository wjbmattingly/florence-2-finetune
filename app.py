import gradio as gr
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
import spaces
from PIL import Image 
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import requests
from io import BytesIO
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import os

subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

model_dir = "medieval-data/florence2-medieval-bbox-zone-detection"

# Load the configuration
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True
        )

TITLE = "# [Florence-2- Medieval Manuscript Layout Parsing Demo](https://huggingface.co/medieval-data/florence2-medieval-bbox-zone-detection)"
DESCRIPTION = "The demo for Florence-2 fine-tuned on CATMuS Segmentation Dataset. This app has two models: one for line detection and one for zone detection."

# Define a color map for different labels
colormap = plt.cm.get_cmap('tab20')

@spaces.GPU
def process_image(image):
    max_size = 1000
    prompt = "<OD>"

    # Calculate the scaling factor
    original_width, original_height = image.size
    scale = min(max_size / original_width, max_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    image = image.resize((new_width, new_height))

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
    
    return result, image

def visualize_bboxes(result, image):
    fig, ax = plt.subplots(1, figsize=(15, 15))
    ax.imshow(image)

    # Create a set of unique labels
    unique_labels = set(result['<OD>']['labels'])

    # Create a dictionary to map labels to colors
    color_dict = {label: colormap(i/len(unique_labels)) for i, label in enumerate(unique_labels)}

    # Add bounding boxes and labels to the plot
    for bbox, label in zip(result['<OD>']['bboxes'], result['<OD>']['labels']):
        x, y, width, height = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=color_dict[label], facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, label, fontsize=12, bbox=dict(facecolor=color_dict[label], alpha=0.5))

    plt.axis('off')
    return fig

def run_example(image):
    if isinstance(image, str):  # If image is a URL
        response = requests.get(image)
        image = Image.open(BytesIO(response.content))
    elif isinstance(image, np.ndarray):  # If image is a numpy array
        image = Image.fromarray(image)
    
    result, processed_image = process_image(image)
    fig = visualize_bboxes(result, processed_image)
    
    # Convert matplotlib figure to image
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    output_image = Image.open(img_buf)
    
    return output_image
css = """
  #output {
    height: 1000px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Florence-2 Image Processing"):
        input_img = gr.Image(label="Input Picture", elem_id="input_img", height=300, width=300)
        submit_btn = gr.Button(value="Submit")
        with gr.Row():
            output_img = gr.Image(label="Output Image with Bounding Boxes")
        gr.Examples(
            examples=[
                ["https://huggingface.co/datasets/CATMuS/medieval-segmentation/resolve/main/data/train/cambridge-corpus-christi-college-ms-111/page-002-of-003.jpg"],
            ],
            inputs=[input_img],
            outputs=[output_img],
            fn=run_example,
            cache_examples=True,
            label='Try the examples below'
        )
        submit_btn.click(run_example, [input_img], [output_img])

demo.launch(debug=True)