
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW, get_scheduler
from PIL import Image
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import numpy as np
import cv2
from peft import LoraConfig, get_peft_model

from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import requests
import matplotlib.pyplot as plt

import os 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from transformers import AdamW, get_scheduler

from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


# Constants
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
NUM_WORKERS = 0
EPOCHS = 10
LEARNING_RATE = 5e-6
OUTPUT_DIR = "./model_checkpoints"
CHECKPOINT = "microsoft/Florence-2-base-ft"
# CHECKPOINT = "florence2-medieval-region-line"
CATEGORY_MAPPING = {
    # 'DefaultLine': 0,
    # 'InterlinearLine': 1,
    'MainZone': 2,
    # 'HeadingLine': 3,
    'MarginTextZone': 4,
    'DropCapitalZone': 5,
    'NumberingZone': 6,
    # 'TironianSignLine': 7,
    # 'DropCapitalLine': 8,
    'RunningTitleZone': 9,
    'GraphicZone': 10,
    # 'DigitizationArtefactZone': 11,
    'QuireMarksZone': 12,
    'StampZone': 13,
    'DamageZone': 14,
    'MusicZone': 15,
    # 'MusicLine': 16,
    'TitlePageZone': 17,
    'SealZone': 18
}

CATEGORY_MAPPING = {
    'DefaultLine': 0,
    'InterlinearLine': 1,
    # 'MainZone': 2,
    'HeadingLine': 3,
    # 'MarginTextZone': 4,
    # 'DropCapitalZone': 5,
    # 'NumberingZone': 6,
    # 'TironianSignLine': 7,
    # 'DropCapitalLine': 8,
    # 'RunningTitleZone': 9,
    # 'GraphicZone': 10,
    # 'DigitizationArtefactZone': 11,
    # 'QuireMarksZone': 12,
    # 'StampZone': 13,
    # 'DamageZone': 14,
    # 'MusicZone': 15,
    # 'MusicLine': 16,
    # 'TitlePageZone': 17,
    # 'SealZone': 18
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT,
        trust_remote_code=True,
        revision='refs/pr/6'
    ).to(device) 
    processor = AutoProcessor.from_pretrained(CHECKPOINT, 
        trust_remote_code=True, revision='refs/pr/6')

    for param in model.vision_tower.parameters():
        param.is_trainable = False


class MedievalSegmentationDataset(Dataset):
    def __init__(self, split='train'):
        try:
            # self.data = load_dataset("CATMuS/medieval-segmentation", split=split).select(range(700))
            self.data = load_dataset("CATMuS/medieval-segmentation", split=split)
        except Exception:
            self.data = load_dataset("CATMuS/medieval-segmentation", split=split)
        self.max_size = 500
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image'].convert("RGB")
        objects = item['objects']
        
        # Resize image
        original_width, original_height = image.size
        scale = min(self.max_size / original_width, self.max_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        image = transforms.Resize((new_height, new_width))(image) 
        
        prefix = f"<OD>"
        suffix = self.create_suffix(objects, scale, original_width, original_height)
        return prefix, suffix, image
    
    def create_suffix(self, objects, scale, original_width, original_height):
        suffix = ""

        for category, bbox in zip(objects["category"], objects["bbox"]):
            if category in CATEGORY_MAPPING:
                # Adjust bbox coordinates
                x1 = int(bbox[0] * scale)
                y1 = int(bbox[1] * scale)
                x2 = int(bbox[2] * scale)
                y2 = int(bbox[3] * scale)
                
                # Convert to relative coordinates (0-1000 range)
                x1_rel = int(x1 / (original_width * scale) * 1000)
                y1_rel = int(y1 / (original_height * scale) * 1000)
                x2_rel = int(x2 / (original_width * scale) * 1000)
                y2_rel = int(y2 / (original_height * scale) * 1000)
                
                bbox_str = f"<loc_{x1_rel}><loc_{y1_rel}><loc_{x2_rel}><loc_{y2_rel}>"
                suffix += f"{category}{bbox_str}"
        
        return suffix.strip()
    


def collate_fn(batch): 
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers 

train_dataset = MedievalSegmentationDataset(split='train')
val_dataset = MedievalSegmentationDataset(split='validation')
batch_size = 1
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          collate_fn=collate_fn, num_workers=num_workers)

epochs = 10
optimizer = AdamW(model.parameters(), lr=1e-6)
num_training_steps = epochs * len(train_loader)

lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
                              num_warmup_steps=0, num_training_steps=num_training_steps,)
output_dir = "model_checkpoints"
for epoch in range(epochs): 
    model.train() 
    train_loss = 0
    i = -1
    for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
        i += 1
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"] 
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
            inputs, answers = batch
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    output_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    os.makedirs(output_dir, exist_ok=True)
    # model.save_pretrained(output_dir)
    # processor.save_pretrained(output_dir)
    print(val_loss / len(val_loader))
final_output = "florence2-medieval-lines-all"
model.save_pretrained(final_output)
processor.save_pretrained(final_output)