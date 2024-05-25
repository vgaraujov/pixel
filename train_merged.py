import json
import re
import math
from typing import Optional

import torch
from PIL import Image
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm
from transformers import ViTConfig, ViltConfig
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from pixel import (
    AutoConfig,
    AutoModelForSequenceClassification,
    VIPForQuestionAnswering,
    PIXELForSequenceClassification,
    PIXELModel,
    VIPModel,
    Modality,
    PangoCairoTextRenderer,
    PIXELConfig,
    ViTModel,
    PIXELTrainer,
    PIXELTrainingArguments,
    PoolingMode,
    PyGameTextRenderer,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    log_sequence_classification_predictions,
)


filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

# source: https://github.com/allenai/allennlp-models/blob/a36aed540e605c4293c25f73d6674071ca9edfc3/allennlp_models/vision/dataset_readers/vqav2.py#L141
def id_from_filename(filename: str) -> Optional[int]:
    match = filename_re.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(1))

def get_score(count: int) -> float:
    return min(1.0, count / 3)

def load_data(split, config):
    # Opening JSON file    
    f = open(f'VQAv2/v2_OpenEnded_mscoco_{split}2014_questions.json')
    # Return JSON object as dictionary
    data_questions = json.load(f)

    questions = data_questions['questions']

    print(f"Number of {split} questions:", len(questions))

    # root at which all images are stored
    root = f'VQAv2/{split}2014'
    file_names = [f for f in tqdm(listdir(root)) if isfile(join(root, f))]

    filename_to_id = {root + "/" + file: id_from_filename(file) for file in file_names}
    id_to_filename = {v:k for k,v in filename_to_id.items()}

    # Read annotations
    f = open(f'VQAv2/v2_mscoco_{split}2014_annotations.json')
    
    # Return JSON object as dictionary
    data_annotations = json.load(f)

    annotations = data_annotations['annotations']

    print(f"Number of {split} annotations:", len(annotations))

    for annotation in tqdm(annotations):
        answers = annotation['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer["answer"]
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        labels = []
        scores = []
        for answer in answer_count:
            if answer not in list(config.label2id.keys()):
                continue
            labels.append(config.label2id[answer])
            score = get_score(answer_count[answer])
            scores.append(score)
        annotation['labels'] = labels
        annotation['scores'] = scores

    return questions, annotations, id_to_filename

def resize_model_embeddings(model, max_seq_length) -> None:
    """
    Checks whether position embeddings need to be resized. If the specified max_seq_length is longer than
    the model's number of patches per sequence, the position embeddings will be interpolated.
    If max_seq_length is shorter, the position embeddings will be truncated

    Args:
        model (`ViTForImageClassification`):
            The model for which position embeddings may be resized.
        max_seq_length (`int`):
            The maximum sequence length that determines the number of patches (excluding CLS patch) in the
            model.
    """

    patch_size = model.config.patch_size
    if isinstance(model.config.txt_size, tuple) or isinstance(model.config.txt_size, list):
        old_height, old_width = model.config.txt_size
    else:
        old_height, old_width = (model.config.txt_size, model.config.txt_size)

    # ppr means patches per row (image is patchified into grid of [ppr * ppr])
    old_ppr = math.sqrt(old_height * old_width) // patch_size
    new_ppr = math.sqrt(max_seq_length)

    if old_ppr < new_ppr:
        # Interpolate position embeddings
        print(f"Interpolating position embeddings to {max_seq_length}")
        model.config.interpolate_pos_encoding = True
    elif old_ppr > new_ppr:
        print(f"Truncating position embeddings to {max_seq_length}")
        # Truncate position embeddings
        old_pos_embeds = model.vit.embeddings.position_txt_embeddings[:, : max_seq_length + 1, :]
        model.vit.embeddings.position_txt_embeddings.data = old_pos_embeds.clone()
        # Update image_size
        new_height = int(new_ppr * patch_size) if old_height == old_width else int(patch_size)
        new_width = int(new_ppr * patch_size) if old_height == old_width else int(patch_size * new_ppr ** 2)
        model.config.txt_size = [new_height, new_width]
        model.txt_size = [new_height, new_width]
        model.vit.embeddings.patch_txt_embeddings.image_size = [new_height, new_width]

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, questions, annotations, id_to_filename, processor_img, processor_txt):
        self.questions = questions
        self.annotations = annotations
        self.processor_img = processor_img
        self.processor_txt = processor_txt
        self.id_to_filename = id_to_filename

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        encoding = {}
        # get image + text
        annotation = self.annotations[idx]
        questions = self.questions[idx]
        image = Image.open(self.id_to_filename[annotation['image_id']])
        text = questions['question']

        result_img = self.processor_img(image.convert("RGB"), return_tensors="pt")
        # remove batch dimension
        for k,v in result_img.items():
          result_img[k] = v.squeeze()

        format_fn = glue_strip_spaces
        result_txt = self.processor_txt(text=format_fn(text))
        transforms = get_transforms(
            do_resize=True,
            size=(self.processor_txt.pixels_per_patch, self.processor_txt.pixels_per_patch * self.processor_txt.max_seq_length),
        )

        encoding["pixel_txt_values"] = transforms(Image.fromarray(result_txt.pixel_values))
        encoding["attention_txt_mask"] = get_attention_mask(result_txt.num_text_patches, seq_length=self.processor_txt.max_seq_length)
        encoding["pixel_img_values"] = result_img.pixel_values     
        num_patches = (224 // 16) * (224 // 16) # hardcoded for now
        encoding["attention_img_mask"] = torch.ones(num_patches)

        # add labels
        labels = annotation['labels']
        scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))
        for label, score in zip(labels, scores):
              targets[label] = score
        encoding["labels"] = targets

        return encoding


# Variables
max_seq_length = 196
epochs = 2
lr = 5e-5
batch_size = 4

print("Configs")
tmp_config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa",cache_dir="/tmp/huggingface/pixel")
config = ViTConfig.from_pretrained("facebook/vit-mae-base",cache_dir="/tmp/huggingface/pixel")
config.num_labels = tmp_config.num_labels
config.label2id = tmp_config.label2id
config.id2label = tmp_config.id2label

processor_img = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base",cache_dir="/tmp/huggingface/pixel")

print("Render")
renderer_cls = PangoCairoTextRenderer
processor_txt = renderer_cls.from_pretrained(
    "Team-PIXEL/pixel-base",
    rgb=False,
    max_seq_length=max_seq_length,
    fallback_fonts_dir="fallback_fonts",cache_dir="/tmp/huggingface/pixel/datasets"
)

# load data
print("Loading training data .......... \n")
questions, annotations, id_to_filename = load_data("train", config)

questions=questions[:1000]
annotations=annotations[:1000]

train_dataset = VQADataset(questions=questions,
                     annotations=annotations,
                     id_to_filename=id_to_filename,      
                     processor_img=processor_img,
                     processor_txt=processor_txt)

questions, annotations, id_to_filename = load_data("val", config)

print("Loading validation data ........ \n")
questions=questions[:1000]
annotations=annotations[:1000]

valid_dataset = VQADataset(questions=questions,
                     annotations=annotations,
                     id_to_filename=id_to_filename,
                     processor_img=processor_img,
                     processor_txt=processor_txt)

config.img_size = 224
config.txt_size = [16, 8464]
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

print("Loading Model ........ \n")
model = VIPForQuestionAnswering.from_pretrained("Checkpoints/vip", 
                                                       config=config,
                                                       pooling_mode=PoolingMode.from_string("mean"),
                                                       add_layer_norm=True,cache_dir="/tmp/huggingface/pixel")
model.to(device)

resize_model_embeddings(model, max_seq_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("Training ......... \n")
for epoch in range(epochs): #loop over the dataset multiple times
    model.train()
    print(f"Epoch: {epoch}")
    for batch in tqdm(train_dataloader):
        #get the inputs;
        batch={k:v.to(device) for k,v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss

        # print("Loss:", loss.item())
        loss.backward()
        optimizer.step()

    model.eval()

    print("Validation step")

    for step,batch in enumerate(tqdm(valid_dataloader)):
        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = outputs.logits#.argmax(dim=-1)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions)).cpu().numpy()

        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= 0.5)] = 1

        # finally, compute metrics

        y_true = (batch['labels'] > 0).cpu().numpy()
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        
    print(f"epoch {epoch}:", metrics)