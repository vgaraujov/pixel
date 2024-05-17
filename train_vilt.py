import json
import re
import math
from typing import Optional

import torch
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm
from transformers import ViTConfig, ViltConfig, ViltProcessor
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


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

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, questions, annotations, id_to_filename, processor):
        self.questions = questions
        self.annotations = annotations
        self.processor = processor
        self.id_to_filename = id_to_filename

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        questions = self.questions[idx]
        image = Image.open(self.id_to_filename[annotation['image_id']])
        text = questions['question']

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
          encoding[k] = v.squeeze()
        # add labels
        labels = annotation['labels']
        scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))
        for label, score in zip(labels, scores):
              targets[label] = score
        encoding["labels"] = targets

        return encoding

def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  pixel_values = [item['pixel_values'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  token_type_ids = [item['token_type_ids'] for item in batch]
  labels = [item['labels'] for item in batch]

  # create padded pixel values and corresponding pixel mask
  encoding = processor.current_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['attention_mask'] = torch.stack(attention_mask)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = torch.stack(labels)

  return batch


# Variables
max_seq_length = 196
epochs = 25
lr = 5e-5
batch_size = 4

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# load data
questions, annotations, id_to_filename = load_data("val", config)
train_dataset = VQADataset(questions=questions,
                     annotations=annotations,
                     id_to_filename=id_to_filename,      
                     processor=processor)

questions, annotations, id_to_filename = load_data("val", config)
valid_dataset = VQADataset(questions=questions,
                     annotations=annotations,
                     id_to_filename=id_to_filename,
                     processor=processor)

device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 id2label=config.id2label,
                                                 label2id=config.label2id)
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):  # loop over the dataset multiple times
  model.train()
  print(f"Epoch: {epoch}")
  for batch in tqdm(train_dataloader):
      # get the inputs;
      batch = {k:v.to(device) for k,v in batch.items()}

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      # print("Loss:", loss.item())
      loss.backward()
      optimizer.step()

  model.eval()
  for step, batch in enumerate(tqdm(valid_dataloader)):
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
