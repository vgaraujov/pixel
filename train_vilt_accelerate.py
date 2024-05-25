import json
import re
import math
from typing import Optional
#import logging
#logging.basicConfig(level=logging.DEBUG)
import torch
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm
from transformers import ViTConfig, ViltConfig, ViltProcessor, ViltForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from accelerate import Accelerator


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
    f = open(f'/home/david.mogrovejo/PVLM/VQAv2/v2_OpenEnded_mscoco_{split}2014_questions.json')
    # Return JSON object as dictionary
    data_questions = json.load(f)

    questions = data_questions['questions']

    print(f"\n Number of {split} questions:", len(questions)) if accelerator.is_main_process else None

    # root at which all images are stored
    root = f'/home/david.mogrovejo/PVLM/VQAv2/{split}2014'
    file_names = [f for f in listdir(root) if isfile(join(root, f))]

    filename_to_id = {root + "/" + file: id_from_filename(file) for file in file_names}
    id_to_filename = {v:k for k,v in filename_to_id.items()}

    # Read annotations
    f = open(f'/home/david.mogrovejo/PVLM/VQAv2/v2_mscoco_{split}2014_annotations.json')
    
    # Return JSON object as dictionary
    data_annotations = json.load(f)

    annotations = data_annotations['annotations']

    print(f"\n Number of {split} annotations:", len(annotations)) if accelerator.is_main_process else None

    for annotation in annotations:
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
        image = Image.open(self.id_to_filename[annotation['image_id']]).convert('RGB')
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
epochs = 10
lr = 5e-5
batch_size = 64


accelerator = Accelerator()
device = accelerator.device

print("Defining Configs and Processor ......\n") if accelerator.is_main_process else None

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa",cache_dir="/tmp/huggingface/pixel")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm",cache_dir="/tmp/huggingface/pixel")


print("\n Loading training data ...... \n") if accelerator.is_main_process else None
# load data
questions, annotations, id_to_filename = load_data("train", config)
train_dataset = VQADataset(questions=questions,
                     annotations=annotations,
                     id_to_filename=id_to_filename,
                     processor=processor)


print("\n Loading evaluation data..... \n") if accelerator.is_main_process else None

questions, annotations, id_to_filename = load_data("val", config)
valid_dataset = VQADataset(questions=questions,
                     annotations=annotations,
                     id_to_filename=id_to_filename,
                     processor=processor)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n Loading Model....... \n") if accelerator.is_main_process else None

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 id2label=config.id2label,
                                                 label2id=config.label2id,cache_dir="/tmp/huggingface/pixel")
model.to(device)


print("Loading Dataloaders ..... \n") if accelerator.is_main_process else None
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
valid_dataloader = accelerator.prepare(valid_dataloader)



print("\n Training Started ...... \n") if accelerator.is_main_process else None
for epoch in range(epochs):
    model.train()
    print(f"\n Epoch: {epoch+1}") if accelerator.is_main_process else None
    for step,batch in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process,miniters=100,maxinterval=float("inf"))):

        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        # print("Loss:", loss.item())
        #loss.backward()
        accelerator.backward(loss)
        optimizer.step()

    model.eval()

    if (epoch+1)%2==0 or epochs==epoch+1:
        print("\n Evaluation -------------------------------- \n") if accelerator.is_main_process else None
        predictions,labels=[],[]
        for step, batch in enumerate(tqdm(valid_dataloader,disable=not accelerator.is_main_process,miniters=100,maxinterval=float("inf"))):

            # get the inputs;
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            pred = outputs.logits#.argmax(dim=-1)
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.Tensor(pred))
            #probs = sigmoid(torch.Tensor(pred)).cpu().numpy()

            # next, use threshold to turn them into integer predictions
            y_pred = torch.zeros(probs.shape)
            y_pred = torch.where(probs>=0.5,1.0,0.0).to(device)
            #y_pred = np.zeros(probs.shape)
            #y_pred[np.where(probs >= 0.5)] = 1

            # finally, compute metrics
            y_true = torch.where(batch['labels']>0,True,False).to(device)
            #y_true = (batch['labels'] > 0).cpu()

            predictions.append(accelerator.gather(y_pred))
            labels.append(accelerator.gather(y_true))

        if accelerator.is_main_process:

            predictions=torch.cat(predictions,0)
            labels=torch.cat(labels,0)

            predictions = predictions[:len(valid_dataloader.dataset)].cpu()
            labels = labels[:len(valid_dataloader.dataset)].cpu()

            f1_micro_average = f1_score(y_true=labels, y_pred=predictions, average='micro')
            roc_auc = roc_auc_score(labels, predictions, average = 'micro')
            accuracy = accuracy_score(labels, predictions)
            # return as dictionary
            metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
                
            print(f"\n Evaluation in Epoch {epoch+1}:", metrics)
            print("\n --------------------------------------------")
