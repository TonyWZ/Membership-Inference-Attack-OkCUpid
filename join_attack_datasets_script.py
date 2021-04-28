# This script takes the prediction vectors generated from different shadow
# models and group them by their true labels to form the dataset for training
# attack models.

import collections
import os

import torch


device = torch.device('cpu')

dataset_dir = 'attack_datasets/target_model_combined/overfit_target_model'

os.mkdir(dataset_dir)

start_dataset_index = 0
end_dataset_index = 20

train_pred_list_by_label = collections.defaultdict(lambda: [])
test_pred_list_by_label = collections.defaultdict(lambda: [])
for dataset_index in range(start_dataset_index, end_dataset_index):
  train_pred = torch.load(
      'target_model_pred/train_pred_{}.pt'.format(dataset_index),
      map_location=device,
  )
  train_labels = torch.load(
      'target_model_pred/train_labels_{}.pt'.format(dataset_index),
      map_location=device,
  )
  for pred, label in zip(train_pred, train_labels):
    train_pred_list_by_label[label.numpy().tolist()].append(pred.unsqueeze(dim=0))

  test_pred = torch.load(
      'target_model_pred/test_pred_{}.pt'.format(dataset_index),
      map_location=device,
  )
  test_labels = torch.load(
      'target_model_pred/test_labels_{}.pt'.format(dataset_index),
      map_location=device,
  )
  for pred, label in zip(test_pred, test_labels):
    test_pred_list_by_label[label.numpy().tolist()].append(pred.unsqueeze(dim=0))

for key, pred_list in train_pred_list_by_label.items():
  joined_pred = torch.cat(pred_list, dim=0)
  torch.save(joined_pred, '{}/class_{:d}_train.pt'.format(dataset_dir, key))

for key, pred_list in test_pred_list_by_label.items():
  joined_pred = torch.cat(pred_list, dim=0)
  torch.save(joined_pred, '{}/class_{:d}_test.pt'.format(dataset_dir, key))
