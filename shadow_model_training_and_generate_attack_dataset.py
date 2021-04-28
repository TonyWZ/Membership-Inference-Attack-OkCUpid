import collections
import math
import random
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class JobPredictionDataset(torch.utils.data.Dataset):

  def __init__(self, df):
    self.df = df

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    label = int(row['job_class'])
    example_tensor = torch.tensor(row[:-1].values.astype(np.float32), dtype=torch.float32)
    return example_tensor, label


class FeedforwardNN(nn.Module):
  def __init__(self, neuron_num_list):
    super(FeedforwardNN, self).__init__()
    layer_list = []
    prev_layer_size = 521
    for neuron_num in neuron_num_list:
      layer_list.append(nn.Linear(prev_layer_size, neuron_num))
      prev_layer_size = neuron_num
    self.layer_list = nn.ModuleList(layer_list)
    self.last_layer = nn.Linear(prev_layer_size, 21)

  def forward(self, x):
    for layer in self.layer_list:
      x = F.relu(layer(x))
    logits = self.last_layer(x)
    return logits


def get_accuracy(model, data_loader, device):
  total_example_num = 0
  total_corr = 0
  for x, y in data_loader:
    pred = model(x.to(device))
    pred_label = torch.argmax(pred, 1)
    num_corr = (pred_label == y.to(device)).sum()
    total_example_num += y.shape[0]
    total_corr += num_corr
  return float(total_corr) / total_example_num


def train_model(
    model,
    train_loader,
    optimizer,
    loss_func,
    epoch_num,
    device,
    validation_loader=None,
    show_steps=True,
    epoch_output_interval=1,
):
  model.to(device)
  val_acc_list = []
  train_acc_list = []
  loss_list = []

  for e in range(epoch_num):
    train_num_corr = 0
    epoch_example_num = 0
    step_count = 0
    cum_loss = 0.
    step_interval_example_num = 0
    for x, y in train_loader:
      step_count += 1
      pred = model(x.to(device))
      pred_label = torch.argmax(pred, 1)
      num_corr = (pred_label == y.to(device)).sum()
      epoch_example_num += y.shape[0]
      train_num_corr += num_corr
      loss = loss_func(pred, y.to(device))
      cum_loss += loss
      step_interval_example_num += y.shape[0]
      loss.backward()
      with torch.no_grad():
        optimizer.step()
      if step_count % 50 == 0:
        avg_loss = cum_loss / step_interval_example_num
        loss_list.append(avg_loss.item())
        if show_steps:
          print('Step {:d} avg loss: {:.4f}'.format(step_count, avg_loss))
        cum_loss = 0.
        step_interval_example_num = 0
    train_acc = float(train_num_corr) / epoch_example_num
    train_acc_list.append(train_acc)
    if validation_loader is not None:
      val_acc = get_accuracy(model, validation_loader, device)
      val_acc_list.append(val_acc)
    if (e + 1) % epoch_output_interval == 0:
      if validation_loader is not None:
        print('Epoch {:d}, train acc: {:.4f}, val acc: {:.4f}'.format(e, train_acc, val_acc))
      else:
        print('Epoch {:d}, train acc: {:.4f}'.format(e, train_acc))

  return model, loss_list, train_acc_list, val_acc_list


def gen_pred_vectors(model, data_loader):
  pred_list = []
  label_list = []
  for x, y in data_loader:
    pred = model(x.to(device))
    pred_list.append(pred)
    label_list.append(y)
  return torch.cat(pred_list, dim=0), torch.cat(label_list, dim=0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
output_dataset_folder = 'overfit_attack_datasets_raw'
output_model_folder = 'overfit_shadow_models'
os.mkdir('./' + output_dataset_folder)
os.mkdir('./' + output_model_folder)

for i in range(18, 25):
  print('Processing attack model {:d}'.format(i))

  input_dataset_index = i

  processed_col_dtypes = collections.defaultdict(lambda: float)
  processed_col_dtypes['job_class'] = int

  job_shadow_train_df = pd.read_csv(
      '../input/shadow-model-datasets/shadow_train_{:d}.csv'.format(input_dataset_index),
      dtype=processed_col_dtypes,
  )

  job_shadow_test_df = pd.read_csv(
      '../input/shadow-model-datasets/shadow_test_{:d}.csv'.format(input_dataset_index),
      dtype=processed_col_dtypes,
  )

  train_set = JobPredictionDataset(job_shadow_train_df)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
  test_set = JobPredictionDataset(job_shadow_test_df)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=True)

  epoch_num = 500
  loss_func = nn.CrossEntropyLoss()
  model = FeedforwardNN((500, 100, 50))
  lr = 5e-6
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model, loss_list, train_acc_list, val_acc_list = train_model(
      model,
      train_loader,
      optimizer,
      loss_func,
      epoch_num,
      device,
      show_steps=False,
      epoch_output_interval=50,
  )

  train_pred, train_labels = gen_pred_vectors(model, train_loader)
  test_pred, test_labels = gen_pred_vectors(model, test_loader)

  torch.save(train_pred, './{}/train_pred_{:d}.pt'.format(output_dataset_folder, i))
  torch.save(train_labels, './{}/train_labels_{:d}.pt'.format(output_dataset_folder, i))
  torch.save(test_pred, './{}/test_pred_{:d}.pt'.format(output_dataset_folder, i))
  torch.save(test_labels, './{}/test_labels_{:d}.pt'.format(output_dataset_folder, i))
  torch.save(model.state_dict(), './{}/500_100_50_shadow_model_{:d}.pt'.format(output_model_folder, i))
