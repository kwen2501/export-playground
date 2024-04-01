# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ python unflatten_bert.py

import torch

from transformers import BertForMaskedLM, BertConfig

from hf_utils import generate_inputs_for_model


# Model configs
model_class = BertForMaskedLM
model_name = "BertForMaskedLM"
config = BertConfig()
device = torch.device("cuda:0")

# Create model
bert = model_class(config)
bert.to(device)
bert.eval()
print(bert)

# Input configs
batch_size = 1
example_inputs = generate_inputs_for_model(
    model_class, bert, model_name, batch_size, device)

# Export
exported = torch.export.export(bert, (), kwargs=example_inputs)

# Unflatten
unflattened = torch.export.unflatten(exported)
out_unflat = unflattened(**example_inputs)
print(f"Unflatten run: {out_unflat}")
