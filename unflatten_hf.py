import torch

#from transformers import GPT2ForSequenceClassification as ModelClass
#from transformers import GPT2Config as ModelConfig
from transformers import BertForMaskedLM as ModelClass
from transformers import BertConfig as ModelConfig

from hf_utils import generate_inputs_for_model


config = ModelConfig()
model = ModelClass(config)

example_inputs = generate_inputs_for_model(
    ModelClass, model, ModelClass.__name__, 1, "cpu")

ep = torch.export.export(model, (), example_inputs)

# Unflatten
unflattened = torch.export.unflatten(ep)
unflattened.graph.print_tabular()
print(unflattened)
for name, param in unflattened.named_parameters():
    print(f"{name}: {param.size()}")

