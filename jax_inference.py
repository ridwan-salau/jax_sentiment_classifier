from itertools import chain
from typing import Callable
import os
import shutil

import datasets
import evaluate
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from flax.training import train_state
from flax.training.common_utils import (get_metrics, onehot, shard,
                                        shard_prng_key)
from jax import random
from tqdm import tqdm
from transformers import (AutoConfig, AutoTokenizer,
                          FlaxAutoModelForSequenceClassification)
from transformers.models.bert.modeling_flax_bert import FlaxBertForSequenceClassification
from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint

metric = evaluate.load("accuracy")
model_checkpoint = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_checkpoint, num_labels=1)

# model = FlaxAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)

model = FlaxBertForSequenceClassification(config=config)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def pre_process(row):
    processed = tokenizer(row["text"], padding="max_length", max_length=32, truncation=True)
    processed["label"] = row["feeling"]
    return processed

class TrainState(train_state.TrainState):
    eval_function: Callable = flax.struct.field(pytree_node=False)
    # loss_function: Callable = flax.struct.field(pytree_node=False)

@jax.jit    
def eval_fn(logits):
    return (jax.nn.sigmoid(jnp.squeeze(logits)))

state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=optax.adamw(2e-5),
    eval_function=eval_fn,
)

ckpt = {'model': state, 'config': config}
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

state_restored = orbax_checkpointer.restore('tmp/orbax/single_save', item=ckpt)
# print(state_restored)
state = state_restored["model"]

data = datasets.load_dataset("carblacac/twitter-sentiment-analysis", "None")
test_data = data["test"]
test_tokenized = test_data.map(pre_process, batched=True, remove_columns=test_data.column_names,)

def eval_data_loader(dataset, total_batch_size):
    for i in range(len(dataset)//total_batch_size):
        batch = dataset[i * total_batch_size : (i + 1) * total_batch_size]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch
total_batch_size=1024

def eval_step(state, batch):
    logits = state.apply_fn(**batch, params=state.params, train=False)[0]
    return state.eval_function(logits)
parallel_eval_step = jax.pmap(eval_step, axis_name="batch")

state = flax.jax_utils.replicate(state)

with tqdm(total=len(test_tokenized)//(total_batch_size), desc="Evaluating...", leave=False) as progress_bar_eval:
    for i,batch in enumerate(eval_data_loader(test_tokenized, total_batch_size=(total_batch_size))):
        labels = batch.pop("label", None)
        predictions = parallel_eval_step(state, batch)
        metric.add_batch(predictions=jnp.round(predictions.flatten()), references=labels.flatten())
        progress_bar_eval.update(1)
        
    eval_metric = metric.compute()
    eval_score = round(list(eval_metric.values())[0], 3)
    metric_name = list(eval_metric.keys())[0]
    total_batch_size = min(total_batch_size*2,512)

print(f"\nEval {metric_name}: {eval_score}")