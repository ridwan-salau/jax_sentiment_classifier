import os
import shutil
from typing import Callable

import datasets
import evaluate
import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from flax.training import orbax_utils, train_state
from flax.training.common_utils import shard
from tqdm import tqdm
from transformers import (AutoConfig, AutoTokenizer,
                          FlaxAutoModelForSequenceClassification)

data_files={"train": "train.csv", "test": "test.csv"}
data = datasets.load_dataset("carblacac/twitter-sentiment-analysis", "None")

train_data = data["train"]
test_data = data["test"]

model_checkpoint = "bert-base-uncased"
num_labels=1
seed = 0
num_epochs=10
lr = 2e-5
per_device_batch_size = 4
total_batch_size = per_device_batch_size * jax.local_device_count()

num_train_steps = len(train_data) // total_batch_size * num_epochs
print("The number of train steps (all the epochs) is", num_train_steps)

lr_sched = optax.cosine_onecycle_schedule(transition_steps=num_train_steps, peak_value=lr, pct_start=0.1, )

metric = evaluate.load("accuracy")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def pre_process(row):
    processed = tokenizer(row["text"], padding="max_length", max_length=32, truncation=True)
    processed["label"] = row["feeling"]
    return processed

train_tokenized = train_data.map(pre_process, batched=True, remove_columns=train_data.column_names, )
test_tokenized = test_data.map(pre_process, batched=True, remove_columns=test_data.column_names,)

config = AutoConfig.from_pretrained(model_checkpoint, num_labels=num_labels)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = FlaxAutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config, seed=seed)


class TrainState(train_state.TrainState):
    eval_function: Callable = flax.struct.field(pytree_node=False)
    loss_function: Callable = flax.struct.field(pytree_node=False)

def adamw(weight_decay):
    return optax.adamw(learning_rate=lr_sched, b1=0.9, b2=0.999, eps=1e-6, weight_decay=weight_decay)

@jax.jit
def loss_fn(logits, labels):
    loss = optax.sigmoid_binary_cross_entropy(jnp.squeeze(logits), jnp.squeeze(labels))
    return jnp.mean(loss)

@jax.jit    
def eval_fn(logits):
    return (jax.nn.sigmoid(jnp.squeeze(logits)))


state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=adamw(1e-2),
    eval_function=eval_fn,
    loss_function=loss_fn,
)


def train_step(state, batch, dropout_rng):
    targets = batch.pop("label")
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_function(params):
        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        loss = state.loss_function(logits, targets)
        return loss

    grad_function = jax.value_and_grad(loss_function)
    loss, grad = grad_function(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    metrics = jax.lax.pmean({"loss": loss, "learning_rate": lr_sched(state.step)}, axis_name="batch")
    return new_state, metrics, new_dropout_rng


parallel_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))

def eval_step(state, batch):
    logits = state.apply_fn(**batch, params=state.params, train=False)[0]
    return state.eval_function(logits)

parallel_eval_step = jax.pmap(eval_step, axis_name="batch")

def train_data_loader(rng, dataset, total_batch_size):
    steps_per_epoch = len(dataset) // total_batch_size 
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[:steps_per_epoch*total_batch_size]
    perms = perms.reshape((steps_per_epoch, total_batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch
        
def eval_data_loader(dataset, total_batch_size):
    for i in range(len(dataset)//total_batch_size):
        batch = dataset[i * total_batch_size : (i + 1) * total_batch_size]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch
        
state = flax.jax_utils.replicate(state)

rng = jax.random.PRNGKey(seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())


for i, epoch in enumerate(tqdm(range(1, num_epochs + 1), desc=f"Epoch...", position=0, leave=True)):
    rng, input_rng = jax.random.split(rng)

    # train
    with tqdm(total=len(test_tokenized)//total_batch_size, desc="Training...", leave=False) as progress_bar_train:
        for i,batch in enumerate(train_data_loader(input_rng, test_tokenized, total_batch_size)):
            state, train_metrics, dropout_rngs = parallel_train_step(state, batch, dropout_rngs)
            progress_bar_train.update(1)
            # if i>100:
            #     break

    # evaluate
    with tqdm(total=len(test_tokenized)//(total_batch_size*2), desc="Evaluating...", leave=False) as progress_bar_eval:
        for i,batch in enumerate(eval_data_loader(test_tokenized, total_batch_size=(total_batch_size*2))):
            labels = batch.pop("label")
            predictions = parallel_eval_step(state, batch)
            # print((predictions.flatten()), labels.flatten(), (jnp.int32(predictions.flatten())==labels.flatten()).mean(), sep="\n") if i%500==0 else None
            metric.add_batch(predictions=jnp.round(predictions.flatten()), references=labels.flatten())
            progress_bar_eval.update(1)
            # if i>10:
            #     break

    eval_metric = metric.compute()

    loss = round(flax.jax_utils.unreplicate(train_metrics)['loss'].item(), 3)
    eval_score = round(list(eval_metric.values())[0], 3)
    metric_name = list(eval_metric.keys())[0]
    total_batch_size = min(total_batch_size*2,512)

    print(f"\n{i+1}/{num_epochs} | Train loss: {loss} | Eval {metric_name}: {eval_score}")
    
ckpt_dir = 'tmp'

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.
state = flax.jax_utils.unreplicate(state)
ckpt = {'model': state, 'config': config}
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save('tmp/orbax/single_save', ckpt, save_args=save_args)