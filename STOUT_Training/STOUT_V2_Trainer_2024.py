import tensorflow as tf

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# Built In Imports
from datetime import datetime
from glob import glob
import pickle
import sys
import os
import re
import time

import transformer_model as nmt_model_transformer
import re

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
print("Running on TPU ", cluster_resolver.cluster_spec().as_dict()["worker"])

tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


numbers = re.compile(r"(\d+)")


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


inp_lang = pickle.load(open("tokenizer_SMILES.pkl", "rb"))
targ_lang = pickle.load(open("tokenizer_iUPAC_new.pkl", "rb"))
inp_max_length = 602
targ_max_length = 702

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Imports done", flush=True)

total_data = 102400

EPOCHS = 30
num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1
BUFFER_SIZE = 10000
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
steps_per_epoch = total_data // BATCH_SIZE
num_steps = total_data // BATCH_SIZE

print("Total batches: ", num_steps)

input_vocab_size = len(inp_lang.word_index) + 1
target_vocab_size = len(targ_lang.word_index) + 1

AUTO = tf.data.experimental.AUTOTUNE


def read_tfrecord(example):
    feature = {
        #'image_id': tf.io.FixedLenFeature([], tf.string),
        "input_smiles": tf.io.FixedLenFeature([], tf.string),
        "target_iupac": tf.io.FixedLenFeature([], tf.string),
    }

    # decode the TFRecord
    example = tf.io.parse_single_example(example, feature)

    input_smiles = tf.io.decode_raw(example["input_smiles"], tf.int32)
    target_iupac = tf.io.decode_raw(example["target_iupac"], tf.int32)

    return input_smiles, target_iupac


def get_training_dataset(batch_size=BATCH_SIZE, buffered_size=BUFFER_SIZE):

    options = tf.data.Options()
    filenames = sorted(
        tf.io.gfile.glob("gs://tpu-test-koh/STOUT_V2/Zinc/Training_data/*.tfrecord"),
        key=numericalSort,
    )

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    train_dataset = (
        dataset.with_options(options)
        .map(read_tfrecord, num_parallel_calls=AUTO)
        .shuffle(buffered_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=AUTO)
    )
    return train_dataset


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


with strategy.scope():

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
    )
    learning_rate = CustomSchedule(d_model)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def loss_function(real, pred):
        mask = real != 0
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        pred = tf.argmax(pred, axis=2)
        real = tf.cast(real, pred.dtype)
        match = real == pred

        mask = real != 0

        match = match & mask

        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(match) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy", dtype=tf.float32)
    validation_loss = tf.keras.metrics.Mean(name="validation_loss", dtype=tf.float32)
    validation_accuracy = tf.keras.metrics.Mean(
        name="validation_accuracy", dtype=tf.float32
    )

    # Initialize Transformer
    transformer = nmt_model_transformer.Transformer(
        num_layer,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        inp_max_length,
        targ_max_length,
        rate=dropout_rate,
    )

    # Build the model
    input_example = tf.ones((1, inp_max_length))
    target_example = tf.ones((1, targ_max_length))
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        input_example, target_example
    )

    transformer(
        input_example,
        target_example,
        False,
        enc_padding_mask,
        combined_mask,
        dec_padding_mask,
    )
    print(transformer.summary())

checkpoint_path = "gs://tpu-test-koh/STOUT_V2/Zinc/Checkpoints_adamw/"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=150)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])

per_replica_batch_size = BATCH_SIZE // strategy.num_replicas_in_sync

print("Batch Size", BATCH_SIZE)
print("Per replica", per_replica_batch_size)
train_dataset = strategy.experimental_distribute_dataset(get_training_dataset())

# the loss_plot array will be reset many times
loss_plot = []
train_accuracy_data = []
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function
def train_step(iterator):
    def step_fn(inputs):
        inp, target = inputs
        loss = 0

        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer.apply_gradients(
            zip(clipped_gradients, transformer.trainable_variables)
        )

        train_loss.update_state(loss * strategy.num_replicas_in_sync)
        train_accuracy.update_state(
            accuracy_function(tar_real, predictions) * strategy.num_replicas_in_sync
        )

    strategy.run(step_fn, args=(iterator,))


loss_plot = []
accuracy_plot = []
val_loss_plot = []
val_acc = []
f = open("Training__AdmWcus.txt", "w")
sys.stdout = f

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    batch = 0
    validation_batch = 0
    train_loss.reset_states()
    train_accuracy.reset_states()

    for x in train_dataset:
        img_tensor, target = x
        train_step(x)
        batch += 1

        if batch % 100 == 0:
            print(
                "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()
                ),
                flush=True,
            )

        if batch == num_steps:
            loss_plot.append(train_loss.result().numpy())
            accuracy_plot.append(train_accuracy.result().numpy())
            ckpt_manager.save()

            print(
                "Epoch {} Training_Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1, train_loss.result(), train_accuracy.result()
                ),
                flush=True,
            )
            print(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                "Time taken for 1 epoch {} sec\n".format(time.time() - start),
                flush=True,
            )
            break

    train_loss.reset_states()
    train_accuracy.reset_states()

plt.plot(loss_plot, "-o", label="Training loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(ncol=2, loc="upper right")
plt.gcf().set_size_inches(20, 20)
plt.savefig("Lossplot_SMILES.jpg")
plt.close()

plt.plot(accuracy_plot, "-o", label="Training accuracy")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(ncol=2, loc="lower right")
plt.gcf().set_size_inches(20, 20)
plt.savefig("accuracyplot_SMILES.jpg")
plt.close()

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Network Completed", flush=True)
f.close()
