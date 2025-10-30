import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformer_model import Transformer

# ========== Load and Preprocess Dataset ==========
wiki_data = tfds.load("samsum", split="train[:100%]", as_supervised=True)

input_texts = []
target_texts = []
for input_text, target_text in tfds.as_numpy(wiki_data):
    input_texts.append(input_text.decode("utf-8"))
    target_texts.append(f"<start> {target_text.decode('utf-8')} <end>")

tokenizer = Tokenizer(filters="", oov_token="<unk>")
tokenizer.fit_on_texts(input_texts + target_texts)

assert "<start>" in tokenizer.word_index and "<end>" in tokenizer.word_index

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

input_sequences = pad_sequences(input_sequences, padding="post", truncating="post")
target_sequences = pad_sequences(target_sequences, padding="post", truncating="post")

input_vocab_size = len(tokenizer.word_index) + 1
target_vocab_size = input_vocab_size

BUFFER_SIZE = len(input_sequences)
BATCH_SIZE = 4

dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ========== Training Setup ==========
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

def create_padding_mask(seq):
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    return tf.reduce_mean(loss_ * mask)

d_model = 64
num_layers = 2
num_heads = 2
dff = 128

max_input_len = input_sequences.shape[1]
max_target_len = target_sequences.shape[1]

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          max_input_len, max_target_len)

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask = create_padding_mask(inp)
    dec_target_padding_mask = create_padding_mask(tar_inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    with tf.GradientTape() as tape:
        predictions = transformer(inp, tar_inp, training=True,
                                  enc_padding_mask=enc_padding_mask,
                                  look_ahead_mask=combined_mask,
                                  dec_padding_mask=enc_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    return loss

# ========== Training Loop ==========
for epoch in range(10):
    total_loss = 0
    for batch, (inp, tar) in enumerate(dataset):
        batch_loss = train_step(inp, tar)
        total_loss += batch_loss
    print(f"Epoch {epoch + 1}: Loss {total_loss.numpy() / (batch + 1):.4f}")

# ========== Inference Utilities ==========
def sample_from_logits(logits, top_k=5, temperature=1.0):
    logits = logits / temperature
    values, _ = tf.math.top_k(logits, k=top_k)
    min_values = values[-1]
    logits = tf.where(logits < min_values, tf.fill(tf.shape(logits), -1e9), logits)
    probs = tf.nn.softmax(logits).numpy()
    return np.random.choice(len(probs), p=probs)

def evaluate(sentence, max_length=40):
    seq = tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=input_sequences.shape[1], padding="post")

    start_token = tokenizer.word_index["<start>"]
    end_token = tokenizer.word_index["<end>"]
    output = [start_token]

    for _ in range(max_length):
        out_tensor = tf.convert_to_tensor([output])
        enc_mask = create_padding_mask(seq)
        dec_padding_mask = create_padding_mask(out_tensor)
        look_ahead_mask = create_look_ahead_mask(len(output))
        combined_mask = tf.maximum(dec_padding_mask, look_ahead_mask)

        predictions = transformer(seq, out_tensor, training=False,
                                  enc_padding_mask=enc_mask,
                                  look_ahead_mask=combined_mask,
                                  dec_padding_mask=enc_mask)

        logits = predictions[:, -1, :].numpy()[0]
        pred_id = sample_from_logits(logits, top_k=5, temperature=0.9)

        if pred_id == end_token:
            break
        if len(output) > 1 and pred_id == output[-1] == output[-2]:
            continue

        output.append(pred_id)

    decoded = [tokenizer.index_word.get(i, "<unk>") for i in output[1:]]
    return " ".join(decoded)

# ========== Test ==========
sample_input = input_texts[0]
print("\nSample Chat Input:\n", sample_input)
print("\nGround Truth Summary:\n", target_texts[0])
print("\nModel-generated Summary:\n", evaluate(sample_input))

# ===========================
# SUMMARY OF THIS PROJECT
# ===========================
# In this project, we implemented a Transformer-based text summarization model using TensorFlow on the SAMSum dialogue dataset.
# We:
# - Loaded and preprocessed the full dataset using TensorFlow Datasets.
# - Built input/output sequences using a custom tokenizer with special <start> and <end> tokens.
# - Designed a basic Transformer model with 2 encoder-decoder layers, multi-head attention, and positional encoding.
# - Trained the model for 10 epochs with decreasing loss values, confirming learning progress.
# - Added top-k sampling during inference to improve output diversity and avoid repetition.
#
# What we learned:
# - Transformer models can be trained from scratch to summarize dialogues using limited resources.
# - Proper masking (padding and look-ahead) is critical for accurate sequence learning.
# - Generating good summaries depends not just on architecture but also on dataset size, tokenization quality, and training duration.
# - Even a small model trained on the full dataset can produce summaries with reasonable relevance.
#
# Next steps could include adding checkpointing, validation metrics (e.g., ROUGE), increasing model capacity, and integrating a pre-trained tokenizer for better generalization.
