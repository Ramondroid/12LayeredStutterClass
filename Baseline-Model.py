import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import Input, Conv1D, Dense, GlobalAveragePooling1D, Add, LayerNormalization, \
    MultiHeadAttention, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# Directory where your dataset is stored
data_directory = "C:/Users/sampl/PycharmProjects/pythonProject/DataSets/12K-Sorted/"
sample_rate = 16000  # Required sample rate (16kHz)
duration = 3  # Desired duration in seconds
n_mels = 128  # Number of Mel bands
label_map = {
    'block': 0,
    'Interjection': 1,
    'NoStutter': 2,
    'prolongation': 3,
    'Sound Repetition': 4,
    'wordrep': 5
}


# Function to check duration and sample rate
def check_audio_properties(file_path, expected_sr, expected_duration):
    audio, sr = librosa.load(file_path, sr=None)  # Load without resampling first to check SR
    actual_duration = librosa.get_duration(y=audio, sr=sr)

    # Check if the sample rate is correct
    if sr != expected_sr:
        print(f"Resampling {file_path} from {sr}Hz to {expected_sr}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=expected_sr)

    # Check if the duration is correct and adjust if necessary
    expected_length = expected_sr * expected_duration
    if len(audio) < expected_length:
        print(f"Padding {file_path} from {len(audio)} samples to {expected_length} samples")
        audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
    elif len(audio) > expected_length:
        print(f"Trimming {file_path} from {len(audio)} samples to {expected_length} samples")
        audio = audio[:expected_length]

    return audio


# Modify the audio processing function to check properties
def process_audio(file_path, sample_rate, duration, n_mels, max_length):
    # Ensure audio is 16kHz and exactly 3 seconds
    audio = check_audio_properties(file_path, sample_rate, duration)

    # Generate log-mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Adjust the shape to the maximum length (padding/trimming)
    if log_mel_spectrogram.shape[1] > max_length:
        log_mel_spectrogram = log_mel_spectrogram[:, :max_length]
    else:
        pad_length = max_length - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, pad_length)))

    return log_mel_spectrogram


# Load audio files and process them, keeping track of successful loads
def load_audio_files(data_directory, label_map, sample_rate, duration, n_mels):
    X = []
    y = []
    total_files = 0
    successfully_loaded = 0

    for label, idx in label_map.items():
        folder = os.path.join(data_directory, label)
        for file_name in os.listdir(folder):
            total_files += 1
            file_path = os.path.join(folder, file_name)

            try:
                # Process each file and append to the dataset
                log_mel_spectrogram = process_audio(file_path, sample_rate, duration, n_mels, max_length=100)
                X.append(log_mel_spectrogram)
                y.append(idx)
                successfully_loaded += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"Total files found: {total_files}")
    print(f"Successfully loaded: {successfully_loaded}")
    print(f"Failed to load: {total_files - successfully_loaded}")

    return np.array(X), np.array(y)


# Load and process dataset
X, y = load_audio_files(data_directory, label_map, sample_rate, duration, n_mels)

# Add channel dimension to X (required for Conv1D input)
X = X[..., np.newaxis]

# Split dataset into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert labels to categorical (one-hot encoding)
from tensorflow.keras.utils import to_categorical

y_train_categorical = to_categorical(y_train, num_classes=len(label_map))
y_val_categorical = to_categorical(y_val, num_classes=len(label_map))


# Balanced Batch Generator class
class BalancedBatchGenerator(Sequence):
    def __init__(self, X, y, batch_size, num_classes):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_classes = num_classes
        # Dictionary of indices per class
        self.indices_per_class = {class_idx: np.where(y == class_idx)[0] for class_idx in range(num_classes)}

    def __len__(self):
        # Calculate number of batches per epoch
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = []
        samples_per_class = max(1, self.batch_size // self.num_classes)

        for class_idx in range(self.num_classes):
            class_indices = self.indices_per_class[class_idx]

            # Skip class if there are no samples for that class
            if len(class_indices) == 0:
                continue

            # Select samples with replacement if not enough samples in the class
            selected_indices = np.random.choice(class_indices, samples_per_class,
                                                replace=len(class_indices) < samples_per_class)
            batch_indices.extend(selected_indices)

        # Shuffle the selected batch indices
        np.random.shuffle(batch_indices)

        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        return X_batch, y_batch

    def on_epoch_end(self):
        pass




# Custom GELU activation function
def gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))


# Positional encoding
def positional_encoding(maxlen, d_model):
    positions = np.arange(maxlen)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding = np.zeros((maxlen, d_model))
    pos_encoding[:, 0::2] = np.sin(positions * div_term)
    pos_encoding[:, 1::2] = np.cos(positions * div_term)
    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)


# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Layer normalization and self-attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="gelu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])



# CNN + Transformer based architecture
def build_model(input_shape, num_classes, transformer_layers=6, head_size=512, num_heads=16, ff_dim=512, dropout=0.01):
    inputs = Input(shape=input_shape)

    # Step 1: 2x Conv1D
    x = Conv1D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = Conv1D(64, 3, padding="same", activation="relu", name="conv2")(x)

    # Step 2: Positional Encoding
    maxlen = input_shape[0]
    d_model = x.shape[-1]
    pos_encoding = positional_encoding(maxlen, d_model)
    pos_encoding = tf.reshape(pos_encoding, (-1, maxlen, d_model))  # Reshape pos_encoding to match x shape
    x = Add(name="pos_encoding")([x, pos_encoding])

    # Step 3: Transformer Encoder Blocks
    for i in range(6):
        x = transformer_encoder(x, head_size=512, num_heads=16, ff_dim=512, dropout=0.01)

    # Step 4: Additional Conv1D layers
    x = Conv1D(32, 3, padding="same", activation="relu", name="conv3")(x)
    x = Conv1D(64, 3, padding="same", activation="relu", name="conv4")(x)
    x = Conv1D(128, 3, padding="same", activation="relu", name="conv5")(x)
    x = Conv1D(256, 3, padding="same", activation="relu", name="conv6")(x)

    # Step 5: Global Average Pooling + Projection
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu", name="dense_proj")(x)
    x = Dropout(0.1)(x)

    # Step 6: Output Layer (for classification into stutter types)
    outputs = Dense(num_classes, activation="softmax", name="output_layer")(x)

    # Model
    model = Model(inputs, outputs)
    return model











# Define batch size and number of classes (based on your label map)
batch_size = 32
num_classes = len(label_map)

# Create an instance of the balanced batch generator
train_generator = BalancedBatchGenerator(X_train, y_train_categorical, batch_size=batch_size, num_classes=num_classes)
val_generator = BalancedBatchGenerator(X_val, y_val_categorical, batch_size=batch_size, num_classes=num_classes)

# Build the model
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape, num_classes)

# Step 1: Freeze the Conv1D and Transformer Encoder layers
for layer in model.layers:
    if "conv" in layer.name or "multi_head_attention" in layer.name:
        layer.trainable = False

# Step 2: Compile the model (initial training with frozen layers)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy',
              metrics=['accuracy'])




# Step 3: Train the model with frozen layers
initial_epochs = 50  # Number of epochs to train with frozen layers
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

# Create a class weight dictionary
class_weights_dict = {
    0: 4.0,  # block
    1: 2.5,  # Interjection
    2: 1.0,  # NoStutter
    3: 4.0,  # prolongation
    4: 3.0,  # Sound Repetition
    5: 3.0   # wordrep
}


# Pass the class weights to the model
model.fit(train_generator,
          validation_data=val_generator,
          epochs=initial_epochs,
          class_weight=class_weights_dict,
          callbacks=[early_stopping_callback, reduce_lr_callback])

# Step 4: Unfreeze the layers and continue training
for layer in model.layers:
    layer.trainable = True  # Unfreeze all layers

# Step 5: Compile the model again (with all layers trainable)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model with unfrozen layers

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

# Create a class weight dictionary
class_weights_dict = {
    0: 4.0,  # block
    1: 2.5,  # Interjection
    2: 1.0,  # NoStutter
    3: 4.0,  # prolongation
    4: 3.0,  # Sound Repetition
    5: 3.0   # wordrep
}

print(class_weights)

# Pass the class weights to the model
model.fit(train_generator,
          validation_data=val_generator,
          epochs=initial_epochs,
          class_weight=class_weights_dict,
          callbacks=[early_stopping_callback, reduce_lr_callback])
# Save the model after training
model_save_path = 'Baseline_model_freeze_unfreeze.keras'
model.save(model_save_path)




val_loss, val_acc = model.evaluate(X_val, y_val_categorical, verbose=2)
print('\nValidation accuracy:', val_acc)

# Predictions
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val_categorical, axis=1)

# Confusion matrix
print(confusion_matrix(y_true_classes, y_pred_classes))

# Classification report
print(classification_report(y_true_classes, y_pred_classes))





