# model_training.py
# !pip install tensorflow-io
# TFIO not working.

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, Concatenate, LayerNormalization,
    MultiHeadAttention, Flatten, GlobalAveragePooling1D, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K  # For handling dynamic shapes
import tensorflow_io as tfio
import pickle

# ----------------------------
# Data Generator Functions using TensorFlow I/O
# ----------------------------
def create_dataset(h5_file_path, batch_size, num_classes):
    """
    Create a tf.data.Dataset from the HDF5 file using TensorFlow I/O.
    """
    # Read datasets directly from HDF5 file
    X_1h_dataset = tfio.IODataset.from_hdf5(h5_file_path, dataset='/X_1h')
    X_1d_dataset = tfio.IODataset.from_hdf5(h5_file_path, dataset='/X_1d')
    y_dataset = tfio.IODataset.from_hdf5(h5_file_path, dataset='/y_1d')
    ticker_info_dataset = tfio.IODataset.from_hdf5(h5_file_path, dataset='/ticker_info_encoded')
    ticker_indices_dataset = tfio.IODataset.from_hdf5(h5_file_path, dataset='/ticker_indices')

    # Zip the datasets together
    dataset = tf.data.Dataset.zip((
        X_1h_dataset,
        X_1d_dataset,
        y_dataset,
        ticker_info_dataset,
        ticker_indices_dataset
    ))

    # Map to input dictionary and one-hot encode labels
    def map_fn(X_1h, X_1d, y, ticker_info, ticker_index):
        # Adjust y shape and cast to int32
        y = tf.cast(y, tf.int32)
        y = tf.reshape(y, [])  # Ensure y is scalar
        y_one_hot = tf.one_hot(y, depth=num_classes)

        # Reshape ticker_index to match expected input shape
        ticker_index = tf.cast(ticker_index, tf.int32)
        ticker_index = tf.reshape(ticker_index, [1])

        # Prepare the batch inputs
        batch_inputs = {
            'ohlcv_input_1h': X_1h,
            'ohlcv_input_1d': X_1d,
            'ticker_info_input': ticker_info,
            'ticker_input': ticker_index
        }
        return batch_inputs, y_one_hot

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for performance optimization
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Get dataset size for reference
    with h5py.File(h5_file_path, 'r') as h5_file:
        dataset_size = h5_file['X_1h'].shape[0]

    return dataset, dataset_size

# ----------------------------
# Model Building Function
# ----------------------------
def build_model(sequence_length_1h, sequence_length_1d, num_features_1h, num_features_1d,
                num_classes, ticker_info_input_dim, embedding_dims, ticker_vocab_size):
    """
    Build a transformer model with cross-attention.
    """
    # Inputs for 1h and 1d OHLCV data
    ohlcv_input_1h = Input(shape=(sequence_length_1h, num_features_1h), name='ohlcv_input_1h')
    ohlcv_input_1d = Input(shape=(sequence_length_1d, num_features_1d), name='ohlcv_input_1d')

    # Ticker Input and Embedding
    ticker_input = Input(shape=(1,), dtype='int32', name='ticker_input')
    ticker_embedding = Embedding(input_dim=ticker_vocab_size + 1, output_dim=16, name='ticker_embedding')(ticker_input)
    ticker_embedding = Flatten()(ticker_embedding)  # Shape: (batch_size, 16)

    # Ticker Info Input and Embeddings
    ticker_info_input = Input(shape=(len(ticker_info_input_dim),), dtype='int32', name='ticker_info_input')
    ticker_info_embeddings = []
    for i, (input_dim, output_dim) in enumerate(zip(ticker_info_input_dim, embedding_dims)):
        # Extract the ith element using Lambda layer
        get_i = Lambda(lambda x: x[:, i])(ticker_info_input)
        # Create an embedding layer for the ith categorical feature
        embedding = Embedding(input_dim=input_dim + 1, output_dim=output_dim, name=f'ticker_info_embedding_{i}')(get_i)
        ticker_info_embeddings.append(embedding)  # Each embedding has shape: (batch_size, output_dim)

    # Concatenate Ticker Info Embeddings
    ticker_info_embedded = Concatenate()(ticker_info_embeddings + [ticker_embedding])  # Shape: (batch_size, total_embed_dim)
    ticker_info_embedded = Flatten()(ticker_info_embedded)
    ticker_info_embedded = Dense(32, activation='relu')(ticker_info_embedded)  # Reduce dimensionality

    # Transformer Encoder Block
    def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout_rate, name_prefix):
        # Get the input dimension
        input_dim = K.int_shape(inputs)[-1]
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f'{name_prefix}_attn')(
            inputs, inputs)
        attn_output = Dropout(dropout_rate)(attn_output)
        # Add & Norm
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

        # Feed-Forward Network
        ffn_output = Dense(ff_dim, activation='relu', name=f'{name_prefix}_ffn_dense1')(out1)
        ffn_output = Dense(input_dim, name=f'{name_prefix}_ffn_dense2')(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        # Add & Norm
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        return out2

    # Apply Transformer Encoder to 1h and 1d Data
    x1h = transformer_encoder(
        ohlcv_input_1h,
        num_heads=2,
        key_dim=32,
        ff_dim=64,
        dropout_rate=0.1,
        name_prefix='1h'
    )
    x1d = transformer_encoder(
        ohlcv_input_1d,
        num_heads=2,
        key_dim=32,
        ff_dim=64,
        dropout_rate=0.1,
        name_prefix='1d'
    )

    # Cross-Attention between 1h and 1d Data
    cross_attn_output = MultiHeadAttention(num_heads=2, key_dim=32, name='cross_attn')(x1h, x1d)
    cross_attn_output = Dropout(0.1)(cross_attn_output)
    # Add & Norm
    x_cross = LayerNormalization(epsilon=1e-6)(x1h + cross_attn_output)

    # Feed-Forward Network after Cross-Attention
    input_dim_cross = K.int_shape(x_cross)[-1]
    ffn_cross = Dense(64, activation='relu', name='cross_ffn_dense1')(x_cross)
    ffn_cross = Dense(input_dim_cross, name='cross_ffn_dense2')(ffn_cross)
    ffn_cross = Dropout(0.1)(ffn_cross)
    # Add & Norm
    x_cross = LayerNormalization(epsilon=1e-6)(x_cross + ffn_cross)

    # Global Average Pooling
    x_pooled = GlobalAveragePooling1D()(x_cross)  # Shape: (batch_size, feature_dim)

    # Combine with Ticker Info Embeddings
    combined = Concatenate()([x_pooled, ticker_info_embedded])  # Shape depends on both inputs

    # Dense Layers for Classification
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    # Output layer with softmax activation
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    # Build and return the model
    model = Model(inputs=[ohlcv_input_1h, ohlcv_input_1d, ticker_info_input, ticker_input], outputs=outputs)
    return model

# ----------------------------
# Main Model Training Function
# ----------------------------
def main_model_training():
    """
    Main function to build, compile, train, and save the transformer model with cross-attention.
    """
    # ----------------------------
    # Enable Mixed Precision
    # ----------------------------
    mixed_precision.set_global_policy('mixed_float16')

    # ----------------------------
    # Determine Batch Size
    # ----------------------------
    batch_size = 128  # Adjusted batch size

    # ----------------------------
    # Load Encoders and Determine Dimensions
    # ----------------------------
    # Load Ticker LabelEncoder
    with open('ticker_label_encoder.pkl', 'rb') as f:
        ticker_le = pickle.load(f)
    ticker_vocab_size = len(ticker_le.classes_)

    # Load Categorical Encoders and Mappings
    with open('categorical_encoders.pkl', 'rb') as f:
        categorical_encoders = pickle.load(f)
    with open('categorical_mappings.pkl', 'rb') as f:
        categorical_mappings = pickle.load(f)

    # Determine Input Dimensions for Ticker Info Embeddings
    ticker_info_input_dim = []
    embedding_dims = []
    for col in ['Industry', 'Sector', 'QuoteType']:
        # Input dimension is maximum index value plus one
        input_dim = max(categorical_mappings[col].values()) + 1
        ticker_info_input_dim.append(input_dim)
        # Rule of thumb for embedding dimension
        embedding_dims.append(min(50, (input_dim + 1) // 2))

    # ----------------------------
    # Determine Sequence Lengths and Feature Counts
    # ----------------------------
    # Open the HDF5 files to get sequence lengths and feature counts
    with h5py.File('train_sequences.h5', 'r') as h5f_train:
        sequence_length_1h = h5f_train['X_1h'].shape[1]
        num_features_1h = h5f_train['X_1h'].shape[2]
        num_train_samples = h5f_train['X_1h'].shape[0]
        # Use 'y_1d' labels for predicting next day movements
        num_classes = int(h5f_train['y_1d'][:].max()) + 1  # Ensures all classes are covered

    with h5py.File('test_sequences.h5', 'r') as h5f_test:
        sequence_length_1d = h5f_test['X_1d'].shape[1]
        num_features_1d = h5f_test['X_1d'].shape[2]
        num_val_samples = h5f_test['X_1h'].shape[0]
        # Update num_classes if needed based on test data
        num_classes = max(num_classes, int(h5f_test['y_1d'][:].max()) + 1)

    # Calculate steps per epoch
    steps_per_epoch = int(np.ceil(num_train_samples / batch_size))
    validation_steps = int(np.ceil(num_val_samples / batch_size))

    # ----------------------------
    # Build the Model
    # ----------------------------
    model = build_model(
        sequence_length_1h=sequence_length_1h,
        sequence_length_1d=sequence_length_1d,
        num_features_1h=num_features_1h,
        num_features_1d=num_features_1d,
        num_classes=num_classes,
        ticker_info_input_dim=ticker_info_input_dim,
        embedding_dims=embedding_dims,
        ticker_vocab_size=ticker_vocab_size
    )

    # ----------------------------
    # Define Learning Rate Schedule
    # ----------------------------
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=steps_per_epoch * 10,  # Decay every 10 epochs
        decay_rate=0.9
    )

    # ----------------------------
    # Compile the Model
    # ----------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ----------------------------
    # Create Datasets
    # ----------------------------
    train_dataset, _ = create_dataset('train_sequences.h5', batch_size=batch_size, num_classes=num_classes)
    val_dataset, _ = create_dataset('test_sequences.h5', batch_size=batch_size, num_classes=num_classes)

    # ----------------------------
    # Training Callbacks
    # ----------------------------
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # ----------------------------
    # Train the Model
    # ----------------------------
    model.fit(
        train_dataset,
        epochs=50,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=[early_stopping],
        verbose=1
    )

    # ----------------------------
    # Save the Trained Model
    # ----------------------------
    model.save('transformer_cross_attention_model2.keras')
    print("Model saved to 'transformer_cross_attention_model2.keras'")

main_model_training()
