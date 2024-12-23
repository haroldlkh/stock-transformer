# model_training.py

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, Concatenate, LayerNormalization,
    MultiHeadAttention, Flatten, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# ----------------------------
# Data Generator Functions using tf.data API
# ----------------------------
def hdf5_data_generator(h5_file_path, batch_size, num_classes):
    """
    Generator function to read data from HDF5 file in batches.
    """
    def generator():
        with h5py.File(h5_file_path, 'r') as h5_file:
            dataset_size = h5_file['X_1h'].shape[0]
            indices = np.arange(dataset_size)

            for idx in range(0, dataset_size, batch_size):
                batch_indices = indices[idx:idx + batch_size]
                # Read batch data
                X_1h_batch = h5_file['X_1h'][batch_indices]
                X_1d_batch = h5_file['X_1d'][batch_indices]
                y_batch = h5_file['y_1d'][batch_indices]  # Using 'y_1d' labels
                ticker_info_batch = h5_file['ticker_info_encoded'][batch_indices]
                ticker_indices_batch = h5_file['ticker_indices'][batch_indices]

                # One-hot encode labels
                y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=num_classes)

                # Prepare input dictionary
                batch_inputs = {
                    'ohlcv_input_1h': X_1h_batch,
                    'ohlcv_input_1d': X_1d_batch,
                    'ticker_info_input': ticker_info_batch,
                    'ticker_input': ticker_indices_batch.reshape(-1, 1)
                }

                yield batch_inputs, y_batch
    return generator

def create_dataset(h5_file_path, batch_size, num_classes):
    """
    Create a tf.data.Dataset from the generator function.
    """
    generator = hdf5_data_generator(h5_file_path, batch_size, num_classes)

    # Open the HDF5 file briefly to get shapes and types
    with h5py.File(h5_file_path, 'r') as h5_file:
        x_1h_shape = h5_file['X_1h'].shape
        x_1d_shape = h5_file['X_1d'].shape
        ticker_info_shape = h5_file['ticker_info_encoded'].shape

    # Define output types and shapes for the dataset
    output_types = (
        {
            'ohlcv_input_1h': tf.float32,
            'ohlcv_input_1d': tf.float32,
            'ticker_info_input': tf.int32,
            'ticker_input': tf.int32
        },
        tf.float32  # y_batch
    )

    output_shapes = (
        {
            'ohlcv_input_1h': tf.TensorShape([None, x_1h_shape[1], x_1h_shape[2]]),
            'ohlcv_input_1d': tf.TensorShape([None, x_1d_shape[1], x_1d_shape[2]]),
            'ticker_info_input': tf.TensorShape([None, ticker_info_shape[1]]),
            'ticker_input': tf.TensorShape([None, 1])
        },
        tf.TensorShape([None, num_classes])  # y_batch
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=output_types,
        output_shapes=output_shapes
    )

    # Prefetch data for optimal performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ----------------------------
# Model Building Function
# ----------------------------
def build_model(sequence_length_1h, sequence_length_1d, num_features_1h, num_features_1d,
                num_classes, ticker_info_input_dim, embedding_dims, ticker_vocab_size):
    """
    Build a transformer model with separate 1h and 1d branches that interact via cross-attention.
    """
    # Inputs for 1h and 1d OHLCV data
    ohlcv_input_1h = Input(shape=(sequence_length_1h, num_features_1h), name='ohlcv_input_1h')
    ohlcv_input_1d = Input(shape=(sequence_length_1d, num_features_1d), name='ohlcv_input_1d')

    # Ticker Input and Embedding
    ticker_input = Input(shape=(1,), dtype='int32', name='ticker_input')
    ticker_embedding = Embedding(input_dim=ticker_vocab_size + 1, output_dim=16, name='ticker_embedding')(ticker_input)
    ticker_embedding = Flatten()(ticker_embedding)

    # Ticker Info Input and Embeddings
    ticker_info_input = Input(shape=(len(ticker_info_input_dim),), dtype='int32', name='ticker_info_input')
    ticker_info_embeddings = []
    for i, (input_dim, output_dim) in enumerate(zip(ticker_info_input_dim, embedding_dims)):
        embedding = Embedding(input_dim=input_dim + 1, output_dim=output_dim, name=f'ticker_info_embedding_{i}')(
            ticker_info_input[:, i])
        embedding = Flatten()(embedding)
        ticker_info_embeddings.append(embedding)

    # Concatenate Ticker Info Embeddings
    ticker_info_embedded = Concatenate()(ticker_info_embeddings + [ticker_embedding])
    ticker_info_embedded = Dense(32, activation='relu')(ticker_info_embedded)

    # Transformer Encoder Block
    def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout_rate, name_prefix):
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f'{name_prefix}_attn')(
            inputs, inputs)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

        # Feed-Forward Network
        ffn_output = Dense(ff_dim, activation='relu', name=f'{name_prefix}_ffn_dense1')(out1)
        ffn_output = Dense(inputs.shape[-1], name=f'{name_prefix}_ffn_dense2')(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        return out2

    # Apply Transformer Encoder to 1h and 1d Data
    x1h = transformer_encoder(
        ohlcv_input_1h,
        num_heads=4,
        key_dim=64,
        ff_dim=128,
        dropout_rate=0.1,
        name_prefix='1h'
    )
    x1d = transformer_encoder(
        ohlcv_input_1d,
        num_heads=4,
        key_dim=64,
        ff_dim=128,
        dropout_rate=0.1,
        name_prefix='1d'
    )

    # Cross-Attention between 1h and 1d Data
    cross_attn_output = MultiHeadAttention(num_heads=4, key_dim=64, name='cross_attn')(x1h, x1d)
    cross_attn_output = Dropout(0.1)(cross_attn_output)
    x_cross = LayerNormalization(epsilon=1e-6)(x1h + cross_attn_output)

    # Feed-Forward Network after Cross-Attention
    ffn_cross = Dense(128, activation='relu', name='cross_ffn_dense1')(x_cross)
    ffn_cross = Dense(x_cross.shape[-1], name='cross_ffn_dense2')(ffn_cross)
    ffn_cross = Dropout(0.1)(ffn_cross)
    x_cross = LayerNormalization(epsilon=1e-6)(x_cross + ffn_cross)

    # Global Average Pooling
    x_pooled = GlobalAveragePooling1D()(x_cross)

    # Combine with Ticker Info Embeddings
    combined = Concatenate()([x_pooled, ticker_info_embedded])

    # Dense Layers for Classification
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Build Model
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
    # Determine Batch Size
    # ----------------------------
    batch_size = 64  # Adjust based on memory constraints

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
        input_dim = max(categorical_mappings[col].values())
        ticker_info_input_dim.append(input_dim)
        # Rule of thumb for embedding dimension
        embedding_dims.append(min(50, (input_dim + 1) // 2))

    # ----------------------------
    # Determine Sequence Lengths and Feature Counts
    # ----------------------------
    # Open the HDF5 file to get sequence lengths and feature counts
    with h5py.File('train_sequences.h5', 'r') as h5f_train:
        sequence_length_1h = h5f_train['X_1h'].shape[1]
        num_features_1h = h5f_train['X_1h'].shape[2]
        sequence_length_1d = h5f_train['X_1d'].shape[1]
        num_features_1d = h5f_train['X_1d'].shape[2]
        num_classes = int(h5f_train['y_1h'][:].max()) + 1  # Ensures all classes are covered

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
    # Compile the Model
    # ----------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ----------------------------
    # Create Datasets
    # ----------------------------
    train_dataset = create_dataset('train_sequences.h5', batch_size=batch_size, num_classes=num_classes)
    val_dataset = create_dataset('test_sequences.h5', batch_size=batch_size, num_classes=num_classes)

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
        validation_data=val_dataset,
        callbacks=[early_stopping],
        verbose=1
    )

    # ----------------------------
    # Save the Trained Model
    # ----------------------------
    model.save('transformer_cross_attention_model1-1.keras')
    print("Model saved to 'transformer_cross_attention_model1-1.keras'")

main_model_training()
