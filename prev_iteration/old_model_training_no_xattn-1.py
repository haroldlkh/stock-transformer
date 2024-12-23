# # model_training.py

# import numpy as np
# import pandas as pd
# import h5py
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention, Flatten
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# import pickle

# def build_model(sequence_length_1h, sequence_length_1d, num_features_1h, num_features_1d,
#                 num_classes, ticker_info_input_dim, embedding_dims, ticker_vocab_size):
#     """
#     Build a hierarchical transformer model that processes 1h and 1d data separately
#     and combines their representations along with ticker info and ticker embeddings.
#     """
#     # 1h OHLCV Input
#     ohlcv_input_1h = Input(shape=(sequence_length_1h, num_features_1h), name='ohlcv_input_1h')

#     # 1d OHLCV Input
#     ohlcv_input_1d = Input(shape=(sequence_length_1d, num_features_1d), name='ohlcv_input_1d')

#     # Ticker Input
#     ticker_input = Input(shape=(1,), dtype='int32', name='ticker_input')
#     ticker_embedding = Embedding(input_dim=ticker_vocab_size + 2, output_dim=16)(ticker_input)
#     ticker_embedding = Flatten()(ticker_embedding)

#     # Ticker Info Input
#     ticker_info_input = Input(shape=(len(ticker_info_input_dim),), dtype='int32', name='ticker_info_input')

#     # Embedding for Ticker Info
#     ticker_info_embeddings = []
#     for i, (input_dim, output_dim) in enumerate(zip(ticker_info_input_dim, embedding_dims)):
#         # Adjust input_dim for possible unknown categories
#         embedding = Embedding(input_dim=input_dim + 2, output_dim=output_dim)(ticker_info_input[:, i])
#         embedding = Flatten()(embedding)
#         ticker_info_embeddings.append(embedding)

#     # Concatenate ticker embeddings
#     ticker_info_embedded = Concatenate()(ticker_info_embeddings + [ticker_embedding])
#     ticker_info_embedded = Dense(32, activation='relu')(ticker_info_embedded)

#     # 1h Transformer Encoder
#     x1h = ohlcv_input_1h
#     attn_output_1h = MultiHeadAttention(num_heads=4, key_dim=64)(x1h, x1h)
#     attn_output_1h = Dropout(0.1)(attn_output_1h)
#     x1h = LayerNormalization(epsilon=1e-6)(x1h + attn_output_1h)
#     ffn_output_1h = Dense(64, activation="relu")(x1h)
#     ffn_output_1h = Dense(num_features_1h)(ffn_output_1h)
#     ffn_output_1h = Dropout(0.1)(ffn_output_1h)
#     x1h = LayerNormalization(epsilon=1e-6)(x1h + ffn_output_1h)
#     x1h_pooled = GlobalAveragePooling1D()(x1h)

#     # 1d Transformer Encoder
#     x1d = ohlcv_input_1d
#     attn_output_1d = MultiHeadAttention(num_heads=4, key_dim=64)(x1d, x1d)
#     attn_output_1d = Dropout(0.1)(attn_output_1d)
#     x1d = LayerNormalization(epsilon=1e-6)(x1d + attn_output_1d)
#     ffn_output_1d = Dense(64, activation="relu")(x1d)
#     ffn_output_1d = Dense(num_features_1d)(ffn_output_1d)
#     ffn_output_1d = Dropout(0.1)(ffn_output_1d)
#     x1d = LayerNormalization(epsilon=1e-6)(x1d + ffn_output_1d)
#     x1d_pooled = GlobalAveragePooling1D()(x1d)

#     # Combine 1h and 1d representations
#     combined_time = Concatenate()([x1h_pooled, x1d_pooled])

#     # Combine with ticker info embeddings
#     combined = Concatenate()([combined_time, ticker_info_embedded])

#     # Dense layers
#     x = Dense(128, activation="relu")(combined)
#     x = Dropout(0.2)(x)
#     x = Dense(64, activation="relu")(x)
#     outputs = Dense(num_classes, activation="softmax")(x)

#     # Build Model
#     model = Model(inputs=[ohlcv_input_1h, ohlcv_input_1d, ticker_info_input, ticker_input], outputs=outputs)
#     return model

# def main_model_training():
#     """
#     Main function to build, compile, train, and save the hierarchical transformer model.
#     """
#     # Load sequences for 1h data
#     h5f_1h = h5py.File('/content/1h_train_data_20240926_042110.h5', 'r')
#     X_train_seq_1h = h5f_1h['X_sequences'][:]
#     y_train_seq_1h = h5f_1h['y_sequences'][:]
#     ticker_indices_1h = h5f_1h['ticker_indices'][:]
#     h5f_1h.close()

#     # Load sequences for 1d data
#     h5f_1d = h5py.File('/content/1d_train_data_20240926_042417.h5', 'r')
#     X_train_seq_1d = h5f_1d['X_sequences'][:]
#     y_train_seq_1d = h5f_1d['y_sequences'][:]
#     ticker_indices_1d = h5f_1d['ticker_indices'][:]
#     h5f_1d.close()

#     # Ensure the lengths match
#     min_len = min(len(X_train_seq_1h), len(X_train_seq_1d))
#     X_train_seq_1h = X_train_seq_1h[:min_len]
#     X_train_seq_1d = X_train_seq_1d[:min_len]
#     y_train_seq = y_train_seq_1h[:min_len]  # Use y_train_seq_1h or y_train_seq_1d
#     ticker_indices = ticker_indices_1h[:min_len]  # Assuming tickers are aligned

#     # Load the label encoders for OHLCV data
#     with open('label_encoders_1h.pkl', 'rb') as f:
#         ohlcv_label_encoders = pickle.load(f)

#     # Get the LabelEncoder for 'Ticker'
#     ticker_le = ohlcv_label_encoders['Ticker']
#     ticker_vocab_size = len(ticker_le.classes_)

#     # Map encoded tickers to ticker symbols using the LabelEncoder
#     ticker_symbols = ticker_le.inverse_transform(ticker_indices)

#     # Load the raw ticker info data
#     ticker_info_data = pd.read_parquet('/content/raw_ticker_info_data_20240926_041538.parquet')
#     ticker_info_data.set_index('Ticker', inplace=True)

#     # Load label encoders and max category indices for ticker info data
#     with open('label_encoders_ticker.pkl', 'rb') as f:
#         ticker_label_encoders = pickle.load(f)
#     with open('max_category_indices_ticker.pkl', 'rb') as f:
#         ticker_max_category_indices = pickle.load(f)

#     ticker_categorical_cols = ['Industry', 'Sector', 'QuoteType']

#     # Encode categorical columns in ticker_info_data using the saved encoders
#     for col in ticker_categorical_cols:
#         le = ticker_label_encoders[col]
#         ticker_info_data[col] = le.transform(ticker_info_data[col].astype(str))

#     # Prepare input dimensions for embeddings
#     ticker_info_input_dim = []
#     for col in ticker_categorical_cols:
#         max_index = ticker_max_category_indices[col]
#         ticker_info_input_dim.append(max_index + 1)

#     # Prepare embedding dimensions
#     embedding_dims = [min(50, (dim + 1) // 2) for dim in ticker_info_input_dim]

#     # Ensure that tickers are present in ticker_info_data
#     valid_indices = [i for i, ticker in enumerate(ticker_symbols) if ticker in ticker_info_data.index]

#     # Filter sequences and labels based on valid tickers
#     X_train_seq_1h = X_train_seq_1h[valid_indices]
#     X_train_seq_1d = X_train_seq_1d[valid_indices]
#     y_train_seq = y_train_seq[valid_indices]
#     ticker_indices = ticker_indices[valid_indices]
#     ticker_symbols = [ticker_symbols[i] for i in valid_indices]

#     # Extract ticker info data for the tickers in our sequences
#     ticker_info_array = ticker_info_data.loc[ticker_symbols][ticker_categorical_cols].values.astype(np.int32)

#     # Prepare inputs for the model
#     num_classes = y_train_seq.shape[1]
#     num_features_1h = X_train_seq_1h.shape[2]
#     num_features_1d = X_train_seq_1d.shape[2]
#     sequence_length_1h = X_train_seq_1h.shape[1]
#     sequence_length_1d = X_train_seq_1d.shape[1]

#     # Build the model
#     model = build_model(
#         sequence_length_1h=sequence_length_1h,
#         sequence_length_1d=sequence_length_1d,
#         num_features_1h=num_features_1h,
#         num_features_1d=num_features_1d,
#         num_classes=num_classes,
#         ticker_info_input_dim=ticker_info_input_dim,
#         embedding_dims=embedding_dims,
#         ticker_vocab_size=ticker_vocab_size
#     )

#     # Compile the model with optimizer, loss, and metrics
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     # Prepare inputs for training
#     train_inputs = {
#         'ohlcv_input_1h': X_train_seq_1h,
#         'ohlcv_input_1d': X_train_seq_1d,
#         'ticker_info_input': ticker_info_array,
#         'ticker_input': ticker_indices.reshape(-1, 1)
#     }

#     # Train the model with early stopping
#     model.fit(
#         train_inputs,
#         y_train_seq,
#         epochs=50,
#         batch_size=64,
#         validation_split=0.1,
#         callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
#     )

#     # Save the trained model
#     model.save('hierarchical_transformer_model.keras')
#     print("Model saved to hierarchical_transformer_model.keras")

# # Run model training
# main_model_training()
