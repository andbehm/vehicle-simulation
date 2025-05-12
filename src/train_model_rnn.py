import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# MinMaxScaler no longer needed if we scale manually
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight # Keep for potential use

# --- Constants and Configuration ---
SENSOR_MAX_RANGE = 300.0
CSV_PATTERN = "vehicle_telemetry_SUCCESS_*.csv"
MODEL_SAVE_PATH = "vehicle_steering_model_rnn.keras" # New model name

# --- RNN Hyperparameters ---
SEQUENCE_LENGTH = 20 # Number of time steps to look back
LSTM_UNITS = 64      # Number of units in the LSTM layer
DENSE_UNITS = 32     # Number of units in the dense layer after LSTM

# --- Training Hyperparameters ---
LEARNING_RATE = 0.0005
EPOCHS = 6000 # RNNs might need more epochs, adjust based on results
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# --- 1. Load and Prepare Data for RNN ---

def create_sequences(data_x, data_y, sequence_length):
    """Transforms data into sequences for RNNs."""
    x_sequences = []
    y_sequences = []
    # Start index is sequence_length - 1 because we need sequence_length points up to and including index i
    # The corresponding y is the action *at* index i (taken after observing sequence ending at i)
    for i in range(sequence_length - 1, len(data_x)):
        x_sequences.append(data_x[i - (sequence_length - 1) : i + 1]) # Sequence from i-len+1 to i
        y_sequences.append(data_y[i]) # Label at time i

    return np.array(x_sequences), np.array(y_sequences)


def load_and_preprocess_data_rnn(csv_pattern, sequence_length):
    """Loads data, preprocesses, creates sequences, and splits for RNN training."""
    all_files = glob.glob(csv_pattern)
    if not all_files:
        print(f"Error: No CSV files found matching pattern '{csv_pattern}'.")
        return None, None, None, None

    print(f"Found {len(all_files)} telemetry files.")

    all_x_seq = []
    all_y_seq = []
    all_y_categorical_flat = [] # For stratify and class weights

    for f in all_files:
        print(f" - Processing {f}")
        try:
            df = pd.read_csv(f)
            df.dropna(inplace=True) # Drop NaNs specific to this file
            if len(df) < sequence_length:
                print(f"   Skipping {f}, not enough data points ({len(df)}) for sequence length {sequence_length}.")
                continue

            # --- Feature Engineering (Input X) ---
            sensor_columns = ['Sensor1_Fwd', 'Sensor2_L30', 'Sensor3_L60', 'Sensor4_R30', 'Sensor5_R60']
            X_norm = df[sensor_columns].values
            X_norm = np.clip(X_norm, 0, SENSOR_MAX_RANGE) / SENSOR_MAX_RANGE

            # --- Target Engineering (Output y) ---
            key_left = df['Key_Left'].astype(bool).values
            key_right = df['Key_Right'].astype(bool).values
            y_categorical = np.zeros(len(df), dtype=int)
            y_categorical[key_left] = 1
            y_categorical[key_right] = 2
            y_one_hot = to_categorical(y_categorical, num_classes=3)

            # --- Create Sequences for this file ---
            # We need y_one_hot corresponding to the *end* of each sequence
            file_x_seq, file_y_seq = create_sequences(X_norm, y_one_hot, sequence_length)

            if len(file_x_seq) > 0:
                all_x_seq.append(file_x_seq)
                all_y_seq.append(file_y_seq)
                # Store the categorical labels corresponding to the sequences created
                all_y_categorical_flat.extend(y_categorical[sequence_length - 1:])
            else:
                 print(f"   No sequences generated for {f} (length {len(df)}).")


        except Exception as e:
            print(f"   Error processing file {f}: {e}")

    if not all_x_seq:
        print("Error: No valid sequences generated from any files.")
        return None, None, None, None

    # Concatenate sequences from all files
    X_final = np.concatenate(all_x_seq, axis=0)
    y_final = np.concatenate(all_y_seq, axis=0)
    y_categorical_final = np.array(all_y_categorical_flat) # For stratify/weights

    print(f"\nTotal sequences generated: {len(X_final)}")
    print(f"Final X shape: {X_final.shape}") # Should be (num_samples, sequence_length, num_features)
    print(f"Final y shape: {y_final.shape}") # Should be (num_samples, num_classes)


    # --- Data Splitting ---
    if len(np.unique(y_categorical_final)) < 2:
        print("Warning: Only one class present in the final dataset. Stratification disabled.")
        stratify_option = None
    else:
        stratify_option = y_categorical_final

    X_train, X_val, y_train, y_val = train_test_split(
        X_final, y_final, test_size=VALIDATION_SPLIT, random_state=42, stratify=stratify_option
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Optional: Check class distribution
    print("\nClass distribution in training data (sequences):")
    print(pd.Series(np.argmax(y_train, axis=1)).value_counts(normalize=True))


    # --- Calculate Class Weights (Optional but recommended) ---
    class_weight_dict = None
    try:
        unique_classes = np.unique(y_categorical_final)
        if len(unique_classes) > 1:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=y_categorical_final
            )
            class_weight_dict = dict(enumerate(class_weights))
            # Need to map unique class labels (0, 1, 2) to dictionary keys if they aren't contiguous
            class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}
            print(f"\nUsing Class Weights: {class_weight_dict}")
        else:
            print("\nSkipping class weights: Only one class found.")

    except Exception as e:
        print(f"\nCould not compute class weights: {e}")


    return X_train, X_val, y_train, y_val, class_weight_dict

# --- 2. Build the RNN Model ---

def build_rnn_model(sequence_length, num_features, num_classes, lstm_units, dense_units, learning_rate):
    """Builds and compiles the Keras RNN (LSTM) model."""
    model = keras.Sequential(
        [
            keras.Input(shape=(sequence_length, num_features), name="input_layer"),
            # LSTM Layer - return_sequences=False because the next layer is Dense
            layers.LSTM(lstm_units, name="lstm_layer"),
            # Optional: Add Dropout for regularization
            # layers.Dropout(0.2),
            # Dense layer after LSTM
            layers.Dense(dense_units, activation="relu", name="dense_layer_1"),
             # Optional: Add Dropout for regularization
            # layers.Dropout(0.2),
            # Output layer
            layers.Dense(num_classes, activation="softmax", name="output_layer")
        ],
        name="steering_predictor_rnn"
    )

    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model

# --- 3. Training and Evaluation ---

if __name__ == "__main__":
    print("--- Starting RNN Model Training ---")

    # Load and preprocess data into sequences
    X_train, X_val, y_train, y_val, class_weight_dict = load_and_preprocess_data_rnn(
        CSV_PATTERN, SEQUENCE_LENGTH
    )

    if X_train is None:
        print("\nExiting due to data loading/processing errors.")
        sys.exit(1)


    # Build the RNN model
    num_features = X_train.shape[2] # Should be 5
    num_classes = y_train.shape[1]   # Should be 3
    model = build_rnn_model(
        SEQUENCE_LENGTH, num_features, num_classes, LSTM_UNITS, DENSE_UNITS, LEARNING_RATE
    )

    print(f"\n--- Training RNN Model for {EPOCHS} Epochs ---")

    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping_cb, reduce_lr_cb],
        class_weight=class_weight_dict, # Apply class weights if calculated
        verbose=1 # Keep verbose=1 to see progress
    )

    print("\n--- Evaluating RNN Model ---")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # --- 4. Save the Model ---
    print(f"\n--- Saving Trained RNN Model to {MODEL_SAVE_PATH} ---")
    try:
        model.save(MODEL_SAVE_PATH)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\n--- RNN Training Complete ---")

    # Optional: Plot training history
    # (Plotting code remains the same)
