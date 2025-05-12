import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # Alternative for scaling
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# --- Constants and Configuration ---
# Ensure this matches the value used in the simulation script for normalization
SENSOR_MAX_RANGE = 300.0
CSV_PATTERN = "vehicle_telemetry_SUCCESS_*.csv" # Pattern to find telemetry files
MODEL_SAVE_PATH = "vehicle_steering_model.keras" # Where to save the trained model

# Neural Network Hyperparameters
HIDDEN_UNITS = 32
LEARNING_RATE = 0.001
EPOCHS = 500000
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2 # Use 20% of data for validation

# --- 1. Load and Prepare Data ---

def load_and_preprocess_data(csv_pattern):
    """Loads data from CSV files, preprocesses, and prepares for training."""
    all_files = glob.glob(csv_pattern)
    if not all_files:
        print(f"Error: No CSV files found matching pattern '{csv_pattern}'.")
        print("Please run the simulation and ensure successful runs are saved.")
        return None, None

    print(f"Found {len(all_files)} telemetry files:")
    for f in all_files:
        print(f" - {f}")

    # Load all found CSVs into a single DataFrame
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    print(f"\nLoaded total {len(df)} data points.")

    # Drop rows with potential NaN values (if any occurred)
    df.dropna(inplace=True)
    if len(df) == 0:
        print("Error: No valid data points after dropping NaNs.")
        return None, None

    # --- Feature Engineering (Input X) ---
    sensor_columns = ['Sensor1_Fwd', 'Sensor2_L30', 'Sensor3_L60', 'Sensor4_R30', 'Sensor5_R60']
    X = df[sensor_columns].values

    # Normalize sensor data (Scale to 0-1 range based on max range)
    # Clip values first to avoid division by zero if range is 0, although unlikely here
    X = np.clip(X, 0, SENSOR_MAX_RANGE)
    X = X / SENSOR_MAX_RANGE
    print(f"Input features (X) shape: {X.shape}")
    # print("Sample normalized X:\n", X[:5]) # Optional: print sample input


    # --- Target Engineering (Output y) ---
    key_left = df['Key_Left'].astype(bool).values
    key_right = df['Key_Right'].astype(bool).values

    # Create categorical labels: 0: Straight, 1: Left, 2: Right
    # Assumes Key_Left and Key_Right are never True simultaneously (as per simulation logic)
    y_categorical = np.zeros(len(df), dtype=int) # Default to 0 (Straight)
    y_categorical[key_left] = 1 # Set 1 for Left
    y_categorical[key_right] = 2 # Set 2 for Right

    # One-Hot Encode the categorical labels
    num_classes = 3 # (Straight, Left, Right)
    y_one_hot = to_categorical(y_categorical, num_classes=num_classes)

    print(f"Output labels (y_one_hot) shape: {y_one_hot.shape}")
    # print("Sample y_categorical:", y_categorical[:10]) # Optional: print sample categories
    # print("Sample y_one_hot:\n", y_one_hot[:5]) # Optional: print sample one-hot


    # --- Data Splitting ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_one_hot, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_categorical
        # Use stratify to try and keep class proportions similar in train/val sets
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Check class distribution (optional but good practice)
    print("\nClass distribution in original data:")
    print(pd.Series(y_categorical).value_counts(normalize=True))
    print("\nClass distribution in training data:")
    print(pd.Series(np.argmax(y_train, axis=1)).value_counts(normalize=True)) # Convert one-hot back for counting
    print("\nClass distribution in validation data:")
    print(pd.Series(np.argmax(y_val, axis=1)).value_counts(normalize=True))


    return X_train, X_val, y_train, y_val

# --- 2. Build the Neural Network Model ---

def build_model(input_shape, num_classes, hidden_units, learning_rate):
    """Builds and compiles the Keras sequential model."""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="input_layer"),
            layers.Dense(hidden_units, activation="relu", name="hidden_layer_1"),
            # You could add more hidden layers here if needed:
            # layers.Dense(hidden_units // 2, activation="relu", name="hidden_layer_2"),
            layers.Dense(num_classes, activation="softmax", name="output_layer") # Softmax for multi-class probabilities
        ],
        name="steering_predictor"
    )

    model.summary() # Print model architecture

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy", # Use for one-hot encoded labels + softmax
        optimizer=optimizer,
        metrics=["accuracy"] # Monitor accuracy during training
    )
    return model

# --- 3. Training and Evaluation ---

if __name__ == "__main__":
    print("--- Starting Neural Network Training ---")

    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data(CSV_PATTERN)

    if X_train is None:
        print("\nExiting due to data loading errors.")
        sys.exit(1) # Exit if data loading failed


    # Build the model
    input_shape = (X_train.shape[1],) # Should be (5,) based on sensor columns
    num_classes = y_train.shape[1]   # Should be 3 based on one-hot encoding
    model = build_model(input_shape, num_classes, HIDDEN_UNITS, LEARNING_RATE)

    print(f"\n--- Training Model for {EPOCHS} Epochs ---")

    # Set up callbacks (optional but recommended)
    # EarlyStopping: Stop training if validation loss doesn't improve for a number of epochs
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # ReduceLROnPlateau: Reduce learning rate if validation loss plateaus
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)


    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping_cb, reduce_lr_cb], # Add callbacks here
        verbose=1 # Show progress bar per epoch
    )

    print("\n--- Evaluating Model ---")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # --- 4. Save the Model ---
    print(f"\n--- Saving Trained Model to {MODEL_SAVE_PATH} ---")
    try:
        model.save(MODEL_SAVE_PATH)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\n--- Training Complete ---")

    # Optional: Plot training history (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1) # Adjust y-axis limits if needed
        plt.title('Model Training History')
        plt.xlabel('Epochs')
        plt.show()
    except ImportError:
        print("\nInstall matplotlib (`pip install matplotlib`) to see training graphs.")
