import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense # Assuming Dense layers for detailed calc
import numpy as np
import argparse
import os

# --- Constants ---
SENSOR_MAX_RANGE = 300.0

def scale_inputs(inputs, max_range):
    """Scales input values to be between 0 and 1."""
    scaled = np.array(inputs) / max_range
    # Optional: Clip values to ensure they are strictly within [0, 1]
    # scaled = np.clip(scaled, 0.0, 1.0)
    return scaled

def stable_softmax(x):
    """Compute softmax values for x in a numerically stable way."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def main(model_path, input_string):
    """
    Loads a Keras model, processes inputs, and performs a step-by-step forward pass.
    """
    # --- Validate Model Path ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    # --- Process Inputs ---
    try:
        raw_inputs = [float(val.strip()) for val in input_string.split(',')]
        print(f"Raw Inputs: {raw_inputs}")
    except ValueError:
        print("Error: Input string must contain comma-separated numeric values.")
        return

    # Scale inputs
    scaled_inputs = scale_inputs(raw_inputs, SENSOR_MAX_RANGE)
    print(f"Scaled Inputs (Input / {SENSOR_MAX_RANGE}): {scaled_inputs}")

    # Reshape for model (add batch dimension) -> (1, num_features)
    model_input = np.array([scaled_inputs])
    print(f"Model Input (Shape: {model_input.shape}):\n{model_input}\n")
    print("-" * 50)

    # --- Load Model ---
    try:
        # Load without compiling if optimizer state isn't needed for prediction
        model = load_model(model_path, compile=False)
        print(f"Successfully loaded model '{model.name}' from '{model_path}'")
        model.summary() # Print summary for context
        print("-" * 50)

        # --- Verify Input Shape Compatibility ---
        try:
             # Use model.input_shape which usually includes None for batch size
             expected_input_shape = model.input_shape
             if isinstance(expected_input_shape, list): # Handle models with multiple inputs
                 expected_input_shape = expected_input_shape[0] # Assume first input

             if len(expected_input_shape) > 1 and expected_input_shape[1] is not None:
                 expected_features = expected_input_shape[1]
                 if model_input.shape[1] != expected_features:
                     print(f"Error: Input shape mismatch. Model expects {expected_features} features, but received {model_input.shape[1]}.")
                     print(f"Model expected input shape (including batch): {expected_input_shape}")
                     return
             else:
                  print("Warning: Could not reliably verify input feature count from model.input_shape.")

        except Exception as shape_error:
             print(f"Warning: Could not verify input shape against model. Error: {shape_error}")


    except Exception as e:
        print(f"Error loading Keras model: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step-by-Step Forward Pass ---
    current_output = model_input
    print("\n--- Starting Step-by-Step Calculation ---\n")

    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        print(f"--- Layer {i}: '{layer_name}' ({type(layer).__name__}) ---")
        print(f"Input to Layer:\n{current_output}")

        # Check if it's a layer with weights (like Dense)
        if isinstance(layer, Dense):
            weights, biases = layer.get_weights()
            # weights shape: (num_input_features, num_output_neurons)
            # biases shape: (num_output_neurons,)

            print(f"\nWeights (Shape: {weights.shape}):\n{weights}")
            print(f"\nBiases (Shape: {biases.shape}):\n{biases}")

            # Calculate Z = W * X + b
            # Use np.dot for matrix multiplication: (1, n_in) x (n_in, n_out) -> (1, n_out)
            z = np.dot(current_output, weights) + biases
            print(f"\nPre-activation Output (Z = Input @ Weights + Biases) (Shape: {z.shape}):\n{z}")

            # Get activation function
            activation_func = tf.keras.activations.get(layer.activation)
            activation_name = layer.activation.__name__

            # Apply activation function
            # Need to convert numpy array to tensor for TF activations
            current_output = activation_func(tf.constant(z, dtype=tf.float32)).numpy()

            print(f"\nActivation Function: '{activation_name}'")
            print(f"Activated Output (A = activation(Z)) (Shape: {current_output.shape}):\n{current_output}\n")

        elif hasattr(layer, 'activation'):
             # Handle layers like Activation explicitly applied
             activation_func = tf.keras.activations.get(layer.activation)
             activation_name = layer.activation.__name__
             z = current_output # Input is the pre-activation value for an Activation layer
             print("\nApplying Activation Layer...")
             current_output = activation_func(tf.constant(z, dtype=tf.float32)).numpy()
             print(f"Activation Function: '{activation_name}'")
             print(f"Activated Output (Shape: {current_output.shape}):\n{current_output}\n")

        else:
            # For layers without weights/standard activations (e.g., Flatten, Dropout in inference)
            # we can try to call the layer directly if needed, but often just pass output
            print(f"\nPassing output through layer (no detailed calculation shown for type {type(layer).__name__})...")
            # For simplicity, we assume the output shape doesn't change in a way
            # that breaks the next Dense layer's dot product.
            # A more robust solution might involve calling layer(tf.constant(current_output))
            # but that requires the model to be built/runnable.
            # Since we loaded with compile=False and didn't run dummy data,
            # we'll just pass the value through for non-Dense layers in this example.
            pass # current_output remains the same

        print("-" * 50)


    print("--- Final Output ---")
    print(f"Final Prediction (Output of last layer):\n{current_output}")
    print(f"Shape: {current_output.shape}")

    # Optional: If the last layer is softmax, find the predicted class index
    if activation_name == 'softmax': # Check based on the last activation processed
         predicted_class_index = np.argmax(current_output, axis=-1)
         print(f"\nPredicted Class Index (if applicable): {predicted_class_index[0]}") # [0] because of batch dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a Keras model and perform step-by-step prediction.")
    parser.add_argument("model_file", help="Path to the Keras model file (.keras or .h5)")
    parser.add_argument("input_values", help="Comma-separated string of numeric input values (e.g., '110.9,50.7,39.2')")

    args = parser.parse_args()

    main(args.model_file, args.input_values)
