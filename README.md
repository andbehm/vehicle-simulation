# AI Vehicle Simulation for High School Students

## Introduction

This repository contains a vehicle simulation designed to introduce high school students to the concepts of Artificial Intelligence (AI), Deep Learning, and Neural Networks through a motivating and interactive example of autonomous driving.

The core idea is a simple game where a vehicle, moving at a constant speed, must navigate a parcour without hitting walls.

**Manual Driving Mode:**
* The player controls the vehicle using the **left and right arrow keys**.
* The objective is to keep the vehicle moving and avoid collisions with the walls.
* Reaching the **green square** signifies the end of a successful run.
* Upon reaching the green square, the simulation saves a history of telemetric data to a CSV file. This data is crucial for training the AI model.
* Telemetric data is recorded 5 times per second and includes:
    * Distance to the nearest wall directly in front.
    * Distance to the nearest wall at a 30-degree angle to the left.
    * Distance to the nearest wall at a 60-degree angle to the left.
    * Distance to the nearest wall at a 30-degree angle to the right.
    * Distance to the nearest wall at a 60-degree angle to the right.

**AI Model & Autonomous Mode:**
* The collected telemetric data is used to train a neural network.
* The basic neural network architecture consists of:
    * An input layer with 5 nodes (corresponding to the 5 distance sensors).
    * A hidden layer with a configurable number of nodes (typically between 4 and 32).
    * An output layer with 3 nodes, predicting the vehicle's next action: go straight, turn left, or turn right.
* The simulation can be switched to **autonomous mode**, where the trained AI model makes driving decisions based on the live telemetric data.
* In autonomous mode, the top-right corner of the simulation displays the current telemetric data and the model's predicted action.
* Interestingly, the model is *not* explicitly trained on reaching the green square; its sole objective, learned from the training data, is to avoid hitting walls.
* It was discovered that a simple feed-forward neural network was not always sufficient for robust navigation. An **RNN (Recurrent Neural Network)-based model**, utilizing a sequence of 10 previous telemetric readings, significantly improved performance.

## How to Use the Python Scripts

Make sure you have Python installed, along with the necessary libraries (see `requirements.txt`).

1.  **`parcour_sim_auto.py`**:
    * **Purpose:** Starts the vehicle simulation game using the standard feed-forward neural network model.
    * **Usage:** `python parcour_sim_auto.py`
    * **Parameters:** None required.
    * **Details:** The script includes two pre-defined parcours, `layout1_walls` and `layout2_walls` (currently identical for demonstration purposes). You can switch between manual and autonomous mode within the simulation. Data from successful manual runs is saved to CSV files.

2.  **`parcour_sim_auto_rnn.py`**:
    * **Purpose:** Starts the vehicle simulation game using the RNN-based model.
    * **Usage:** `python parcour_sim_auto_rnn.py`
    * **Parameters:** None required.
    * **Details:** Similar to `parcour_sim_auto.py` but utilizes the more advanced RNN model for autonomous driving.

3.  **`train_model.py`**:
    * **Purpose:** Trains the standard feed-forward neural network model.
    * **Usage:** `python train_model.py`
    * **Parameters:** None required.
    * **Details:** This script will automatically find and read all `.csv` files (containing telemetric data from manual runs) in the current directory to train the model. The trained model is typically saved as `vehicle_steering_model.keras` (or a similar name).

4.  **`train_model_rnn.py`**:
    * **Purpose:** Trains the RNN-based model.
    * **Usage:** `python train_model_rnn.py`
    * **Parameters:** None required.
    * **Details:** Similar to `train_model.py`, but it trains the RNN model, typically saving it with a name like `vehicle_steering_model_rnn.keras`. It expects the CSV data to be suitable for sequence-based learning.

5.  **`keras_debug.py`**:
    * **Purpose:** A utility script to inspect the predictions of a trained Keras model for a given set of input telemetric data. This is useful for understanding how the model responds to specific sensor readings.
    * **Usage:** `python keras_debug.py <model_file.keras> "<input_data>"`
    * **Example:** `python keras_debug.py vehicle_steering_model4.keras "110.929,50.779,39.225,288.914,98.336"`
    * **Parameters:**
        * `<model_file.keras>`: The path to the saved Keras model file.
        * `<input_data>`: A comma-separated string of 5 floating-point numbers representing the telemetric data (distances forward, left 30°, left 60°, right 30°, right 60°). **Ensure the input data is enclosed in quotes.**

## Presentation Slides & Resources

This section contains links to the slides and other resources used in the presentation to introduce the concepts.

* **Understanding AI: From Neural Networks to Large Language Models:** [[Slide](https://claude.ai/public/artifacts/40087620-452a-4d45-babc-a361dacc55be)]
* **Why Understanding AI Matters for Today's 10th Graders:** [[Slide](https://claude.ai/public/artifacts/08998e11-30dc-45cd-9bcd-497867a2a09d)]
* **Building the Vehicle Simulation with Gemini:** [[Slide](https://claude.ai/public/artifacts/da36f44c-240c-428b-840a-2ef480caa129)]
* **Training Data Example:** [[Slide](https://claude.ai/public/artifacts/201a1d14-bcd6-4661-bf99-a012d174b83b)]
* **Neural Network Structure:** [[Slide](https://claude.ai/public/artifacts/06a01d25-5a43-460b-8cb0-aa2c3c136154)]
* **Neural Network Structure:** [[Slide](https://claude.ai/public/artifacts/ee23a3a1-e9b3-4491-b79a-9fc56f33aecb)]
* **Neural Network Structure:** [[Slide](https://claude.ai/public/artifacts/3e414c02-2f0c-4b3e-98a1-942792b76f1c)]
* **Loss Function Motivation for Neural Networks:** [[Slide](https://claude.ai/public/artifacts/6de48b25-cb69-487c-b995-29eb3720d9ab)]
* **Gradient Descent & Loss Function:** [[Slide](https://claude.ai/public/artifacts/254054ae-a8da-40b1-9c15-fd22a002880a)]
* **Model Training History:** [images/model-training-history.png]
* **Neural Network Structure:** [[Slide](https://claude.ai/public/artifacts/5aab3942-c5f8-4fac-b2c0-798427d1650e)] (Note: calculation below is not updated)
* **From Neural Networks to Large Language Models:** [[Slide](https://claude.ai/public/artifacts/6f0d2988-9ea5-4be1-9f20-6142956989a4)]
* **How LLMs Answer Questions:** [[Slide](https://claude.ai/public/artifacts/5d64498b-587a-410c-97f1-ab776f1c4ab7)]
* **Modern AI Systems:** [[Slide](https://claude.ai/public/artifacts/157d8b08-4e34-44e7-962f-4954cc8bfc27)]
* **Using GenAI Responsibly in IB Projects:** [[Slide](https://claude.ai/public/artifacts/ccf0c3ff-3598-4bba-b580-c8776290f555)]
* **AI Project Ideas for IB Diploma:** [[Slide](https://claude.ai/public/artifacts/57dc60bc-4dc3-484f-ad36-9000860e6b24)]

### Backup Slides

* **Why Memory Helps:** [[Slide](https://claude.ai/public/artifacts/ed64cae3-2dc4-4d83-aeeb-12d6817bd36f)]
* **From Feed-Forward to Recurrent Neural Networks:** [[Slide](https://claude.ai/public/artifacts/64f71205-b6b0-4114-b871-79b1e2b46080)]


## Potential Todos & Future Enhancements

* **Read Parcour Layout from File(s):** Allow parcours to be defined in external files (e.g., JSON, CSV, or a custom format) instead of being hardcoded.
* **Parcour Layout Editor:** Create a simple graphical tool or interface to design and save new parcours.
* **Switch to PyTorch:** Explore migrating the model training and inference code from TensorFlow/Keras to PyTorch as an alternative deep learning framework.
* **More Sophisticated Sensor Data:** Include other sensor types (e.g., vehicle's own speed if it becomes variable, angle to the green square).
* **Advanced AI Techniques:** Experiment with Reinforcement Learning (e.g., Q-learning, Deep Q-Networks) where the agent learns by trial and error directly within the simulation.
* **Improved Graphics & UI:** Enhance the visual appeal and user interface of the simulation.
