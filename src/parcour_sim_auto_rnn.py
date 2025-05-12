import pygame
import math
import csv
import time
import sys
import random
import numpy as np
import tensorflow as tf
import os
from collections import deque # Use deque for efficient sequence management

# --- Constants ---
# (Keep constants like SCREEN_WIDTH, HEIGHT, FPS, Colors, Vehicle settings, etc.)
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900
FPS = 60
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
GRAY = (150, 150, 150)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255) # For Model I/O Box

VEHICLE_WIDTH = 10
VEHICLE_HEIGHT = 15
VEHICLE_SPEED = 30
VEHICLE_TURN_RATE = 45
SENSOR_MAX_RANGE = 300.0
SENSOR_ANGLES = [0, -30, -60, 30, 60]
SENSOR_NAMES = ["Fwd", "L30", "L60", "R30", "R60"]
ACTION_NAMES = ["P(Straight)", "P(Left)", "P(Right)"]

# --- AI Model & RNN Specific ---
MODEL_PATH = "vehicle_steering_model_rnn.keras" # <<< Use RNN model path
SEQUENCE_LENGTH = 10 # <<< MUST MATCH the sequence length used in training
NUM_FEATURES = len(SENSOR_ANGLES) # Should be 5

# Telemetry Settings
TELEMETRY_INTERVAL = 0.2 # seconds

# --- Parcour Layout Definitions ---
# (Keep layout definitions)
# --- Parcour Layout Definitions ---

# Layout 1: U-Shape
layout3_walls = [
    # Left wall
    (100, 100, 20, 700),
    # Bottom wall
    (100, 580, 600, 20),
    # Inner wall 1
    (180, 180, 20, 300), # Vertical part
    (180, 180, 200, 20), # Horizontal part
    (380, 180, 20, 150),
    # Right wall
    (680, 100, 20, 700),
     # Top wall
    (100, 100, 600, 20),
    # Inner wall 2
    (400, 300, 20, 300), # Vertical part
    (300, 500, 100, 20), # Horizontal part
]
layout3_start_pos = (150, 550)
layout3_start_angle = 0
layout3_destination = pygame.Rect(120, 120, 60, 60) # Top-left corner area

# Layout 2: S-Shape Track
layout4_walls = [
    # Left wall
    (100, 100, 20, 700),
    # Bottom wall
    (100, 580, 600, 20),
    # Inner wall 1
    (180, 180, 20, 300), # Vertical part
    (180, 180, 200, 20), # Horizontal part
    (380, 180, 20, 150),
    # Right wall
    (680, 100, 20, 700),
     # Top wall
    (100, 100, 600, 20),
    # Inner wall 2
    (400, 300, 20, 300), # Vertical part
    (300, 500, 100, 20), # Horizontal part
]
layout4_start_pos = (150, 550)
layout4_start_angle = 0
layout4_destination = pygame.Rect(120, 120, 60, 60) # Top-left corner area

layout1_walls = [
    # Outer Walls
    (50, 50, 700, 20),    # Top
    (50, 630, 700, 20),   # Bottom
    (50, 50, 20, 680),    # Top Left
    #(50, 380, 20, 270),   # Bottom Left
    (730, 50, 20, 680),   # Top Right
    #(730, 380, 20, 270),  # Bottom Right

    # Inner Walls creating S-shape path
    (150, 150, 500, 20),  # Top inner horizontal
    (150, 150, 20, 280),  # Left inner vertical
    (630, 280, 20, 280),  # Right inner vertical
    (250, 480, 500, 20),  # Bottom inner horizontal
    (250, 280, 20, 350),
    (280, 280, 20, 50),
    (470, 150, 20, 220),
    (280, 280, 100, 20),
]
layout1_start_pos = (100, 580) # Start bottom left, facing right
layout1_start_angle = 0
layout1_destination = pygame.Rect(650, 80, 60, 60) # Finish top right

layout2_walls = [
    # Outer Walls
    (50, 50, 700, 20),    # Top
    (50, 630, 700, 20),   # Bottom
    (50, 50, 20, 680),    # Top Left
    #(50, 380, 20, 270),   # Bottom Left
    (730, 50, 20, 680),   # Top Right
    #(730, 380, 20, 270),  # Bottom Right

    # Inner Walls creating S-shape path
    (150, 150, 500, 20),  # Top inner horizontal
    (150, 150, 20, 280),  # Left inner vertical
    (630, 280, 20, 280),  # Right inner vertical
    (250, 480, 500, 20),  # Bottom inner horizontal
    (250, 280, 20, 350),
    (280, 280, 20, 50),
    (470, 150, 20, 220),
    (280, 280, 100, 20),
]
layout2_start_pos = (100, 580) # Start bottom left, facing right
layout2_start_angle = 0
layout2_destination = pygame.Rect(650, 80, 60, 60) # Finish top right

# List of available layouts (This part should be correct already)
parcour_layouts = [
    {"walls": layout1_walls, "start_pos": layout1_start_pos, "start_angle": layout1_start_angle, "dest": layout1_destination},
    {"walls": layout2_walls, "start_pos": layout2_start_pos, "start_angle": layout2_start_angle, "dest": layout2_destination},
    # Add more layout dictionaries here if desired
]

# --- Classes ---
# (Vehicle and Wall classes remain mostly the same)
# Small potential optimization: store normalized sensors if needed often
class Vehicle(pygame.sprite.Sprite):
    # ... __init__ ...
    # ... turn, move, update_image_rotation, update, check_collision ...
    # ... get_sensor_readings, draw_sensors ...
    # (No changes needed inside Vehicle class for RNN specifically,
    #  but ensure get_sensor_readings is efficient)
    def __init__(self, x, y, start_angle=0):
        super().__init__()
        self.original_image = pygame.Surface([VEHICLE_WIDTH, VEHICLE_HEIGHT], pygame.SRCALPHA)
        self.original_image.fill(ORANGE)
        pygame.draw.polygon(self.original_image, YELLOW, [(VEHICLE_WIDTH // 2, 0), (0, VEHICLE_HEIGHT), (VEHICLE_WIDTH, VEHICLE_HEIGHT)])
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(0, 0)
        self.angle = start_angle
        self.speed = VEHICLE_SPEED
        self.turn_rate = VEHICLE_TURN_RATE
        self.sensor_readings = [SENSOR_MAX_RANGE] * len(SENSOR_ANGLES)

    def turn(self, direction, dt):
        if direction != 0:
            turn_amount = self.turn_rate * direction * dt
            self.angle = (self.angle + turn_amount) % 360

    def move(self, dt):
        rad_angle = math.radians(self.angle)
        self.velocity.x = self.speed * math.cos(rad_angle)
        self.velocity.y = self.speed * math.sin(rad_angle)
        self.position += self.velocity * dt
        self.rect.centerx = int(self.position.x)
        self.rect.centery = int(self.position.y)

    def update_image_rotation(self):
        self.image = pygame.transform.rotate(self.original_image, -self.angle - 90)
        self.rect = self.image.get_rect(center=self.rect.center)

    def update(self, dt):
        self.move(dt)
        self.update_image_rotation()

    def check_collision(self, walls_group):
        collided_walls = pygame.sprite.spritecollide(self, walls_group, False)
        return bool(collided_walls)

    def get_sensor_readings(self, walls):
        self.sensor_readings = []
        for sensor_angle_offset in SENSOR_ANGLES:
            sensor_abs_angle_deg = (self.angle + sensor_angle_offset) % 360
            sensor_abs_angle_rad = math.radians(sensor_abs_angle_deg)
            direction_vector = pygame.math.Vector2(math.cos(sensor_abs_angle_rad), math.sin(sensor_abs_angle_rad))
            start_point = self.position
            end_point_far = start_point + direction_vector * SENSOR_MAX_RANGE
            closest_dist = SENSOR_MAX_RANGE
            for wall in walls:
                try:
                    clipped_line = wall.rect.clipline(start_point, end_point_far)
                    if clipped_line:
                        intersect_start, intersect_end = clipped_line
                        p1 = pygame.math.Vector2(intersect_start)
                        p2 = pygame.math.Vector2(intersect_end)
                        dist1 = (p1 - start_point).length()
                        dist2 = (p2 - start_point).length()
                        current_intersect_dist = min(dist1, dist2)
                        if current_intersect_dist < closest_dist:
                            closest_dist = current_intersect_dist
                except TypeError:
                    pass
            self.sensor_readings.append(closest_dist)
        return self.sensor_readings

    def draw_sensors(self, screen):
         for i, sensor_angle_offset in enumerate(SENSOR_ANGLES):
            if i < len(self.sensor_readings): dist = self.sensor_readings[i]
            else: continue
            if dist < SENSOR_MAX_RANGE:
                sensor_abs_angle_deg = (self.angle + sensor_angle_offset) % 360
                sensor_abs_angle_rad = math.radians(sensor_abs_angle_deg)
                direction_vector = pygame.math.Vector2(math.cos(sensor_abs_angle_rad), math.sin(sensor_abs_angle_rad))
                end_point = self.position + direction_vector * dist
                pygame.draw.line(screen, RED, self.position, end_point, 1)

class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(GRAY)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)


# --- Game Setup Helper ---
# (setup_walls and start_new_game remain the same)
def setup_walls(walls_data, wall_group, all_sprites):
    # ... same ...
    for w in walls_data:
        wall = Wall(*w)
        wall_group.add(wall)
        all_sprites.add(wall)

def start_new_game(all_sprites_group, walls_sprite_group):
    # ... same ...
    all_sprites_group.empty()
    walls_sprite_group.empty()
    chosen_layout = random.choice(parcour_layouts)
    print(f"Starting new game with Layout {parcour_layouts.index(chosen_layout) + 1}")
    setup_walls(chosen_layout["walls"], walls_sprite_group, all_sprites_group)
    vehicle = Vehicle(*chosen_layout["start_pos"], start_angle=chosen_layout["start_angle"])
    all_sprites_group.add(vehicle)
    destination_rect = chosen_layout["dest"]
    return vehicle, destination_rect

# --- Main Game Function ---
def game_loop():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Autonomous Vehicle Simulation (RNN)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    info_font = pygame.font.Font(None, 24)
    io_font = pygame.font.Font(None, 22)

    # --- Load RNN Model ---
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Successfully loaded RNN model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}. Autonomous mode unavailable.")
    else:
        print(f"Model file not found at {MODEL_PATH}. Autonomous mode unavailable.")

    # --- Sprite Groups ---
    all_sprites = pygame.sprite.Group()
    walls_group = pygame.sprite.Group()

    # --- Initialize first game ---
    vehicle, destination_rect = start_new_game(all_sprites, walls_group)

    # --- RNN State Initialization ---
    # Use deque to store the sequence of recent sensor readings
    # Initialize with padding (e.g., zeros or initial reading repeated)
    initial_readings_raw = vehicle.get_sensor_readings(walls_group)
    initial_readings_norm = (np.clip(np.array(initial_readings_raw, dtype=np.float32), 0, SENSOR_MAX_RANGE) / SENSOR_MAX_RANGE)
    sensor_sequence = deque([initial_readings_norm] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    # sensor_sequence = deque([np.zeros(NUM_FEATURES)] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH) # Alternative: Zero padding


    # --- Game State ---
    running = True
    game_over = False
    won = False
    show_sensors = False
    autonomous_mode = False
    paused = False
    # Variables to store last AI data for display
    last_sensor_input_raw = initial_readings_raw # Store last raw sensor reading for telemetry/display if needed
    last_model_input_seq = np.array(sensor_sequence) # Store the sequence used for the last prediction
    last_model_output = None
    last_action_taken = 0

    # --- Telemetry ---
    # ... (telemetry setup remains the same) ...
    telemetry_data = []
    last_telemetry_time = time.time()
    telemetry_header = ['Timestamp', 'Sensor1_Fwd', 'Sensor2_L30', 'Sensor3_L60', 'Sensor4_R30', 'Sensor5_R60', 'Key_Left', 'Key_Right', 'Mode']
    telemetry_data.append(telemetry_header)

    # --- Game Loop ---
    while running:
        dt = clock.tick(FPS) / 1000.0
        if dt == 0 or dt > 0.1 : dt = 1/FPS

        # --- Event Handling ---
        for event in pygame.event.get():
            # ... (event handling for Quit, S, A, P remains the same) ...
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_s: show_sensors = not show_sensors
                if event.key == pygame.K_a:
                    if model: autonomous_mode = not autonomous_mode
                    else: print("Cannot toggle mode: Model not loaded.")
                if event.key == pygame.K_p:
                    paused = not paused
                    if not paused: last_telemetry_time = time.time()
                # Restart handling
                if (game_over or won) and event.key == pygame.K_r:
                    game_over = False; won = False; paused = False
                    telemetry_data = [telemetry_header]; last_telemetry_time = time.time()
                    vehicle, destination_rect = start_new_game(all_sprites, walls_group)
                    # <<< Re-initialize RNN state >>>
                    initial_readings_raw = vehicle.get_sensor_readings(walls_group)
                    initial_readings_norm = (np.clip(np.array(initial_readings_raw, dtype=np.float32), 0, SENSOR_MAX_RANGE) / SENSOR_MAX_RANGE)
                    sensor_sequence = deque([initial_readings_norm] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
                    last_sensor_input_raw = initial_readings_raw
                    last_model_input_seq = np.array(sensor_sequence)
                    last_model_output = None
                    last_action_taken = 0


        # --- Game Logic (Runs only if not paused) ---
        if not paused:
            keys_pressed = pygame.key.get_pressed()
            action_taken = 0
            turn_direction = 0

            if not game_over and not won:
                # Get current sensor readings (raw)
                current_sensor_values_raw = vehicle.get_sensor_readings(walls_group)
                last_sensor_input_raw = current_sensor_values_raw # Store for telemetry

                # Normalize current sensor readings
                current_sensor_norm = (np.clip(np.array(current_sensor_values_raw, dtype=np.float32), 0, SENSOR_MAX_RANGE) / SENSOR_MAX_RANGE)

                # <<< Update sensor sequence >>>
                sensor_sequence.append(current_sensor_norm)

                if autonomous_mode and model:
                    # --- Autonomous RNN Control ---
                    try:
                        # 1. Prepare model input from sequence deque
                        model_input_seq = np.array(sensor_sequence) # Shape (SEQUENCE_LENGTH, NUM_FEATURES)
                        # Reshape for model prediction (add batch dimension)
                        model_input_batch = np.reshape(model_input_seq, (1, SEQUENCE_LENGTH, NUM_FEATURES))

                        # 2. Predict
                        predictions = model.predict(model_input_batch, verbose=0)[0]

                        # 3. Choose action
                        action_taken = np.argmax(predictions)

                        # Store data for display
                        last_model_input_seq = model_input_seq # Store the sequence
                        last_model_output = predictions
                        last_action_taken = action_taken

                    except Exception as e:
                         print(f"Error during model prediction: {e}")
                         action_taken = 0
                         # Keep previous display data?

                elif not autonomous_mode:
                    # --- Manual Control ---
                     if keys_pressed[pygame.K_LEFT]: action_taken = 1
                     elif keys_pressed[pygame.K_RIGHT]: action_taken = 2
                     else: action_taken = 0
                     last_model_output = None # Clear AI data for display
                     last_action_taken = action_taken

                else: # Auto mode selected but no model
                    action_taken = 0
                    last_model_output = None
                    last_action_taken = action_taken

                # Determine turn direction
                if action_taken == 1: turn_direction = -1
                elif action_taken == 2: turn_direction = 1
                else: turn_direction = 0

                # --- Apply Turn & Update Position ---
                vehicle.turn(turn_direction, dt)
                vehicle.update(dt)

                # --- Collision / Win Check ---
                if vehicle.check_collision(walls_group): game_over = True
                if destination_rect.colliderect(vehicle.rect): won = True

                # --- Telemetry ---
                current_time = time.time()
                if current_time - last_telemetry_time >= TELEMETRY_INTERVAL:
                    # Use the raw sensor values from the start of this step
                    timestamp = current_time
                    key_left_for_telemetry = (action_taken == 1)
                    key_right_for_telemetry = (action_taken == 2)
                    mode_str = "Auto" if autonomous_mode else "Manual"
                    row_data = [timestamp] + last_sensor_input_raw + [key_left_for_telemetry, key_right_for_telemetry, mode_str]
                    telemetry_data.append(row_data)
                    last_telemetry_time = current_time

        # --- Drawing (Runs always) ---
        screen.fill(BLACK)
        all_sprites.draw(screen)
        pygame.draw.rect(screen, GREEN, destination_rect)
        if show_sensors:
            vehicle.draw_sensors(screen)

        # --- Draw UI Text ---
        # ... (Instructions and Mode Status drawing remains the same) ...
        sensor_instr = "[S] Show/Hide Sensors"
        mode_instr = "[A] Toggle Auto" if model else "[A] Auto N/A"
        pause_instr = "[P] Pause/Resume"
        restart_instr = "[R] Restart"
        instructions_text = f"{sensor_instr} | {mode_instr} | {pause_instr} | {restart_instr}"
        instr_surf = info_font.render(instructions_text, True, WHITE)
        screen.blit(instr_surf, (10, 10))

        mode_status = "Mode: Autonomous (RNN)" if autonomous_mode else "Mode: Manual"
        if not model and autonomous_mode: mode_status += " (Model Error!)"
        mode_surf = info_font.render(mode_status, True, WHITE)
        screen.blit(mode_surf, (SCREEN_WIDTH - mode_surf.get_width() - 10, 10))


        # --- Draw Model I/O Display (Right Side) ---
        should_show_io = (autonomous_mode and show_sensors) or paused
        # Check if we have RNN output to display
        if should_show_io and last_model_output is not None:
            io_display_x = SCREEN_WIDTH - 220
            io_display_y = 40
            box_width = 210
            box_height = 220 # Adjusted height for less input display detail
            line_height = 18

            io_box_rect = pygame.Rect(io_display_x - 5, io_display_y - 5, box_width, box_height)
            pygame.draw.rect(screen, BLACK, io_box_rect)
            pygame.draw.rect(screen, CYAN, io_box_rect, 1)

            header_surf = info_font.render("--- RNN Model I/O ---", True, YELLOW)
            screen.blit(header_surf, (io_display_x + (box_width - header_surf.get_width()) // 2, io_display_y))
            io_display_y += line_height + 5

            # Input - Just show the *latest* normalized sensor input from the sequence
            input_header_surf = io_font.render("Latest Norm. Input:", True, WHITE)
            screen.blit(input_header_surf, (io_display_x, io_display_y))
            io_display_y += line_height
            latest_input_norm = sensor_sequence[-1] # Get the last element from deque
            for i, val in enumerate(latest_input_norm):
                txt = f" S {SENSOR_NAMES[i]}: {val:.3f}"
                surf = io_font.render(txt, True, WHITE)
                screen.blit(surf, (io_display_x + 5, io_display_y))
                io_display_y += line_height

            io_display_y += 3

            # Outputs (Probabilities)
            output_header_surf = io_font.render("Output Probs:", True, WHITE)
            screen.blit(output_header_surf, (io_display_x, io_display_y))
            io_display_y += line_height
            for i, val in enumerate(last_model_output):
                is_selected = (i == last_action_taken)
                color = GREEN if is_selected else WHITE
                txt = f" {ACTION_NAMES[i]}: {val:.3f}"
                surf = io_font.render(txt, True, color)
                screen.blit(surf, (io_display_x + 5, io_display_y))
                io_display_y += line_height

        # --- Draw Pause/Game Over/Win Messages (On Top) ---
        # ... (Drawing Paused, Game Over, Won text remains the same) ...
        if paused:
            # No overlay for visibility
            pause_surf = font.render("PAUSED", True, YELLOW)
            pause_rect = pause_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(pause_surf, pause_rect)
        if game_over:
             text_surface = font.render("GAME OVER! Press R to Restart", True, RED)
             text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
             screen.blit(text_surface, text_rect)
        elif won:
             text_surface = font.render("YOU WON! Press R to Restart", True, GREEN)
             text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
             screen.blit(text_surface, text_rect)


        pygame.display.flip()

    # --- Game End ---
    # ... (Telemetry saving logic remains the same) ...
    print("\nSimulation Ended.")
    if won:
        # ... (save telemetry CSV) ...
        if telemetry_data and len(telemetry_data) > 1:
            mode_suffix = "AUTORNN" if autonomous_mode else "MANUAL" # Indicate RNN
            filename = f"vehicle_telemetry_SUCCESS_{mode_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            # ... (rest of saving logic) ...
            try:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(telemetry_data)
                print(f"Telemetry data for SUCCESSFUL run saved to {filename}")
            except Exception as e:
                print(f"Error saving telemetry data: {e}")
        else: print("No telemetry data collected for the successful run.")
    else:
        reason = "collision" if game_over else "user exit"
        print(f"Telemetry data NOT saved (Game ended due to {reason}).")

    pygame.quit()
    sys.exit()


# --- Run the Game ---
if __name__ == '__main__':
    game_loop()
