import pygame
import math
import csv
import time
import sys # To cleanly exit
import random # Added for random parcour selection
import numpy as np # For array manipulation and argmax
import tensorflow as tf # To load and use the Keras model
import os # To check if model file exists

# --- Constants ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (150, 150, 150)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255) # For Model I/O Box

# Vehicle Settings
VEHICLE_WIDTH = 10
VEHICLE_HEIGHT = 15
VEHICLE_SPEED = 30  # Pixels per second
VEHICLE_TURN_RATE = 45 # Degrees per second
SENSOR_MAX_RANGE = 300.0 # Ensure this matches training preprocessing (use float)

# Sensor angles relative to vehicle forward direction (0 degrees)
SENSOR_ANGLES = [0, -30, -60, 30, 60]
SENSOR_NAMES = ["Fwd", "L30", "L60", "R30", "R60"] # For display

# Telemetry Settings
TELEMETRY_INTERVAL = 0.2 # seconds

# --- AI Model ---
MODEL_PATH = "vehicle_steering_model4.keras"
ACTION_NAMES = ["P(Straight)", "P(Left)", "P(Right)"] # For display

# --- Parcour Layout Definitions ---
# (Keep the layout definitions from the previous version)
# Layout 1: U-Shape
layout1_walls = [
    (100, 100, 20, 500), (100, 580, 400, 20), (200, 200, 20, 300),
    (200, 200, 200, 20), (480, 100, 20, 500), (100, 100, 400, 20),
    (300, 300, 20, 300), (300, 500, 100, 20),
]
layout1_start_pos = (150, 550)
layout1_start_angle = 0
layout1_destination = pygame.Rect(120, 120, 60, 60)

# Layout 2: S-Shape Track
layout2_walls = [
#    (50, 50, 700, 20), (50, 630, 700, 20), (50, 50, 20, 250), (50, 380, 20, 270),
#    (730, 50, 20, 250), (730, 380, 20, 270), (150, 150, 500, 20), (150, 150, 20, 250),
#    (630, 250, 20, 250), (250, 480, 500, 20),
    (100, 100, 20, 500), (100, 580, 400, 20), (200, 200, 20, 300),
    (200, 200, 200, 20), (480, 100, 20, 500), (100, 100, 400, 20),
    (300, 300, 20, 300), (300, 500, 100, 20),
]
layout2_start_pos = (150, 550)
layout2_start_angle = 0
layout2_destination = pygame.Rect(120, 120, 60, 60)
#layout2_start_pos = (100, 580)
#layout2_start_angle = 0
#layout2_destination = pygame.Rect(650, 80, 60, 60)

parcour_layouts = [
    {"walls": layout1_walls, "start_pos": layout1_start_pos, "start_angle": layout1_start_angle, "dest": layout1_destination},
    {"walls": layout2_walls, "start_pos": layout2_start_pos, "start_angle": layout2_start_angle, "dest": layout2_destination},
]


# --- Classes ---
# (Vehicle and Wall classes remain the same as the previous 'auto' version)
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, x, y, start_angle=0):
        super().__init__()
        self.original_image = pygame.Surface([VEHICLE_WIDTH, VEHICLE_HEIGHT], pygame.SRCALPHA)
        self.original_image.fill(BLUE)
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
            # Use the latest sensor readings stored in the instance
            if i < len(self.sensor_readings):
                 dist = self.sensor_readings[i]
            else: continue # Skip if sensor readings aren't fully populated yet

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
    for w in walls_data:
        wall = Wall(*w)
        wall_group.add(wall)
        all_sprites.add(wall)

def start_new_game(all_sprites_group, walls_sprite_group):
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
    pygame.display.set_caption("Autonomous Vehicle Simulation w/ Pause & Info")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36) # Font for messages
    info_font = pygame.font.Font(None, 24) # Smaller font for UI text & Model I/O
    io_font = pygame.font.Font(None, 22) # Even smaller for I/O details

    # --- Load AI Model ---
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}. Autonomous mode unavailable.")
    else:
        print(f"Model file not found at {MODEL_PATH}. Autonomous mode unavailable.")

    # --- Sprite Groups ---
    all_sprites = pygame.sprite.Group()
    walls_group = pygame.sprite.Group()

    # --- Initialize first game ---
    vehicle, destination_rect = start_new_game(all_sprites, walls_group)

    # --- Game State ---
    running = True
    game_over = False
    won = False
    show_sensors = False
    autonomous_mode = False
    paused = False # <<< New paused state
    # Variables to store last AI data for display (even when paused)
    last_sensor_input = vehicle.get_sensor_readings(walls_group) # Get initial readings
    last_model_input = None # Normalized inputs
    last_model_output = None # Prediction probabilities
    last_action_taken = 0 # Predicted action index

    # --- Telemetry ---
    telemetry_data = []
    last_telemetry_time = time.time()
    telemetry_header = ['Timestamp', 'Sensor1_Fwd', 'Sensor2_L30', 'Sensor3_L60', 'Sensor4_R30', 'Sensor5_R60', 'Key_Left', 'Key_Right', 'Mode']
    telemetry_data.append(telemetry_header)

    # --- Game Loop ---
    while running:
        dt = clock.tick(FPS) / 1000.0
        # Prevent issues with dt=0 or large dt after pause/lag
        if dt == 0 or dt > 0.1 : dt = 1/FPS # Use nominal dt if actual is weird

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Allow these toggles even when paused
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_s:
                    show_sensors = not show_sensors
                if event.key == pygame.K_a:
                    if model: autonomous_mode = not autonomous_mode
                    else: print("Cannot toggle mode: Model not loaded.")
                if event.key == pygame.K_p: # <<< Toggle Pause
                    paused = not paused
                    print(f"Game Paused: {paused}")
                    # If unpausing, reset last telemetry time to avoid large catch-up
                    if not paused: last_telemetry_time = time.time()

                # Only allow restart if game is over/won
                if (game_over or won) and event.key == pygame.K_r:
                    game_over = False
                    won = False
                    paused = False # Ensure restart unpauses
                    # autonomous_mode = False # Optional: reset mode on restart
                    telemetry_data = [telemetry_header]
                    last_telemetry_time = time.time()
                    vehicle, destination_rect = start_new_game(all_sprites, walls_group)
                    # Reset last sensor/model data on restart
                    last_sensor_input = vehicle.get_sensor_readings(walls_group)
                    last_model_input = None
                    last_model_output = None
                    last_action_taken = 0

        # --- Game Logic (Runs only if not paused) ---
        if not paused:
            keys_pressed = pygame.key.get_pressed() # Get keyboard state

            action_taken = 0 # 0: Straight, 1: Left, 2: Right
            turn_direction = 0 # -1 Left, 1 Right, 0 Straight

            if not game_over and not won:
                # Get current sensor readings
                current_sensor_values = vehicle.get_sensor_readings(walls_group)
                last_sensor_input = current_sensor_values # Store raw sensor readings

                if autonomous_mode and model:
                    # --- Autonomous Control ---
                    try:
                        # 1. Preprocess sensor data
                        sensor_array = np.array(current_sensor_values, dtype=np.float32)
                        sensor_array = np.clip(sensor_array, 0, SENSOR_MAX_RANGE)
                        normalized_sensors = sensor_array / SENSOR_MAX_RANGE
                        model_input = np.reshape(normalized_sensors, (1, 5))

                        # 2. Predict
                        predictions = model.predict(model_input, verbose=0)[0]

                        # 3. Choose action
                        action_taken = np.argmax(predictions)

                        # Store data for display
                        last_model_input = normalized_sensors
                        last_model_output = predictions
                        last_action_taken = action_taken

                    except Exception as e:
                         print(f"Error during model prediction: {e}")
                         action_taken = 0
                         # Keep previous display data on error

                elif not autonomous_mode:
                    # --- Manual Control ---
                     if keys_pressed[pygame.K_LEFT]: action_taken = 1
                     elif keys_pressed[pygame.K_RIGHT]: action_taken = 2
                     else: action_taken = 0
                     # Clear AI display data if switching to manual
                     last_model_input = None
                     last_model_output = None
                     last_action_taken = action_taken # Keep track even if manual

                else: # Auto mode selected but no model
                    action_taken = 0
                    last_model_input = None
                    last_model_output = None
                    last_action_taken = action_taken

                # Determine turn direction from action_taken
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
                    timestamp = current_time
                    key_left_for_telemetry = (action_taken == 1)
                    key_right_for_telemetry = (action_taken == 2)
                    mode_str = "Auto" if autonomous_mode else "Manual"
                    row_data = [timestamp] + current_sensor_values + [key_left_for_telemetry, key_right_for_telemetry, mode_str]
                    telemetry_data.append(row_data)
                    last_telemetry_time = current_time
        # End of "if not paused" block

        # --- Drawing (Runs always) ---
        screen.fill(BLACK)
        all_sprites.draw(screen)
        pygame.draw.rect(screen, GREEN, destination_rect)

        # Draw sensor lines if enabled
        if show_sensors:
            vehicle.draw_sensors(screen) # Uses vehicle's current sensor_readings attribute

        # --- Draw UI Text ---
        # Instructions (Top Left)
        sensor_instr = "[S] Show/Hide Sensors"
        mode_instr = "[A] Toggle Auto" if model else "[A] Auto N/A"
        pause_instr = "[P] Pause/Resume"
        restart_instr = "[R] Restart"
        instructions_text = f"{sensor_instr} | {mode_instr} | {pause_instr} | {restart_instr}"
        instr_surf = info_font.render(instructions_text, True, WHITE)
        screen.blit(instr_surf, (10, 10))

        # Mode Status (Top Right)
        mode_status = "Mode: Autonomous" if autonomous_mode else "Mode: Manual"
        if not model and autonomous_mode: mode_status += " (Model Error!)"
        mode_surf = info_font.render(mode_status, True, WHITE)
        screen.blit(mode_surf, (SCREEN_WIDTH - mode_surf.get_width() - 10, 10))

        # --- Draw Model I/O Display (Right Side) ---
        # Display if (Auto mode AND Show Sensors are ON) OR if game is Paused
        should_show_io = (autonomous_mode and show_sensors) or paused
        if should_show_io and last_model_input is not None and last_model_output is not None:
            io_display_x = SCREEN_WIDTH - 220 # X position for the display box
            io_display_y = 40 # Starting Y position
            box_width = 210
            box_height = 210 # Adjust as needed
            line_height = 18

            # Draw background box
            io_box_rect = pygame.Rect(io_display_x - 5, io_display_y - 5, box_width, box_height)
            pygame.draw.rect(screen, BLACK, io_box_rect) # Background
            pygame.draw.rect(screen, CYAN, io_box_rect, 1) # Border

            # Header
            header_surf = info_font.render("--- Model I/O ---", True, YELLOW)
            screen.blit(header_surf, (io_display_x + (box_width - header_surf.get_width()) // 2, io_display_y))
            io_display_y += line_height + 5

            # Inputs
            input_header_surf = io_font.render("Inputs (Normalized):", True, WHITE)
            screen.blit(input_header_surf, (io_display_x, io_display_y))
            io_display_y += line_height
            for i, val in enumerate(last_model_input):
                txt = f" Sensor {SENSOR_NAMES[i]}: {val:.3f}" # More precision
                surf = io_font.render(txt, True, WHITE)
                screen.blit(surf, (io_display_x + 5, io_display_y))
                io_display_y += line_height

            io_display_y += 3 # Spacer

            # Outputs
            output_header_surf = io_font.render("Outputs (Probabilities):", True, WHITE)
            screen.blit(output_header_surf, (io_display_x, io_display_y))
            io_display_y += line_height
            for i, val in enumerate(last_model_output):
                is_selected = (i == last_action_taken) # Use last action taken
                color = GREEN if is_selected else WHITE # Highlight selected action
                txt = f" {ACTION_NAMES[i]}: {val:.3f}"
                surf = io_font.render(txt, True, color)
                screen.blit(surf, (io_display_x + 5, io_display_y))
                io_display_y += line_height

        # --- Draw Pause/Game Over/Win Messages (On Top) ---
        if paused:
            # Semi-transparent overlay
            # overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            # overlay.fill((0, 0, 0, 150)) # Dark overlay
            # screen.blit(overlay, (0, 0))
            # Pause Text
            pause_surf = font.render("PAUSED", True, YELLOW)
            pause_rect = pause_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(pause_surf, pause_rect)

        if game_over:
             text_surface = font.render("GAME OVER! Press R to Restart", True, RED)
             text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40)) # Shift down if paused text is there
             screen.blit(text_surface, text_rect)
        elif won:
             text_surface = font.render("YOU WON! Press R to Restart", True, GREEN)
             text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40)) # Shift down
             screen.blit(text_surface, text_rect)

        pygame.display.flip()

    # --- Game End ---
    print("\nSimulation Ended.")
    # (Telemetry saving logic remains the same - only saves if 'won')
    if won:
        if telemetry_data and len(telemetry_data) > 1:
            mode_suffix = "AUTO" if autonomous_mode else "MANUAL"
            filename = f"vehicle_telemetry_SUCCESS_{mode_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            try:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(telemetry_data)
                print(f"Telemetry data for SUCCESSFUL run saved to {filename}")
            except Exception as e:
                print(f"Error saving telemetry data: {e}")
        else:
            print("No telemetry data collected for the successful run.")
    else:
        reason = "collision" if game_over else "user exit"
        print(f"Telemetry data NOT saved (Game ended due to {reason}).")


    pygame.quit()
    sys.exit()

# --- Run the Game ---
if __name__ == '__main__':
    game_loop()
