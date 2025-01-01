# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# Code Comprehensively Altered and Expanded By: Nadan Liddar

import math
import sys
import tkinter as tk
import threading
import random
import neat
import pygame
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser

# Constraints
WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit

TRACK_NUM = 15

# Global Variables
current_generation = 0  # Generation Counter
# Added by Nadan
testing_tracklist = []
reactive_tracklist = []
training_tracklist = []
epochs = 3
train_perc = 0.8
do_shuffle = 0
do_testing = 0
do_independent = 0
do_bespoke_tests = 0
seed = 34242
population = None


# Existing Car class unmodified except where specified
class Car:
    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load(
            "../assets/car.png"
        ).convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # self.position = [690, 740] # Starting Position
        self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2,
        ]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

        self.counter = 0  # Used for reactive decision-making

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + degree))) * length
        )
        y = int(
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + degree))) * length
        )

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + degree))) * length
            )
            y = int(
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + degree))) * length
            )

        # Calculate Distance To Border And Append To Radars List
        dist = int(
            math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        )
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2,
        ]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length,
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length,
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length,
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length,
        ]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    # Reactive Agent that prioritises survival. (Author: Nadan)
    def tortoise_choice(self):
        # Returns an array of radar distances
        # 0 -> most left, 1 -> left, 2 -> center, 3 -> right, 4 -> most right
        distance = self.get_data()

        # Increment counter
        self.counter += 1

        # Handle immediate collision avoidance
        fear_limit = 0
        for i in range(1, 4):
            if distance[i] <= fear_limit:
                if distance[1] < distance[3]:
                    return 0
                else:
                    return 1

        # Handle slowing down 1 (due to narrowing track) (only perform once every three times)
        fear_limit = 1
        if (distance[0] + distance[4]) <= (fear_limit * 2) and self.counter % 3 == 0:
            return 2

        # Handle incoming collision avoidance
        fear_limit = 1
        for i in range(1, 4):
            if distance[i] <= fear_limit:
                if distance[1] < distance[3]:
                    return 0
                else:
                    return 1

        # Handle slowing down 2 (due to narrowing track) (only perform once every three times)
        fear_limit = 1
        if (distance[0] + distance[4]) <= (fear_limit * 2) and self.counter % 3 == 0:
            return 2

        # Else anticipate any collisions
        if distance[1] < distance[3]:
            return 0
        else:
            return 1

    # Reactive Agent that prioritises speed and performance (Author: Nadan)
    def hare_choice(self):
        # Returns an array of radar distances
        # 0 -> most left, 1 -> left, 2 -> center, 3 -> right, 4 -> most right
        distance = self.get_data()

        # Increment counter
        self.counter += 1

        # Speed up at track start
        if self.counter < 3:
            return 3

        # Handle incoming collision avoidance
        fear_limit = 5
        for i in range(1, 4):
            if distance[i] <= fear_limit:
                if distance[1] < distance[3]:
                    return 0
                else:
                    return 1

        # Handle speeding up (due to widening track)
        fear_limit = 2
        if (distance[0] + distance[4]) >= (fear_limit * 2):
            return 3

        # Else anticipate incoming collision
        if distance[1] < distance[3]:
            return 0
        else:
            return 1


# Runs simulation for reactive agents (Author: Nadan - Based on Existing Code)
def run_reactive_simulation(do_tortoise):
    print("RUN REACTIVE SIMULATION")
    global current_generation, reactive_tracklist, do_shuffle, do_testing, train_perc

    # Create an array storing the distance travelled by the car
    car_distance = 0

    # Increment Generation
    current_generation += 1

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.WINDOWMAXIMIZED)

    # Initialise Car
    car = Car()

    # Clock Settings
    # Font Settings
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)

    # ---------- Select Track ---------- #
    # ---------- Run Every Track
    map_name = "../assets/maps/map{}.png".format(
        reactive_tracklist[current_generation - 1]
    )

    # Load Map
    game_map = pygame.image.load(map_name).convert()  # Convert Speeds Up A Lot

    # Counter To Limit Time
    counter = 0

    # Game Loop
    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # Get the cars action
        if do_tortoise:
            choice = car.tortoise_choice()
        else:
            choice = car.hare_choice()

        # Perform the cars action
        if choice == 0:
            car.angle += 10  # Left
        elif choice == 1:
            car.angle -= 10  # Right
        elif choice == 2:
            if (car.speed - 2) >= 12:
                car.speed -= 2  # Slow Down
        else:
            car.speed += 2  # Speed Up

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        if car.is_alive():
            still_alive += 1
            car.update(game_map)
            car_distance += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            break

        # Draw Map And Car
        screen.blit(game_map, (0, 0))
        if car.is_alive():
            car.draw(screen)

        # Display Info
        text = generation_font.render(
            "Stage: " + str(current_generation), True, (0, 0, 0)
        )
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(180)  # 180 = 60 FPS

    # ---------- Data Recording (Simulation Ended) ---------- #
    # (Author: Nadan)
    # Convert car distance to an array for file write
    car_distances = [car_distance]

    # Write data to either tortoise or hare datasets
    if do_tortoise:
        # Write the distance data to the tortoise CSV file
        with open("../data/tortoise_fitness.csv", "a", newline="") as csv_file:
            # Create CSV writer object
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(car_distances)
    else:
        # Write the distance data to the hare CSV file
        with open("../data/hare_fitness.csv", "a", newline="") as csv_file:
            # Create CSV writer object
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(car_distances)


# Simulates the NEAT agents for training, testing and independent training
# (Contributor: Nadan - Modified and Extended)
def run_simulation(genomes, config):
    print("RUN TRAINING SIMULATION")
    global current_generation, reactive_tracklist, training_tracklist, testing_tracklist, do_shuffle, do_testing

    # Increment Generation
    current_generation += 1

    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.WINDOWMAXIMIZED)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    # Font Settings
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)

    # ---------- Select Track ---------- #
    # (Author: Nadan)
    # ---------- Training
    if do_testing == 0:
        # ---------- Shuffle Tracks
        if do_shuffle == 1:
            map_name = "../assets/maps/map{}.png".format(
                training_tracklist[
                    math.ceil(current_generation % len(training_tracklist))
                ]
            )

        # ---------- Sequential Tracks
        else:
            map_name = "../assets/maps/map{}.png".format(
                training_tracklist[math.ceil(current_generation / epochs) - 1]
            )
    # ---------- Testing
    else:
        # ---------- Sequential Tracks
        map_name = "../assets/maps/map{}.png".format(
            testing_tracklist[current_generation - 1]
        )

    # ---------- Independent
    if do_independent != 0:
        map_name = "../assets/maps/map{}.png".format(
            reactive_tracklist[do_independent - 1]
        )

    # Load Map
    game_map = pygame.image.load(map_name).convert()  # Convert Speeds Up A Lot

    # Simple Counter To Limit Time
    counter = 0

    # Game Loop
    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Action It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if (car.speed - 2) >= 12:
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render(
            "Generation: " + str(current_generation), True, (0, 0, 0)
        )
        if do_testing == 1:
            text = generation_font.render(
                "Testing Generation: " + str(current_generation), True, (0, 0, 0)
            )

        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(180)  # 180 = 60 FPS

    # ---------- Data Recording (Simulation Ended) ---------- #
    # (Author: Nadan)
    # Create an array storing the distance travelled by each car each generation
    car_distances = []
    for i, car in enumerate(cars):
        car_distances.append(genomes[i][1].fitness)

    # Write data to independent, training or testing datasets
    if do_independent != 0 and current_generation == epochs:
        # Write the distance data to the independent CSV file (once training is finished)
        with open("../data/independent_fitness.csv", "a", newline="") as csv_file:
            # Create CSV writer object
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(car_distances)
    elif do_testing == 0:
        # Write the distance data to the training CSV file
        with open("../data/training_fitness.csv", "a", newline="") as csv_file:
            # Create CSV writer object
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(car_distances)
    else:
        # Write the distance data to the testing CSV file
        with open("../data/testing_fitness.csv", "a", newline="") as csv_file:
            # Create CSV writer object
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(car_distances)


# Creates NEAT population and calls run_simulation for training
# (Author: Nadan - Uses Some Existing Code)
def simulate_training():
    global training_tracklist, testing_tracklist, population, current_generation

    # Reset Generation Count
    current_generation = 0

    # Load Config
    config_path = "../config/config.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Clear the training CSV file
    with open("../data/training_fitness.csv", "w", newline="") as csv_file:
        pass

    # Run Simulation for the number of epochs specified
    population.run(run_simulation, epochs * len(training_tracklist))
    pygame.quit()
    print("SIMULATED TRAINING")


# Creates NEAT population and calls run_simulation for independent training
# (Author: Nadan - Uses Some Existing Code)
def simulate_independent():
    global reactive_tracklist, train_perc, do_independent, current_generation, do_shuffle

    # Reset Generation Count
    current_generation = 0

    # Do not shuffle so tracks are trained sequentially
    do_shuffle = 0

    # All tracks need to be trained so set train perc to 1
    train_perc = 1

    # Clear the independent training CSV file
    with open("../data/independent_fitness.csv", "w", newline="") as csv_file:
        pass

    # Run separate training for each track
    for i in range(len(reactive_tracklist)):
        # Reset Generation Count
        current_generation = 0

        # Update do_independent flag for data recording and map selection
        do_independent += 1

        # Load Config
        config_path = "../config/config.txt"
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        # Create Population And Add Reporters
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        # Run Simulation for the number of epochs
        population.run(run_simulation, epochs)
        pygame.quit()

    do_independent = 0
    print("INDEPENDENT TRAINING")


# Creates NEAT population and calls run_simulation for testing
# (Author: Nadan)
def simulate_testing():
    global testing_tracklist, population, do_testing, current_generation

    # Ensure simulation runs in testing mode
    # Reset Generation Count
    do_testing = 1
    current_generation = 0

    # Exit if NEAT population has not been trained
    if population is None:
        print("POPULATION EMPTY")
        return

    # Clear the testing CSV file
    with open("../data/testing_fitness.csv", "w", newline="") as csv_file:
        pass

    # Run Simulation for each testing track
    population.run(run_simulation, len(testing_tracklist))
    pygame.quit()

    # Reset testing flag
    do_testing = 0
    print("SIMULATED TESTING")


# Calls run_reactive_simulation for Tortoise agent
# (Author: Nadan)
def simulate_tortoise():
    global reactive_tracklist, do_testing, train_perc, current_generation

    # Reset Generation Count
    current_generation = 0

    # Clear the tortoise_fitness CSV file
    with open("../data/tortoise_fitness.csv", "w", newline="") as csv_file:
        pass

    # Run Simulation once for each track
    for i in range(1, len(reactive_tracklist) + 1):
        run_reactive_simulation(True)

    pygame.quit()
    print("SIMULATED TORTOISE")


# Calls run_reactive_simulation for Hare agent
# (Author: Nadan)
def simulate_hare():
    global reactive_tracklist, do_testing, train_perc, current_generation

    # Reset Generation Count
    current_generation = 0

    # Clear the hare_fitness CSV file
    with open("../data/hare_fitness.csv", "w", newline="") as csv_file:
        pass

    # Run Simulation once for each track
    for i in range(1, len(reactive_tracklist) + 1):
        run_reactive_simulation(False)

    pygame.quit()
    print("SIMULATED HARE")


# Handles start training onclick behaviour
# (Author: Nadan)
def start_train_onclick():
    simulation_thread = threading.Thread(target=simulate_training)
    simulation_thread.start()


# Handles start testing onclick behaviour
# (Author: Nadan)
def start_test_onclick():
    simulation_thread = threading.Thread(target=simulate_testing)
    simulation_thread.start()


# Handles start tortoise onclick behaviour
# (Author: Nadan)
def start_tortoise_onclick():
    simulation_thread = threading.Thread(target=simulate_tortoise)
    simulation_thread.start()


# Handles start hare onclick behaviour
# (Author: Nadan)
def start_hare_onclick():
    simulation_thread = threading.Thread(target=simulate_hare)
    simulation_thread.start()


# Handles independent training onclick behaviour
# (Author: Nadan)
def independent_train_onclick():
    simulation_thread = threading.Thread(target=simulate_independent)
    simulation_thread.start()


# Handles training progress onclick behaviour
# (Author: Nadan)
def training_progress_onclick():
    global epochs

    # Read the data and store in dataframe
    training_data = read_from_csv("../data/training_fitness.csv")
    df = pd.DataFrame(training_data)
    # Start index at 1
    df.index += 1

    # Limit anomalous data points
    df[df > 1000000] = 1000000

    # Calculate the average distance travelled for each generation
    avg_distance = df.mean(axis=1)

    # Calculate the max distance travelled for each generation
    max_distance = df.max(axis=1)

    # Calculate the average best fitness achieved over each epoch
    # Get max distances for each epoch
    max_distance_per_epoch = np.array_split(max_distance, epochs)

    # Store the number of generations passed for the current epoch
    generations_per_epoch = 0
    # Array to store each generation where a full epoch was completed
    generations_per_epoch_list = []

    # Array to store the average best fitness values for each epoch
    epoch_avg_best = []
    for epoch_max_distances in max_distance_per_epoch:
        # Calculate and store the average best fitness
        epoch_avg_best.append(np.mean(epoch_max_distances))
        # Calculate and store the number of generations passed to complete this epoch
        generations_per_epoch += len(epoch_max_distances)
        generations_per_epoch_list.append(generations_per_epoch)

    # Store average best fitness indexed by generations per epoch
    epoch_avg_best_series = pd.Series(epoch_avg_best, index=generations_per_epoch_list)

    # Create a dataframe with average and max distance values
    distances_df = pd.DataFrame(
        {
            "Average Distance": avg_distance,
            "Best Distance": max_distance,
            "Average Best Distance per Epoch": epoch_avg_best_series,
        }
    )

    # Interpolate missing values in epoch_avg_best_series to get a steady line
    distances_df["Average Best Distance per Epoch"] = distances_df[
        "Average Best Distance per Epoch"
    ].interpolate()

    # Plot a line graph showing average and max distance value over generations
    plt.figure(figsize=(12, 8))
    plt.plot(
        distances_df.index,
        distances_df["Average Distance"],
        label="Average Distance",
        linestyle="dashed",
    )
    # plt.plot(distances_df.index, distances_df['Best Distance'], label='Best Distance', linestyle='dashed')

    # Plot line showing the Average Best Distance per Epoch
    plt.plot(
        distances_df.index,
        distances_df["Average Best Distance per Epoch"],
        label="Average Best Distance per Epoch",
        linewidth=2,
        color="orange",
    )

    # Plot markers showing the Average Best Distance per Epoch when each epoch was completed
    for generation, distance in epoch_avg_best_series.items():
        plt.plot(generation, distance, marker="o", markersize=10, color="orange")

    plt.xlabel("Generation")
    plt.ylabel("Distance Travelled (Fitness)")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()


# Handles testing results button onclick behaviour
# (Author: Nadan)
def results_testing_onclick():
    # Read the data and store in dataframe
    training_data = read_from_csv("../data/testing_fitness.csv")
    df = pd.DataFrame(training_data)

    # Limit anomalous data points
    df[df > 1000000] = 1000000

    # Calculate the max distance travelled for each generation
    max_distance = df.max(axis=1)

    # Create a dataframe with max distance values
    distances_df = pd.DataFrame({"Best Distance": max_distance})

    distances_df.plot(kind="bar", figsize=(12, 8), color="lightblue")
    plt.title("Best Distance Achieved by NEAT AI")
    plt.xlabel("Track")
    plt.ylabel("Distance Travelled (Fitness)")
    plt.xticks(range(len(testing_tracklist)), testing_tracklist)
    # plt.ylim(0, 600000)
    plt.show()


# Handles independent testing results onlick behaviour
# (Author: Nadan)
def results_independent():
    # Read the data and store in dataframe
    independent_data = read_from_csv("../data/independent_fitness.csv")
    df = pd.DataFrame(independent_data)

    # Limit anomalous data points
    df[df > 1000000] = 1000000

    # Calculate the max distance travelled for each generation
    max_distance = df.max(axis=1)

    # Create a dataframe with max distance values
    distances_df = pd.DataFrame({"Best Distance": max_distance})

    distances_df.plot(kind="bar", figsize=(12, 8), color="Blue")
    plt.title("Best Distance Achieved by NEAT AI After Independent Stage Training")
    plt.xlabel("Track")
    plt.ylabel("Distance Travelled (Fitness)")
    plt.xticks(range(len(reactive_tracklist)), reactive_tracklist)
    # plt.ylim(0, 600000)
    plt.show()


# Handles tortoise results button onclick behaviour
# (Author: Nadan)
def results_tortoise_onclick():
    # Read the data and store in dataframe
    tortoise_data = read_from_csv("../data/tortoise_fitness.csv")
    df = pd.DataFrame(tortoise_data, columns=["Best Distance"])

    # Limit anomalous data points
    df[df > 1000000] = 1000000

    df.plot(kind="bar", figsize=(12, 8), color="lightgreen")
    plt.title("Best Distance Achieved by Tortoise Reactive Agent")
    plt.xlabel("Track")
    plt.ylabel("Distance Travelled (Fitness)")
    plt.xticks(range(len(reactive_tracklist)), reactive_tracklist)
    # plt.ylim(0, 600000)
    plt.show()


# Handles tortoise results button onclick behaviour
# (Author: Nadan)
def results_hare_onclick():
    # Read the data and store in dataframe
    hare_data = read_from_csv("../data/hare_fitness.csv")
    df = pd.DataFrame(hare_data, columns=["Best Distance"])

    # Limit anomalous data points
    df[df > 1000000] = 1000000

    df.plot(kind="bar", figsize=(12, 8), color="orange")
    plt.title("Best Distance Achieved by Hare Reactive Agent")
    plt.xlabel("Track")
    plt.ylabel("Distance Travelled (Fitness)")
    plt.xticks(range(len(reactive_tracklist)), reactive_tracklist)
    # plt.ylim(0, 600000)
    plt.show()


# Handles update settings button onclick behaviour
# (Author: Nadan)
def update_onclick():
    global epochs, train_perc, do_shuffle, seed, do_bespoke_tests

    # Update settings
    epochs = int(epochs_entry.get())
    seed = int(seed_entry.get())
    train_perc = float(train_perc_entry.get())
    do_shuffle = int(do_shuffle_var.get())
    do_bespoke_tests = int(do_bespoke_tests_var.get())

    # Save settings to config file
    config["Settings"] = {
        "epochs": str(epochs),
        "train_perc": str(train_perc),
        "do_shuffle": str(do_shuffle),
        "seed": str(seed),
        "do_bespoke_tests": str(do_bespoke_tests),
    }
    with open("../config/settings.ini", "w") as settings_file:
        config.write(settings_file)

    # Update track lists
    update_tracklist()


# Reads data from a csv file
# (Author: Nadan)
def read_from_csv(file_path):
    # Array to hold data
    data = []
    # Open csv file
    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        # Append each row into data array
        for row in csv_reader:
            data.append([float(field) for field in row])
    return data


# Update all track lists based on settings
# (Author: Nadan)
def update_tracklist():
    global training_tracklist, testing_tracklist, reactive_tracklist, do_bespoke_tests, TRACK_NUM, train_perc, seed

    # Create seed for random value reproducibility
    random.seed(seed)

    # Create the ordered list of tracks
    tracklist = list(range(1, TRACK_NUM + 1))

    # Create random permutation of tracks
    rand_tracklist = random.sample(range(1, TRACK_NUM + 1), TRACK_NUM)

    # Fixed tracks used only for testing
    fixed_test_tracklist = [101, 102, 103, 104, 105]

    # Split the track list up to the percentage of training tracks specified
    training_tracklist = rand_tracklist[: math.ceil(TRACK_NUM * train_perc)]

    # Decide on reactive and testing tracklists
    if do_bespoke_tests == 1:
        # NEAT and Reactive agents will test on the bespoke testing tracks
        testing_tracklist = fixed_test_tracklist
        reactive_tracklist = fixed_test_tracklist
    else:
        # Remainder of the track list after the training percentage is used for testing
        testing_tracklist = rand_tracklist[math.ceil(TRACK_NUM * train_perc) :]

        # Reactive will test on the original tracks
        reactive_tracklist = tracklist


# Main - Initialises from existing settings and creates GUI
# (Author: Nadan)
if __name__ == "__main__":
    # Open config file
    config = configparser.ConfigParser()
    config.read("../config/settings.ini")

    # Load current values from the settings config file (If missing load fallback value)
    epochs = int(config.get("Settings", "epochs", fallback=3))
    train_perc = float(config.get("Settings", "train_perc", fallback=0.8))
    do_shuffle = int(config.get("Settings", "do_shuffle", fallback=1))
    seed = int(config.get("Settings", "seed", fallback=1))
    do_bespoke_tests = int(config.get("Settings", "do_bespoke_tests", fallback=0))

    # Update track lists
    update_tracklist()

    # Create the menu window
    root = tk.Tk()
    root.title("Menu")
    # Set window size
    root.geometry("700x300")

    # Set button size
    button_height = 2
    button_width = 22

    # Create application frame
    app_frame = tk.LabelFrame(root, text="Configurations")
    app_frame.grid(row=0, column=0, padx=10, pady=10)

    # Create training button
    train_button = tk.Button(
        app_frame,
        text="Start Training",
        command=start_train_onclick,
        width=button_width,
        height=button_height,
    )
    train_button.grid(row=0, column=0, padx=5, pady=5)

    # Create training progress button
    train_progress_button = tk.Button(
        app_frame,
        text="Show Training Progress",
        command=training_progress_onclick,
        width=button_width,
        height=button_height,
    )
    train_progress_button.grid(row=0, column=1, padx=5, pady=5)

    # Create testing button
    test_button = tk.Button(
        app_frame,
        text="Start Testing",
        command=start_test_onclick,
        width=button_width,
        height=button_height,
    )
    test_button.grid(row=1, column=0, padx=5, pady=5)

    # Create testing results button
    test_results_button = tk.Button(
        app_frame,
        text="Show Test Results",
        command=results_testing_onclick,
        width=button_width,
        height=button_height,
    )
    test_results_button.grid(row=1, column=1, padx=5, pady=5)

    # Create tortoise testing button
    tortoise_button = tk.Button(
        app_frame,
        text="Run Tortoise",
        command=start_tortoise_onclick,
        width=button_width,
        height=button_height,
    )
    tortoise_button.grid(row=2, column=0, padx=5, pady=5)

    # Create tortoise results button
    tortoise_results_button = tk.Button(
        app_frame,
        text="Show Tortoise Results",
        command=results_tortoise_onclick,
        width=button_width,
        height=button_height,
    )
    tortoise_results_button.grid(row=2, column=1, padx=5, pady=5)

    # Create hare testing button
    hare_button = tk.Button(
        app_frame,
        text="Run Hare",
        command=start_hare_onclick,
        width=button_width,
        height=button_height,
    )
    hare_button.grid(row=3, column=0, padx=5, pady=5)

    # Create hare results button
    hare_results_button = tk.Button(
        app_frame,
        text="Show Hare Results",
        command=results_hare_onclick,
        width=button_width,
        height=button_height,
    )
    hare_results_button.grid(row=3, column=1, padx=5, pady=5)

    # Create independent training button
    update_button = tk.Button(
        app_frame,
        text="Independent Training",
        command=independent_train_onclick,
        width=button_width,
        height=button_height,
    )
    update_button.grid(row=4, column=0, padx=5, pady=5)

    # Create independent training results button
    update_button = tk.Button(
        app_frame,
        text="Independent Training Results",
        command=results_independent,
        width=button_width,
        height=button_height,
    )
    update_button.grid(row=4, column=1, padx=5, pady=5)

    # Create settings frame
    settings_frame = tk.LabelFrame(root, text="Settings")
    settings_frame.grid(row=0, column=1, padx=10, pady=10)

    # Create epochs input
    epochs_label = tk.Label(settings_frame, text="Epochs:")
    epochs_label.grid(row=0, column=0, padx=5, pady=5)
    epochs_entry = tk.Entry(settings_frame, width=10)
    epochs_entry.grid(row=0, column=1, padx=5, pady=5)
    epochs_entry.insert(0, str(epochs))

    # Create seed input
    seed_label = tk.Label(settings_frame, text="Seed:")
    seed_label.grid(row=1, column=0, padx=5, pady=5)
    seed_entry = tk.Entry(settings_frame, width=10)
    seed_entry.grid(row=1, column=1, padx=5, pady=5)
    seed_entry.insert(0, str(seed))

    # Create training percentage input
    train_perc_label = tk.Label(settings_frame, text="Training Percentage:")
    train_perc_label.grid(row=2, column=0, padx=5, pady=5)
    train_perc_entry = tk.Entry(settings_frame, width=10)
    train_perc_entry.grid(row=2, column=1, padx=5, pady=5)
    train_perc_entry.insert(0, str(train_perc))

    # Create do_shuffle checkbox
    do_shuffle_var = tk.IntVar()
    do_shuffle_checkbox = tk.Checkbutton(
        settings_frame, text="Do Shuffle", variable=do_shuffle_var
    )
    do_shuffle_checkbox.grid(row=3, column=0, padx=5, pady=5)
    do_shuffle_checkbox.select() if do_shuffle else do_shuffle_checkbox.deselect()

    # Create do_bespoke_tests checkbox
    do_bespoke_tests_var = tk.IntVar()
    do_bespoke_tests_checkbox = tk.Checkbutton(
        settings_frame, text="Perform Bespoke Tests", variable=do_bespoke_tests_var
    )
    do_bespoke_tests_checkbox.grid(row=3, column=1, padx=5, pady=5)
    do_bespoke_tests_checkbox.select() if do_bespoke_tests else do_bespoke_tests_checkbox.deselect()

    # Create update button
    update_button = tk.Button(
        settings_frame, text="Update Settings", command=update_onclick
    )
    update_button.grid(row=4, column=1, padx=5, pady=5)

    root.mainloop()
