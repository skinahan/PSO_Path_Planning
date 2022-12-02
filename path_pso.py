import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm
import math
import random
from sympy import Point, Line, Segment, Ray, Circle, Polygon

from IPython import display
import time

from mapf_map import *


class Particle:
    def __init__(self, position, velocity, best_position, best_value, inertia_weight, acceleration_coefficients,
                 random_number_generator):
        self.position = position
        self.velocity = velocity
        self.best_position = best_position
        self.best_value = best_value
        self.inertia_weight = inertia_weight
        self.acceleration_coefficients = acceleration_coefficients
        self.random_number_generator = random_number_generator

    def update_velocity(self, global_best_position):
        r1 = self.random_number_generator.uniform(0, 1)
        r2 = self.random_number_generator.uniform(0, 1)
        cognitive_component = self.acceleration_coefficients[0] * r1 * (self.best_position - self.position)
        social_component = self.acceleration_coefficients[1] * r2 * (global_best_position - self.position)
        self.velocity = self.inertia_weight * self.velocity + cognitive_component + social_component

    def update_position(self):
        self.position = self.position + self.velocity


class ParticleSwarmOptimization:
    def __init__(self, function, number_of_particles, number_of_dimensions, number_of_iterations,
                 initial_position_xrange, initial_position_yrange, initial_velocity_range, inertia_weight,
                 acceleration_coefficients,
                 random_number_generator, robot_pos, map, goal):
        self.function = function
        self.number_of_particles = number_of_particles
        self.number_of_dimensions = number_of_dimensions
        self.number_of_iterations = number_of_iterations
        self.initial_position_xrange = initial_position_xrange
        self.initial_position_yrange = initial_position_yrange
        self.initial_velocity_range = initial_velocity_range
        self.inertia_weight = inertia_weight
        self.acceleration_coefficients = acceleration_coefficients
        self.random_number_generator = random_number_generator
        self.particles = []
        self.robot_pos = robot_pos
        self.global_best_position = None
        self.global_best_value = None
        self.path = []
        self.map = map
        self.goal = goal

    def initialize_particles(self):
        goal = self.goal
        # goal = torch.tensor([end.x, end.y])
        #initial_position_xrange = [self.position[0] - sensor_range, self.position[0] + sensor_range]
        #initial_position_yrange = [self.position[1] - sensor_range, self.position[1] + sensor_range]

        top_left = [self.initial_position_xrange[0], self.initial_position_yrange[0]]
        bottom_left = [self.initial_position_xrange[0], self.initial_position_yrange[1]]

        top_right = [self.initial_position_xrange[1], self.initial_position_yrange[0]]
        bottom_right = [self.initial_position_xrange[1], self.initial_position_yrange[1]]

        corners = [top_left, bottom_left, top_right, bottom_right]
        for cpos in corners:
            position = cpos
            if particle_pos_valid(position, self.robot_pos, self.map):
                velocity = self.random_number_generator.uniform(self.initial_velocity_range[0],
                                                                self.initial_velocity_range[1],
                                                                self.number_of_dimensions)
                position = torch.tensor(position)
                velocity = torch.tensor(velocity)
                best_position = position

                best_value = self.function(position, self.goal, self.robot_pos)
                particle = Particle(position, velocity, best_position, best_value, self.inertia_weight,
                                    self.acceleration_coefficients, self.random_number_generator)
                self.particles.append(particle)

        for i in range(self.number_of_particles):
            xpos = self.random_number_generator.uniform(self.initial_position_xrange[0],
                                                        self.initial_position_xrange[1])
            # print(xpos)
            ypos = self.random_number_generator.uniform(self.initial_position_yrange[0],
                                                        self.initial_position_yrange[1])
            # print(ypos)
            position = [xpos, ypos]
            while not particle_pos_valid(position, self.robot_pos, self.map):
                xpos = self.random_number_generator.uniform(self.initial_position_xrange[0],
                                                            self.initial_position_xrange[1])
                ypos = self.random_number_generator.uniform(self.initial_position_yrange[0],
                                                            self.initial_position_yrange[1])
                position = [xpos, ypos]
            velocity = self.random_number_generator.uniform(self.initial_velocity_range[0],
                                                            self.initial_velocity_range[1],
                                                            self.number_of_dimensions)
            position = torch.tensor(position)
            velocity = torch.tensor(velocity)
            best_position = position

            best_value = self.function(position, self.goal, self.robot_pos)
            particle = Particle(position, velocity, best_position, best_value, self.inertia_weight,
                                self.acceleration_coefficients, self.random_number_generator)
            self.particles.append(particle)

    def update_global_best_position(self):
        for particle in self.particles:
            if self.global_best_value is None or particle.best_value < self.global_best_value:
                self.global_best_value = particle.best_value
                self.global_best_position = particle.best_position

    def optimize(self):
        self.initialize_particles()
        self.update_global_best_position()
        use_subgoals = False
        stag_ctr = 0
        for i in range(self.number_of_iterations):
            tmp_best_val = self.global_best_value
            tmp_best_pos = self.global_best_position.clone().detach().to(dtype=torch.float64)#torch.tensor(, dtype=torch.float64)
            # part_pos = torch.vstack([p.position for p in self.particles])
            # values = opt_func_vec(part_pos, self.goal, self.robot_pos)
            for idx, particle in enumerate(self.particles):
                bad_part = False
                old_vel = particle.velocity
                old_pos = particle.position
                particle.update_velocity(self.global_best_position)
                particle.update_position()
                # If the particle's new position intersects an obstacle (or is on the far side of an obstacle from the robot),
                # reject the particle's repositioning and try again
                retry_ctr = 0
                while not particle_pos_valid(particle.position, self.robot_pos, self.map):
                    if retry_ctr > 4:
                        bad_part = True
                        break
                    particle.velocity = old_vel
                    particle.position = old_pos
                    particle.update_velocity(self.global_best_position)
                    particle.update_position()
                    retry_ctr = retry_ctr + 1
                value = self.function(particle.position, self.goal, self.robot_pos)
                # Do not let this particle be considered for the update, if it was not valid
                if value < particle.best_value and not bad_part:
                    particle.best_value = value
                    particle.best_position = particle.position
            # self.global_iteration_ctr = self.global_iteration_ctr + 1
            self.update_global_best_position()
            gbp = self.global_best_position.clone().detach().to(dtype=torch.float64)#torch.tensor(, dtype=torch.float64)
            if torch.equal(tmp_best_pos, gbp):
                if stag_ctr == 4:
                    break
                stag_ctr += 1


# Return true only if:
# 1. the position is within the map bounds
# 2. the position is not inside an obstacle
# 3. the line connecting robot_pos to position does not intersect an obstacle
def particle_pos_valid(position, robot_pos, map):
    x = position[0]
    y = position[1]
    x, y = check_bounds(x, y, map)
    if x < 0 or y < 0:
        # print("INVALID: Negative")
        return False
    if x >= len(map[0]) or y >= len(map):
        # print("INVALID: OOB")
        return False
    nearest_node = get_nearest(position, map)
    rob_nearest = get_nearest(robot_pos, map)
    if nearest_node == rob_nearest:
        return False
    # print(f'NEAREST POS: {nearest_node.x}, {nearest_node.y}')
    obstacles = ['@', 'O', 'T']
    if nearest_node.terrain in obstacles:
        # print(f'INVALID: Obstacle: {nearest_node.terrain}')
        return False
    p1 = Point(robot_pos[0], robot_pos[1])
    p2 = Point(x, y)
    if p1.equals(p2):
        return False
    # Construct a line from the robot's position to the particle's position
    l1 = Line(p1, p2)
    all_nearby = nearest_node.get_all_adjacent(map)
    # Check if the line collides with any nearby obstacles
    orig_segment = Line(Point(0, 0), Point(0, 1))
    for neighbor_node in all_nearby:
        if neighbor_node.terrain in obstacles:
            x_n = neighbor_node.x
            y_n = neighbor_node.y
            p3 = Point(x_n, y_n)
            # Construct a line from the robot's position to the obstacle's position
            # l2 = Line(p1, p3)
            # Get the perpendicular line
            # l3 = l2.perpendicular_line(p1)
            # collis_seg = l3.projection(orig_segment)
            # obs_rad = 0.5
            # low_left = [x_n-obs_rad, y_n-obs_rad]
            # low_right = [x_n+obs_rad, y_n-obs_rad]
            # up_right = [x_n+obs_rad, y_n+obs_rad]
            # up_left = [x_n-obs_rad, y_n+obs_rad]
            # rect = Polygon(low_left, low_right, up_right, up_left)
            # circ = Circle(p3, 0.5)
            intersect = l1.intersection(p3)
            # intersect = l1.intersection(rect)
            # intersect = l1.intersection(collis_seg)
            if len(intersect) > 0:
                # print("INVALID: Obstacle Intersect")
                return False
    return True


def check_bounds(x, y, map):
    if x < 0:
        x = 0
    if x >= len(map[0]):
        x = len(map[0]) - 1
    if y < 0:
        y = 0
    if y >= len(map):
        y = len(map) - 1
    return x, y


def get_nearest(point, map):
    nearest_x = int(np.round(point[0]))
    nearest_y = int(np.round(point[1]))
    nearest_x, nearest_y = check_bounds(nearest_x, nearest_y, map)
    nearest_node = map[nearest_y][nearest_x]
    return nearest_node


def opt_func(point, goal, robot_pos):
    p1_scale = 0.0
    p2_scale = 1.0
    p1 = np.linalg.norm(point - robot_pos)
    p2 = np.linalg.norm(goal - point)

    return (p1_scale*p1) + (p2_scale*p2)


def opt_func_vec(particles, goal, robot_pos):
    p1 = np.linalg.norm(particles - robot_pos, axis=1)
    p2 = np.linalg.norm(goal - particles, axis=1)
    return p1 + p2


class Robot:
    def __init__(self, position, goal, speed_limit, sense_dist, num_particles, map_file):
        self.position = position
        self.start_pos = position
        self.velocity = torch.tensor([0, 0])
        self.goal = goal
        self.use_subgoal = False
        self.subgoal = None
        self.speed_limit = speed_limit
        self.sense_dist = sense_dist
        self.num_particles = num_particles
        self.map_file = map_file
        self.map = read_map(map_file)
        self.distance = 0
        self.path = []

    def search(self, display_map=True, display_steps=False):
        num_total_steps = 4000
        num_subgoal_steps = 14
        subgoal_ctr = 0
        num_backtrack_steps = 13
        goal = self.goal
        pause_time = 0.2
        # seconds between frames

        # Display the map
        # map_file = './mapf-map/Berlin_1_256.map'
        map_nums = read_map_nums(self.map_file)

        obstacles = ['@', 'O', 'T']
        curr_map_nums = np.copy(map_nums)
        for i in tqdm(range(num_total_steps)):
            #curr_map_nums = np.copy(map_nums)

            if subgoal_ctr >= num_subgoal_steps:
                if self.subgoal is not None:
                    reached_subgoal = torch.equal(self.position, self.subgoal)
                    if reached_subgoal:
                        self.use_subgoal = False
                        self.subgoal = None
                        goal = self.goal
                        subgoal_ctr = 0
                    else:
                        subgoal = get_perpendicular_subgoal(self.position, goal, self.speed_limit * 3, self.map)
                        self.subgoal = torch.tensor([subgoal.x, subgoal.y])
                        goal = self.subgoal
                        subgoal_ctr = 0

            if self.use_subgoal:
                subgoal_ctr = subgoal_ctr + 1
            # The PSO particles are generated randomly around the current position of the robot and within its sensing range (i.e. search space)
            number_of_dimensions = 2
            number_of_iterations = 40
            sensor_range = 2.0
            initial_position_xrange = [self.position[0] - sensor_range, self.position[0] + sensor_range]
            initial_position_yrange = [self.position[1] - sensor_range, self.position[1] + sensor_range]
            initial_velocity_range = [-3, 3]
            #initial_velocity_range = [-2, 2]
            # inertia weight (w) determines how much the velocity of a particle at time t
            # influences the velocity at time t + 1 (exploration vs. exploitation)
            inertia_weight = 0.5
            # influences the personal (c1) and global (c2) leaders of the search process
            acceleration_coefficients = [0.5, 0.8]
            random_number_generator = np.random.RandomState(0)

            pso = ParticleSwarmOptimization(opt_func, self.num_particles, number_of_dimensions, number_of_iterations,
                                            initial_position_xrange, initial_position_yrange, initial_velocity_range,
                                            inertia_weight,
                                            acceleration_coefficients, random_number_generator, self.position, self.map,
                                            goal)

            pso.optimize()

            #print(f'best position: {pso.global_best_position}')
            # print(f'best value: {pso.global_best_value}')

            last_pos = self.position

            # The best position found by PSO tells us which direction the robot should attempt to move to.
            best_node = get_nearest(pso.global_best_position, self.map)
            curr_x = self.position[0]
            curr_y = self.position[1]
            #print(f"{curr_x}, {curr_y}")
            #print(goal)

            best_x = best_node.x
            best_y = best_node.y

            curr_node = get_nearest(self.position, self.map)

            move_options = curr_node.get_adjacent_nodes(self.map)
            self.path.append(curr_node)

            p1 = Point(curr_node.x, curr_node.y)
            p2 = Point(best_x, best_y)

            if not p1.equals(p2):
                s = Segment(p1, p2)
                direction_ray = Ray(*s.args)
                # Basically, create a circle at P1 and see where the vector crosses to know where to move
                end_pt = Circle(p1, self.speed_limit).intersection(direction_ray)[0]
                end_x = float(end_pt.coordinates[0])
                end_y = float(end_pt.coordinates[1])
                end_coords = torch.tensor([end_x, end_y])
                end_node = get_nearest(end_coords, self.map)
                if end_node in move_options:
                    end_x = end_node.x
                    end_y = end_node.y
                    if not end_node.terrain in obstacles:
                        self.position = torch.tensor([end_x, end_y])
                    # print(self.position)

            # Improvement is not being made, likely stuck in local minima
            if torch.equal(last_pos, self.position):
                # Try going after a perpendicular goal for a while...
                if not self.use_subgoal:
                    # print('Using subgoal...')
                    self.use_subgoal = True
                    subgoal_ctr = 0
                    subgoal = get_perpendicular_subgoal(self.position, self.goal, self.speed_limit * 3, self.map)
                    self.subgoal = torch.tensor([subgoal.x, subgoal.y])
                    goal = self.subgoal
                    if not curr_node == subgoal:
                        curr_node.previous = None
                        a_star(self.map, curr_node, subgoal)
                        sub_path = backtrack(subgoal)
                        self.map = read_map(self.map_file)
                        for node in sub_path:
                            self.path.append(node)
                            x = node.y
                            y = node.x
                            curr_map_nums[x][y] = 5
                        last_pos_node = sub_path[-1]
                        if not last_pos_node.terrain in obstacles:
                            self.position = torch.tensor([last_pos_node.x, last_pos_node.y])
                            subgoal_ctr = num_subgoal_steps

            # display_steps = False

            if display_steps:
                if i % 1 == 0:
                    # Display the map
                    plt.figure(figsize=(8, 8))
                    plt.imshow(curr_map_nums)
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
                    plt.show()

            if torch.equal(self.position, self.goal):
                print(f'\n\nGoal reached after {i} search steps.')
                break

        path_length = len(self.path)

        if display_map:
            # Display the map
            map_nums = read_map_nums(self.map_file)
            # Display the path taken
            for node in self.path:
                x = node.y  # row..
                y = node.x  # column...?
                map_nums[x][y] = 4
            print(f"Path Length: {path_length}")
            map_nums[self.start_pos[1]][self.start_pos[0]] = 10
            map_nums[self.goal[1]][self.goal[0]] = 10
            map_name_no_path = os.path.basename(self.map_file)

            figure_title = f"PSO: {map_name_no_path}\nLength: {path_length}"

            plt.figure(figsize=(8, 8))
            plt.suptitle(figure_title)
            plt.imshow(map_nums)
            plt.show()
        return path_length


def get_perpendicular_subgoal(start, goal, dist, map):
    # Define a line from current position to the goal position
    p1 = Point(start[0], start[1])
    p2 = Point(goal[0], goal[1])
    l2 = Line(p1, p2)
    # Get the perpendicular line
    l3 = l2.perpendicular_line(p1)
    valid_subgoal = False
    obstacles = ['@', 'O', 'T']
    perp_options = []
    while not valid_subgoal:
        perp_points = Circle(p1, dist).intersection(l3)
        for p in perp_points:
            p_coords = p.coordinates
            p_x = float(p_coords[0])
            p_y = float(p_coords[1])
            coord = torch.tensor([p_x, p_y])
            perp_node = get_nearest(coord, map)
            if not perp_node.terrain in obstacles:
                valid_subgoal = True
                perp_options.append(perp_node)
        dist = dist + 1
    return random.choice(perp_options)
