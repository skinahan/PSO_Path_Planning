# The maps have the following format:
# All maps begin with the lines:

# type octile
# height y
# width x
# map
# where y and x are the respective height and width of the map.
# The map data is
# stored as an ASCII grid. The upper-left corner of the map is (0,0). The following characters are possible:

# . - passable terrain
# G - passable terrain
# @ - out of bounds
# O - out of bounds
# T - trees (unpassable)
# S - swamp (passable from regular terrain)
# W - water (traversable, but not passable from terrain)

import sys
import numpy as np
from PIL import Image


# 1: read map from file
# 2: create object for each node containing its coordinates, and a list of adjacents
# 3: determine cost for each node
# 4: implement Dijkstra algorithm
# 5: output the result

class Node:
    def __init__(self, x, y, terrain):
        self.x = int(x)
        self.y = int(y)
        self.terrain = terrain
        self.cost = 0
        self.distance = float("inf")
        self.previous = None

    def __repr__(self):
        return f"({self.x}, {self.y}): {self.cost}"

    def __str__(self):
        return f"({self.x}, {self.y}): {self.cost}"

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.distance < other.distance

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.x == other.x) and (self.y == other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def get_adjacent_nodes(self, map):
        result = []
        if self.y > 0 and map[self.y - 1][self.x].terrain != '@' and map[self.y - 1][self.x].terrain != 'O':
            result.append(map[self.y - 1][self.x])
        if self.x > 0 and map[self.y][self.x - 1].terrain != '@' and map[self.y][self.x - 1].terrain != 'O':
            result.append(map[self.y][self.x - 1])
        if self.y < len(map) - 1 and map[self.y + 1][self.x].terrain != '@' and map[self.y + 1][self.x].terrain != 'O':
            result.append(map[self.y + 1][self.x])
        if self.x < len(map[0]) - 1 and map[self.y][self.x + 1].terrain != '@' and map[self.y][
            self.x + 1].terrain != 'O':
            result.append(map[self.y][self.x + 1])
        if self.x > 0 and self.y > 0 and map[self.y - 1][self.x - 1].terrain != '@' and map[self.y - 1][
            self.x - 1].terrain != 'O':
            result.append(map[self.y - 1][self.x - 1])
        if self.x < len(map[0]) - 1 and self.y > 0 and map[self.y - 1][self.x + 1].terrain != '@' and map[self.y - 1][
            self.x + 1].terrain != 'O':
            result.append(map[self.y - 1][self.x + 1])
        if self.x > 0 and self.y < len(map) - 1 and map[self.y + 1][self.x - 1].terrain != '@' and map[self.y + 1][
            self.x - 1].terrain != 'O':
            result.append(map[self.y + 1][self.x - 1])
        if self.x < len(map[0]) - 1 and self.y < len(map) - 1 and map[self.y + 1][self.x + 1].terrain != '@' and \
                map[self.y + 1][self.x + 1].terrain != 'O':
            result.append(map[self.y + 1][self.x + 1])
        return result

    def get_all_adjacent(self, map):
        result = []
        # print(self.x)
        # print(self.y)
        if self.y > 0:
            result.append(map[self.y - 1][self.x])
        if self.x > 0 and self.y < len(map) - 1:
            result.append(map[self.y][self.x - 1])
        if self.y < len(map) - 1:
            result.append(map[self.y + 1][self.x])
        if self.x < len(map[0]) - 1 and self.y < len(map) - 1:
            result.append(map[self.y][self.x + 1])
        if self.x > 0 and self.y > 0:
            result.append(map[self.y - 1][self.x - 1])
        if self.x < len(map[0]) - 1 and self.y > 0:
            result.append(map[self.y - 1][self.x + 1])
        if self.x > 0 and self.y < len(map) - 1:
            result.append(map[self.y + 1][self.x - 1])
        if self.x < len(map[0]) - 1 and self.y < len(map) - 1:
            result.append(map[self.y + 1][self.x + 1])
        return result


def check_map_file(map_file):
    with open(map_file) as f:
        for i in range(4):
            line = f.readline().strip()
            if line == 'type octile':
                continue
            if line.startswith('height'):
                y = int(line[6:])
            if line.startswith('width'):
                x = int(line[6:])
            if line == 'map':
                return x, y
    return None


def read_map_nums_resized(map_file):
    map = read_map_nums(map_file)
    row_len = len(map[0])
    col_len = len(map)
    new_row_len = row_len * 10
    new_col_len = col_len * 10
    np_map = np.array(map)
    map_img = Image.fromarray(np_map)

    resized_map = np.array(map_img.resize((new_row_len, new_col_len)))
    resized_map = np.rint(resized_map)
    return resized_map


def read_map_resized(map_file):
    resized_map = read_map_nums_resized(map_file)
    final_map = []
    for row in resized_map:
        new_row = []
        for i in range(len(row)):
            terrain = '.'
            if row[i] == 0.0:
                terrain = '.'
            elif row[i] == 1.0:
                terrain = '@'
            elif row[i] == 2.0:
                terrain = 'S'
            elif row[i] == 3.0:
                terrain = 'W'
            node = Node(i, len(final_map), terrain)

            if node.terrain == '.':
                node.cost = 1
            elif node.terrain == 'G':
                node.cost = 1
            elif node.terrain == '@':
                # node.cost = 10000000.0
                node.cost = float("inf")
            elif node.terrain == 'O':
                # node.cost = 10000000.0
                node.cost = float("inf")
            elif node.terrain == 'T':
                # node.cost = 10000000.0
                node.cost = float("inf")
            elif node.terrain == 'S':
                node.cost = 10
            elif node.terrain == 'W':
                node.cost = 10
            new_row.append(node)
        final_map.append(new_row)
    return final_map


def read_map(map_file):
    with open(map_file) as f:
        map = []
        for line in f:
            line = line.strip()
            if line != 'type octile':
                if not line.startswith('height'):
                    if not line.startswith('width'):
                        if line != 'map':
                            row = []
                            for i in range(len(line)):
                                node = Node(i, len(map), line[i])
                                if node.terrain == '.':
                                    node.cost = 1
                                elif node.terrain == 'G':
                                    node.cost = 1
                                elif node.terrain == '@':
                                    # node.cost = 10000000.0
                                    node.cost = float("inf")
                                elif node.terrain == 'O':
                                    # node.cost = 10000000.0
                                    node.cost = float("inf")
                                elif node.terrain == 'T':
                                    # node.cost = 10000000.0
                                    node.cost = float("inf")
                                elif node.terrain == 'S':
                                    node.cost = 10
                                elif node.terrain == 'W':
                                    node.cost = 10
                                row.append(node)
                            map.append(row)
    return map


# Read the map as an integer grid for easy display
def read_map_nums(map_file):
    with open(map_file) as f:
        map = []
        for line in f:
            line = line.strip()
            if line != 'type octile':
                if not line.startswith('height'):
                    if not line.startswith('width'):
                        if line != 'map':
                            row = []
                            for i in range(len(line)):
                                num_spot = 0
                                node = Node(i, len(map), line[i])
                                if node.terrain == '.':
                                    num_spot = 0  # passable
                                elif node.terrain == 'G':
                                    num_spot = 0  # passable
                                elif node.terrain == '@':
                                    num_spot = 1.0  # out of bounds
                                elif node.terrain == 'O':
                                    num_spot = 1.0  # out of bounds
                                elif node.terrain == 'T':
                                    num_spot = 1.0  # obstacle (trees)
                                elif node.terrain == 'S':
                                    num_spot = 2.0  # obstacle (swamp)
                                elif node.terrain == 'W':
                                    num_spot = 3.0  # obstacle (water)
                                row.append(num_spot)
                            map.append(row)
        return map


def print_map(map):
    for r in map:
        row = ''
        for n in r:
            row += str(n.terrain) + ' '
        print(row + '\n')


def print_live_map(map, x, y):
    curr_X = 0
    curr_Y = 0
    for r in map:
        row = ''
        for n in r:
            if curr_Y == y and curr_X == x:
                row += 'X '
            else:
                row += str(n.terrain) + ' '
            curr_X = curr_X + 1
        curr_Y = curr_Y + 1
        curr_X = 0
        print(row + '\n')


def get_start(map):
    for r in map:
        for n in r:
            if n.terrain == 'G' or n.terrain == '.':
                return n
    raise Exception('Error getting starting point')


def get_end(map):
    row_len = len(map[0])
    ind = row_len - 1
    return map[ind][ind]



def dijkstra(map, start, end):
    visited = set()
    unvisited = set()
    current = start

    start.distance = 0
    start.previous = None
    unvisited.add(start)

    while len(unvisited) > 0:
        unvisited = sorted(unvisited)
        unvisited = set(unvisited)
        current = unvisited.pop()
        # print_live_map(map, current.x, current.y)

        visited.add(current)

        if current == end:
            break

        adjacents = current.get_adjacent_nodes(map)
        for adjacent in adjacents:
            if adjacent in visited:
                continue

            alt_distance = current.distance + current.cost + adjacent.cost
            if alt_distance < adjacent.distance:
                adjacent.distance = alt_distance
                adjacent.previous = current

            unvisited.add(adjacent)


def dfs(map, start, end):
    stack = [start]
    visited = set()
    while len(stack) > 0:
        current = stack.pop()
        if current == end:
            return current
        visited.add(current)
        for n in current.get_adjacent_nodes(map):
            if n not in visited:
                stack.append(n)
                n.previous = current
    return None


def bfs(map, start, end):
    queue = [start]
    start.distance = 0
    while len(queue) > 0:
        current = queue.pop(0)
        if current == end:
            return current
        for n in current.get_adjacent_nodes(map):
            if n.distance == float("inf"):
                n.distance = current.distance + 1
                n.previous = current
                queue.append(n)
    return None


def a_star(map, start, end):
    start.distance = 0
    open_list = [start]
    closed_list = []
    while len(open_list) > 0:
        current = min(open_list)
        if current == end:
            return current
        open_list.remove(current)
        closed_list.append(current)
        for n in current.get_adjacent_nodes(map):
            if n in closed_list:
                continue
            if n not in open_list:
                open_list.append(n)
            if current.distance + n.cost < n.distance:
                n.distance = current.distance + n.cost
                n.previous = current
    return None


def print_path(map, end):
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = current.previous
    path.reverse()
    for n in path:
        map[n.y][n.x].terrain = '*'
    print_map(map)


def backtrack(end):
    result = []
    current = end
    while current != None:
        result.append(current)
        current = current.previous

    return result[::-1]
