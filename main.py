import os.path

from mapf_map import *
from path_pso import *
from timeout_handler import *
from scenario import *
from multiprocessing import Pool

import torch
import csv


@timeout(300)
def search_all(robots):
    path_lengths = []
    for robot in robots:
        print(f"\n\nMap: {robot.map_file}")
        print(f"Start Pos: {robot.position}")
        print(f"End Pos: {robot.goal}\n\n")
        r_length = robot.search(display_map=False)
        path_lengths.append(r_length)
    return path_lengths


def run_scenario(scen_cl):
    # Time limit: 5 minutes (300 seconds)
    t_lim = 300
    num_scenarios = scen_cl.get_scenario_count()
    map_file_root = './mapf-map/'
    timed_out_searching = False
    num_agents_timeout = 0
    r_speed_lim = 1
    all_path_lens = []
    for j in tqdm(range(1, num_scenarios)):
        # Add one agent at a time until the problem can't be solved in a given time limit
        for k in range(0, j):
            robots = []
            start_pos = scen_cl.get_scenario_start(scenario_number=k)
            map_width = scen_cl.get_scenario_map_width(scenario_number=k)
            if map_width >= 250:
                r_speed_lim = 5
            else:
                r_speed_lim = 1
            goal_pos = scen_cl.get_scenario_goal(scenario_number=k)
            map_file = scen_cl.get_scenario_map_string(scenario_number=k)
            map_file = os.path.join(map_file_root, map_file)
            r_new = Robot(start_pos, goal_pos, 1, 1, 3, map_file)
            robots.append(r_new)
        try:
            if len(robots) > 0:
                # search_all(robots)
                path_lens = timeout(t_lim)(search_all)(robots)
                all_path_lens.append(path_lens)
        except TimeoutException:
            print("Timed out!")
            print(f"Map: {map_file}")
            print(f"Start Pos: {start_pos}")
            print(f"End Pos: {goal_pos}")
            timed_out_searching = True
            num_agents_timeout = len(robots)
            break
    if not timed_out_searching:
        num_agents_timeout = num_scenarios
    return num_agents_timeout, all_path_lens


def basic_robot_search():
    # map_file = './mapf-map/Berlin_1_256.map'
    map_file = './mapf-map/empty-16-16.map'
    map = read_map(map_file)
    start = get_start(map)
    end = get_end(map)

    # Doesn't work... whY???
    start_pos = torch.tensor([10, 10])
    end_pos = torch.tensor([1, 1])

    robot = Robot(start_pos, end_pos, 1, 1, 3, map_file)
    robot.search(display_map=True, display_steps=False)


def run_scenarios():
    scen_file_root = './scen/'
    output_file_root = './results/'
    for scen_file in os.listdir(scen_file_root):
        scen_base_name = os.path.splitext(scen_file)[0]

        # load the scenario
        scen_file_path = os.path.join(scen_file_root, scen_file)
        scen_cl = Scenario(scen_file_path)
        num_agents, path_lens = run_scenario(scen_cl)

        out_csv_name = f'{scen_base_name}_timeout.csv'
        out_csv_path = os.path.join(output_file_root, out_csv_name)
        with open(out_csv_path, 'w') as csv_file:
            writer_obj = csv.writer(csv_file)
            writer_obj.writerow([scen_base_name, num_agents])
        out_csv2_name = f'{scen_base_name}_paths.csv'
        out_csv2_path = os.path.join(output_file_root, out_csv2_name)
        with open(out_csv2_path, 'w') as csv_file:
            writer_obj = csv.writer(csv_file)
            for path_arr in path_lens:
                row = [scen_base_name].extend([str(x) for x in path_arr])
                writer_obj.writerow(row)


def main():
    run_scenarios()
    # basic_robot_search()


if __name__ == '__main__':
    main()
