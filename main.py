from mapf_map import *
from path_pso import *
from timeout_handler import *
from scenario import *
from multiprocessing import Pool

import torch


@timeout(600)
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
    # Time limit: 10 minutes (600 seconds)
    t_lim = 600
    num_scenarios = scen_cl.get_scenario_count()
    map_file_root = './mapf-map/'
    timed_out_searching = False
    num_agents_timeout = 0
    r_speed_lim = 1

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
            r_new = Robot(start_pos, goal_pos, 1, 2, 3, map_file)
            robots.append(r_new)
        try:
            if len(robots) > 0:
                # search_all(robots)
                path_lens = timeout(t_lim)(search_all)(robots)
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
    return num_agents_timeout


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
    for scen_file in os.listdir(scen_file_root):
        # load the scenario
        scen_file_path = os.path.join(scen_file_root, scen_file)
        scen_cl = Scenario(scen_file_path)
        run_scenario(scen_cl)


def main():
    run_scenarios()
    # basic_robot_search()


if __name__ == '__main__':
    main()
