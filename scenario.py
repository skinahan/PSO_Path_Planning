import torch


class Scenario:
    def __init__(self, filename):
        self.filename = filename
        self.scenarios = []
        self.scenario_count = 0
        self.scenario_start = []
        self.scenario_goal = []
        self.scenario_optimal_length = 0
        self.scenario_map_width = 0
        self.scenario_map_height = 0
        self.scenario_map_string = ""
        self.read_scenarios()

    def read_scenarios(self):
        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith("version"):
                    continue
                else:
                    if len(line) == 0:
                        continue
                    self.scenarios.append(line.split())
                    self.scenario_count += 1

    def get_scenario(self, scenario_number):
        return self.scenarios[scenario_number]

    def get_scenario_count(self):
        return self.scenario_count

    def get_scenario_start(self, scenario_number):
        str_start = self.scenarios[scenario_number][4:6]
        start_x = int(self.scenarios[scenario_number][4])
        start_y = int(self.scenarios[scenario_number][5])
        return torch.tensor([start_x, start_y])
        #return self.scenarios[scenario_number][4:6]

    def get_scenario_goal(self, scenario_number):
        str_goal = self.scenarios[scenario_number][6:8]
        goal_x = int(self.scenarios[scenario_number][6])
        goal_y = int(self.scenarios[scenario_number][7])
        return torch.tensor([goal_x, goal_y])
        #return self.scenarios[scenario_number][6:8]

    def get_scenario_optimal_length(self, scenario_number):
        return float(self.scenarios[scenario_number][8])

    def get_scenario_map_width(self, scenario_number):
        return int(self.scenarios[scenario_number][2])

    def get_scenario_map_height(self, scenario_number):
        return int(self.scenarios[scenario_number][3])

    def get_scenario_map_string(self, scenario_number):
        return self.scenarios[scenario_number][1]


if __name__ == "__main__":
    s = Scenario("./scen/Berlin_1_256-even-1.scen")
    print(s.get_scenario(0))
    print(s.get_scenario_count())
    print(s.get_scenario_start(0))
    print(s.get_scenario_goal(0))
    print(s.get_scenario_optimal_length(0))
    print(s.get_scenario_map_width(0))
    print(s.get_scenario_map_height(0))
    print(s.get_scenario_map_string(0))
