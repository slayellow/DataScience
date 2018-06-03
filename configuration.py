import os
import configparser as cfp

class Configuration(object):
    def __init__(self):
        pass

    def load_config(self, file):
        if os.path.isfile(file):
            self.iniReader = cfp.ConfigParser()
            self.iniReader.read(file, encoding='UTF8')
            self.index = self.iniReader['single_setting']['index']
            self.batch_size = self.iniReader['single_setting']['batch_size']
            self.print_range = self.iniReader['single_setting']['print_range']
            self.hidden_layer = self.iniReader['single_setting']['hidden_layer']
            self.learning_rate = self.iniReader['multi_setting']['learning_rate'].split(',')
            self.range = self.iniReader['multi_setting']['range'].split(',')
            self.change_variable()

    def change_variable(self):
        self.index = int(self.index)
        self.batch_size = int(self.batch_size)
        self.print_range = int(self.print_range)
        self.hidden_layer = int(self.hidden_layer)
        for i in range(len(self.learning_rate)):
            self.learning_rate[i] = float(self.learning_rate[i])
        for i in range(len(self.range)):
            self.range[i] = int(self.range[i])

