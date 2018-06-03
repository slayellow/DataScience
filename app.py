from configuration import Configuration
from Class5_Single import Class5_Single
from Class5_Multi import Class5_Multi
from Class6_Single import Class6_Single
from Class6_Multi import Class6_Multi

class app(object):
    def __init__(self):
        self.config = Configuration()

    def load_config(self):
        self.config.load_config("config.ini")

    def run_model(self):
        if self.config.index == 5:
            for learn in self.config.learning_rate:
                for ran in self.config.range:
                    Class5_Single(self.config, learn, ran)
                    Class5_Multi(self.config, learn, ran)
        else:
            for learn in self.config.learning_rate:
                for ran in self.config.range:
                    Class6_Single(self.config, learn, ran)
                    Class6_Multi(self.config, learn, ran)


