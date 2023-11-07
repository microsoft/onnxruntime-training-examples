# this file is needed in order to load the pkl file - for the checkpoint convertor

class EasyDict(dict):
    def __init__(self):
        super().__init__()
        self.state_dict = None

    def __setstate__(self, state):
        self.state_dict = state