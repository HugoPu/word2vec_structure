import reader

class PTBInput(object):
    '''The input data.'''

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size # The size of one mini-batch
        self.num_steps = num_steps = config.num_steps # The length of one sentence in one batch
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps # The number of sentences
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name) # x, y