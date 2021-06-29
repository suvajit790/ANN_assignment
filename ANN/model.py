from .network import Network


class Model():
    def __init__(self, topology):
        self.topology = topology
        self.network = Network(topology)
    
    def get_topology(self):
        return self.topology
    
    def set_eta(self, learning_rate):
        Network.eta = learning_rate
    
    def train(self, data, lebel):
        self.network.setInput(data)
        self.network.feedForword()
        self.network.backPropagate(lebel)
        
        return self.network.getError(lebel)
    
    def validate(self, data, lebel):
        self.network.setInput(data)
        self.network.feedForword()

        return self.network.getError(lebel)

    def test(self, data):
        self.network.setInput(data)
        self.network.feedForword()

        return self.network.getResults()

    def test_th(self, data):
        self.network.setInput(data)
        self.network.feedForword()

        return self.network.getThResults()
