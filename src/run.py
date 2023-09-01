import mnist_loader
#import network as netw
import network2 as netw

training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 

"""
net = netw.Network([784, 100, 10])
net.SGD(training_data, 50, 10, 3.0, test_data=test_data)

"""

net = netw.Network([784, 100, 10], cost=netw.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 50, 10, 3.0, evaluation_data=test_data, monitor_evaluation_accuracy=True)