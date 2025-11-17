from nnetwork import Network
import mnist_loader

def _main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    N = Network([784, 30, 10, 5, 5, 5, 8, 10])
    N.train(training_data, 30, 10, 3.0, test_data=test_data)
    #N.SGD(training_data, 30, 10, 3.0, test_data=test_data)

if __name__ == '__main__':
    _main()
