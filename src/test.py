from network import Network
import test_loader

def _main():
    training_data, validation_data, test_data = test_loader.load_data_wrapper('test')
    N = Network([2, 8, 8, 1])
    N.SGD(training_data, 100, 10, 3.0, test_data=test_data)

if __name__ == '__main__':
    _main()
