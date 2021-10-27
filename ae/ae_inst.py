# import warnings
# warnings.filterwarnings("ignore")
import imp
import os
import json
import math
import torch
import numpy
import argparse
import weka.core.jvm
import weka.core.converters
from tempfile import TemporaryFile
import scikit_wrappers
from sklearn.preprocessing import MinMaxScaler
import time

def get_longest_anomaly(data):
    anomalies_length = []
    num_features = 83
    num_anomalies = data.shape[0]
    for i in range(0,num_anomalies):
        for x in range(0,num_features):
            # print(data[i][x])
            length = numpy.count_nonzero(~numpy.isnan(data[i][x]))
            #print(length)
            anomalies_length.append(length)
    return max(anomalies_length)

def get_shortest_anomaly(data):
    anomalies_length = []
    num_features = 83
    num_anomalies = data.shape[0]
    for i in range(0,num_anomalies):
        for x in range(0,num_features):
            # print(data[i][x])
            length = numpy.count_nonzero(~numpy.isnan(data[i][x]))
            #print(length)
            anomalies_length.append(length)
    return min(anomalies_length)

def load_UEA_dataset(path, dataset):
    """
    Loads the UEA dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    # Load train and test array
    # print("Path:",path)
    with open(path + dataset + "_TRAIN.npy", 'rb') as f:
        train = numpy.load(f, allow_pickle=True)
    # print("Train:", train)
    print("Train shape:", train.shape)

    with open(path + dataset + "_TEST.npy", 'rb') as f:
        test = numpy.load(f, allow_pickle=True)
    # print("Test:", test)
    print("Test shape:", test.shape)
    
    print("Shortest anomly in TRAIN:", get_shortest_anomaly(train))
    print("Shortest anomly in TEST:", get_shortest_anomaly(test))
    
    # Scaling
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train.reshape(-1,train.shape[-1])).reshape(train.shape)
    test = scaler.fit_transform(test.reshape(-1,test.shape[-1])).reshape(test.shape)

    print("Train:", train)
    # print("Test:", test)
    return train, test


def fit_hyperparameters(file, train, cuda, gpu, train_labels=None,
                        save_memory=False):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = scikit_wrappers.CausalCNNAutoencoder()

    # Loads a given set of hyperparameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = numpy.shape(train)[1]
    params['cuda'] = cuda
    params['gpu'] = gpu
    
    print("Hyperparameters:", json.dumps(params, indent=4))
    
    classifier.set_params(**params)

    return classifier.fit_autoencoder(
        train, params, os.path.join(args.save_path, args.dataset), train_labels, save_memory=save_memory, verbose=True
        )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of hyperparameters to use ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')

    return parser.parse_args()

def assemble_dataset(train, test):
    print("Train:", train.shape)
    # print(train)
    print("Test:", test.shape)
    # print(test)

    dataset = numpy.concatenate((train,test))
    print("Complete Dataset:", dataset.shape, dataset)
    return dataset

# Encoder code still included, if encoder_only = True, encoder is used
encoder_only = False

whole_data = False

if __name__ == '__main__':
    if encoder_only:
        args = parse_arguments()
        if args.cuda and not torch.cuda.is_available():
            print("CUDA is not available, proceeding without it...")
            args.cuda = False

        train, test  = load_UEA_dataset(
            args.path, args.dataset
        )

        if not args.load and not args.fit_classifier:
            classifier = fit_hyperparameters(
                args.hyper, train, args.cuda, args.gpu,
                save_memory=True)
            print("TRAINING WITH HYPERPARAMETERS DONE!")
        else:
            classifier = scikit_wrappers.CausalCNNEncoderClassifier()
            hf = open(
                os.path.join(
                    args.save_path, args.dataset + '_hyperparameters.json'
                ), 'r'
            )
            hp_dict = json.load(hf)
            hf.close()
            hp_dict['cuda'] = args.cuda
            hp_dict['gpu'] = args.gpu
            classifier.set_params(**hp_dict)
            classifier.load(os.path.join(args.save_path, args.dataset))

        if not args.load:
            if args.fit_classifier:
                classifier.fit_classifier(classifier.encode(train))
            classifier.save(
                os.path.join(args.save_path, args.dataset)
            )
            with open(
                os.path.join(
                    args.save_path, args.dataset + '_hyperparameters.json'
                ), 'w'
            ) as fp:
                json.dump(classifier.get_params(), fp)
        
        score = classifier.score(test, test_labels=None)
        score = {'silhouette_score_TEST': score}
        with open(args.save_path + 'score.json', 'w') as f:
            json.dump(score, f)
        print("Test accuracy: " + str(score['silhouette_score_TEST']))
    else:
        print("AUTOENCODER....")
        args = parse_arguments()
        if args.cuda and not torch.cuda.is_available():
            print("CUDA is not available, proceeding without it...")
            args.cuda = False
        
        train, test = load_UEA_dataset(
            args.path, args.dataset
        )
        if whole_data:
            dataset = assemble_dataset(train, test)
            print("Using Whole Dataset...", dataset.shape)
        else:
            dataset = train

        if not args.load and not args.fit_classifier:
            # Saving of model after fitting
            s = time.time()
            classifier = fit_hyperparameters(
                args.hyper, dataset, args.cuda, args.gpu,
                save_memory=True
                )
            e = time.time()
            print("TRAINING WITH HYPERPARAMETERS DONE!")
            print("Processing Time = {}".format(e-s))
            processing_time = e-s
            score = {'Training Time': str(processing_time)}
            with open(args.save_path + 'Training_ProcessingTime_'+str(processing_time)+'.json', 'w') as f:
                json.dump(score, f)
