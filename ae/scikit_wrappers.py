import math
import numpy
import torch
import sklearn
import sklearn.svm
import sklearn.cluster
import sklearn.model_selection
import sklearn.metrics.cluster
import joblib
import torch.utils.data
import causal_cnn
from operator import itemgetter
from torchsummary import summary
import matplotlib.pyplot as plt
import json
import os

class TimeSeriesAutoencoder(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """
    All inheriting classes should implement the get_params and set_params
    methods, as in the recommendations of scikit-learn.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param encoder Encoder PyTorch module.
    @param params Dictionaries of the parameters of the encoder.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """

    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                 batch_size, nb_steps, lr, penalty, early_stopping,
                 autoencoder, params, in_channels, out_channels, cuda=False,
                 gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.autoencoder = autoencoder
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)

    def load_autoencoder(self, prefix_file):
        """
        Loads the autoencoder.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.autoencoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_autoencoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.autoencoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_autoencoder.pth',
                map_location=lambda storage, loc: storage
            ))
    
    def fit_autoencoder(self, X, params, prefix_file, y=None, save_memory=False, verbose=False):
        print(self.autoencoder)
        # summary(self.autoencoder.float(), (84,1113),26)
        print("###########################################")
        print("DATA:", X.shape)
        train = torch.from_numpy(X)
        train_torch_dataset = X
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        i = 0 # Number of performed optimization steps
        epochs = 0 # Number of performed epochs

        # Training
        loss_vals = []
        print('Nb_steps: ', self.nb_steps)
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            total_loss = 0
            for batch in train_generator:
                # print("Batch:", batch.size())
                self.optimizer.zero_grad()
               
                output = self.autoencoder(batch)
                # print("Output Loss Tensor:", output)
                # print("Output Loss Size:", loss.size())
                
                loss = output.mean()
                
                # SHAP requires a single loss value per sample, we sum over the time and feature dimension
                # Forward function of the autoencoder outputs a loss of shape (batch_size, 1)
                # To obtain original representation of the loss before computing the backward pass through the network,
                # loss is divided number of time steps * number of features and hence is between 0 and 1
                loss = loss/(1113*84)
                print("Batch Loss:", loss.item())

                loss.backward()
                self.optimizer.step()
                i += 1

                total_loss = total_loss + loss.item()
                print("Loss Sum:", total_loss)

                if i >= self.nb_steps:
                    break

            final_loss = total_loss / len(train_generator) 
            print("-------------------------------")
            print("Loss:", final_loss)
            loss_vals.append(final_loss)

            if final_loss < 0.0001:
                print("Stop Training. Loss below 0.0001")
                break

            epochs += 1
        print("###########################################")
        print("TRAINING AUTOENCODER WITH HYPERPARAMETERS DONE!")

        # Plot loss
        print("Plotting loss...")
        plt.plot(loss_vals)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999',linestyle='-', alpha=0.2)
        plt.savefig("/home/sto/Data/0-sto-estimators-autoencoder/MAY1/"+self.architecture+"_Loss_" + str(self.lr)  + "_" + str(self.nb_steps) + "depth_" + str(self.depth) + ".png")
        
        # Save autoencoder (temporary solution: inside fit_function)
        print("Saving model:", prefix_file + '_' + self.architecture + '_autoencoder.pth')

        checkpoint = {
                      'model': self.autoencoder,
                      'state_dict': self.autoencoder.state_dict(),
                      'optimizer' : self.optimizer.state_dict()
                      }
        torch.save(checkpoint, prefix_file + '_' + self.architecture + '_autoencoder.pth')
        
        # Save used hyperparameters
        with open(os.path.join(prefix_file+'_CausalCNN_WHOLE-DATA_hyperparameters.json'), 'w') as fp: json.dump(params, fp)
        
        return self.autoencoder

class CausalCNNAutoencoder(TimeSeriesAutoencoder):
    """
   @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of features in the final output.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """

    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, channels=10, depth=2,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0):
        super(CausalCNNAutoencoder, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_autoencoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, out_channels, cuda, gpu
        )
        self.architecture = 'CausalCNN_Autoencoder'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size
    
    def __create_autoencoder(self, in_channels, channels, depth, reduced_size,
                             out_channels, kernel_size, cuda, gpu):
        autoencoder = causal_cnn.CausalCNNAutoencoder(
                in_channels, channels, depth, reduced_size, out_channels,
                kernel_size
        )
        autoencoder.double()
        if cuda:
            autoencoder.cuda(gpu)
        return autoencoder
    
    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu)
        return self

