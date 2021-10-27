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
import triplet_loss as losses
import causal_cnn
from operator import itemgetter
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, Normalizer
class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):
    """
    "Virtual" class to wrap an encoder of time series as a PyTorch module and
    a SVM classifier with RBF kernel on top of its computed representations in
    a scikit-learn class.

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
                 encoder, params, in_channels, out_channels, cuda=False,
                 gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.TripletLoss(
            compared_length, nb_random_samples, negative_penalty
        )
        self.loss_varying = losses.TripletLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty
        )
        self.clustering_method = 'KMeans'

        if self.clustering_method == 'KMeans':
            self.classifier = sklearn.cluster.KMeans()
        if self.clustering_method == 'DBSCAN':
            self.classifier = sklearn.cluster.DBSCAN()

        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def save_encoder(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )

    def save(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file)
        joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads an encoder and an SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.load_encoder(prefix_file)
        self.classifier = joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def fit_clustering(self, features, X):
        best_paramters = []
        if self.clustering_method == 'KMeans':
            print("[KMeans] parameter tuning...")
            maximum_clusters = 20
            K = range(2, maximum_clusters)
            silhouette_scores = []
            for k in K:
                self.classifier = sklearn.cluster.KMeans(n_clusters=k,random_state=42)
                labels = self.classifier.fit(features).labels_
                silhouette_score = round(sklearn.metrics.silhouette_score(
                    features, labels
                ), 3)
                silhouette_scores.append(silhouette_score)
                best_paramters.append([k, silhouette_score])
                print("Silhouette_score:", silhouette_score,
                      " Number of clusters:", k)
            print("---------------------------------------------------")
            final_param = max(
                enumerate(map(itemgetter(-1), best_paramters)), key=itemgetter(1))
            print("Best parameters:", best_paramters[final_param[0]])

            self.classifier = sklearn.cluster.KMeans(
                n_clusters=best_paramters[final_param[0]][0], random_state=42)
            print("Classifier: ", self.classifier.fit(features))
            print("---------------------------------------------------")

        # ToDo: Fix DBSCAN. Did not provide reasonable silhouette scores. Not used.
        if self.clustering_method == 'DBSCAN':
            # mins_samples usually set to min_samples = 2 * dimension of the dataset
            # length = dimension of the dataset
            ts_training_length = X.shape[2]
            print("Length:", ts_training_length)
            range_min_samples = [1, 2, 3, 4, 5, int(ts_training_length/4), int(
                ts_training_length/2), ts_training_length, ts_training_length*2]
            range_eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                         0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

            print("[DBSCAN] parameter tuning...")
            for min_sample in range_min_samples:
                for epsilon in range_eps:
                    self.classifier = sklearn.cluster.DBSCAN(
                        eps=epsilon, min_samples=min_sample
                    )
                    self.classifier = self.classifier.fit(features)
                    labels = self.classifier.labels_
                    print("Number of labels:", len(numpy.unique(labels)))
                    print("Labels:", labels)
                    if len(numpy.unique(labels)) != 1:
                        self.classifier = sklearn.cluster.DBSCAN(
                            eps=epsilon, min_samples=min_sample
                        )
                        silhouette_score = round(sklearn.metrics.silhouette_score(
                            features, labels
                        ), 3)
                        print("Number labels:", len(numpy.unique(labels)),
                              " Min_samples:", min_sample,
                              " Epsilon:", epsilon,
                              " Silhouette_score:", silhouette_score)
                        best_paramters.append(
                            [min_sample, epsilon, silhouette_score])
            print("################################")
            final_param = max(
                enumerate(map(itemgetter(-1), best_paramters)), key=itemgetter(1))
            print("Best parameters:", best_paramters[final_param[0]])

            self.classifier = sklearn.cluster.DBSCAN(
                eps=best_paramters[final_param[0]
                                   ][1], min_samples=best_paramters[final_param[0]][0]
            )
            print("Classifier: ", self.classifier.fit(features))

        return self.classifier.fit(features)

    def fit_encoder(self, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))
        print("Varying lengths:", varying)

        train = torch.from_numpy(X)
        print("train torch:", type(train))
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = X
        # print("Train_torch_dataset shape:", train_torch_dataset.shape)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )
        print("train_generator:",train_generator)
        print("train_generator", type(train_generator))
        print("train_generator", len(train_generator))
        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False
        
        silhouette_values = []

        # Encoder training
        print('Nb_steps: ', self.nb_steps)
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                print("Batch:", batch.shape)
                # print("Batch:", batch)
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                    print("Loss:", loss.item())
                    print("Loss tensor size:", loss.size())
                    print("Loss:", loss.item())
                else:
                    print("Varying loss")
                    loss = self.loss_varying(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                    print("Loss tensor size:", loss.size())
                    print("Loss tensor:", loss)
                    print("Loss:", loss.item())
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1

            # Early stopping strategy
            if self.early_stopping is not None:
                print("Early stopping...")
                # Computes the best regularization parameters
                features = self.encode(X, y)
                self.classifier = self.fit_clustering(features,X)
                cluster_labels = self.classifier.fit_predict(features)
                # Silhouette score
                score = self.silhouette_score(features, cluster_labels)
                score = round(score,2)
                print("Score [Early stopping strategy]:", score)

                silhouette_values.append(score)

                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.encoder)(**self.params)
                    best_encoder.double()
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.encoder.state_dict())
            if count == self.early_stopping:
                break 

        # Plotting silhouette scores
        # Did not provide useful results
        print("Sihouette scores...")
        print(len(silhouette_values))
        plt.plot(silhouette_values)
        plt.ylabel('silhouette score')
        plt.xlabel('epochs')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999',linestyle='-', alpha=0.2)
        plt.savefig("/home/sto/Data/2-sto-estimators-encoder/may01-whole_data/"+str(max_score)+"_"+self.architecture+"_Loss_" + str(self.lr)  + "_" + str(self.nb_steps) + "depth_" + str(self.depth) + ".png")

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder
     
        # Plot learned embeddings from train set
        # features = self.encode(X, y, plot=True, plot_title='TRAIN')
        # print(type(self.encoder))
        # print("encoder:",self.encoder)
        return self.encoder
    
    def fit(self, X, y, prefix_file, data='TRAIN', save_memory=False, verbose=False):
        """
        Trains sequentially the encoder unsupervisedly and then the classifier
        using the given labels over the learned features.

        @param X Training set.
        @param y Training labels.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        print("########## START: FITTING #########################")
        print(data, X.shape)
        print(data + " Anomalies:", X.shape[0])

        # Fitting encoder
        print("########## Fitting ENCODER ########################")
        self.encoder = self.fit_encoder(
            X, y=y, save_memory=save_memory, verbose=verbose
        )

        # Fitting clustering
        print("########## Fitting CLUSTERING #####################")
        features = self.encode(X, y)
        
        print(data + " EMBEDDINGS:", features.shape)
        self.classifier = self.fit_clustering(features, X)
       
        cluster_labels = self.classifier.fit_predict(features)
        print(data + " EMBEDDINGS CLUSTER LABELS", cluster_labels.shape, cluster_labels)
        
        cluster = list(set(cluster_labels))
        num_clusters = len(set(cluster_labels))
        print(num_clusters,"clusters:", cluster)

        score = self.silhouette_score(features, cluster_labels)
        print("Silhouette Score on" + data +  " DATASET:", score)
        numpy.save(prefix_file + '_' + 'Train_' + 'SilhouetteScore_' + str(score) + '_' +  str(len(set(cluster_labels))) + 'clusters_labels.npy', cluster_labels)

        if data == 'WHOLE DATASET':
            score = self.silhouette_score(features, cluster_labels)
            print("Silhouette Score on COMPLETE DATASET:", score)
            numpy.save(prefix_file + '_' + 'TrainedOn-WHOLE-Dataset_' + 'SilhouetteScore_' + str(score) + '_' +  str(len(set(cluster_labels))) + 'clusters_labels.npy', cluster_labels)

#        for i in range(num_clusters):
#             print("Cluster:",i)
#             features_per_cluster = features[cluster_labels_train == cluster[i]]
#             print("Features per cluster:", features_per_cluster.shape, features_per_cluster) 
        
        print("########## END: FITTING ##########################")

        return self

    def encode(self, X, batch_size=50, plot=False, plot_title='Storage'):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = (X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.encoder(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.encoder(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1
        if plot:
            self.plot_features(features, y, plot_title)

        self.encoder = self.encoder.train()
        return features
    def encode_window(self, X, window, batch_size=50, window_batch_size=10000):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        @param X Testing set.
        @param window Size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        """
        features = numpy.empty((
            numpy.shape(X)[0], self.out_channels,
            numpy.shape(X)[2] - window + 1
        ))
        masking = numpy.empty((
            min(window_batch_size, numpy.shape(X)[2] - window + 1),
            numpy.shape(X)[1], window
        ))
        for b in range(numpy.shape(X)[0]):
            for i in range(math.ceil(
                (numpy.shape(X)[2] - window + 1) / window_batch_size)
            ):
                for j in range(
                    i * window_batch_size,
                    min(
                        (i + 1) * window_batch_size,
                        numpy.shape(X)[2] - window + 1
                    )
                ):
                    j0 = j - i * window_batch_size
                    masking[j0, :, :] = X[b, :, j: j + window]
                features[
                    b, :, i * window_batch_size: (i + 1) * window_batch_size
                ] = numpy.swapaxes(
                    self.encode(masking[:j0 + 1], batch_size=batch_size), 0, 1
                )
        return features


    def plot_features(self, features, y, plot_title, folder_name='2-test-sto_estimators/'):
        print("Plot features ...")
        # Scale the features
        scaler = MinMaxScaler()
        features_scal = scaler.fit_transform(features)
        # Normalize the features
        transformer = Normalizer()
        features_norm = transformer.fit_transform(features_scal)
        print(features_norm.shape)

        print("T-SNE...")
        # Tnse
        tsne = TSNE(random_state=42, n_components=2, verbose=0,
                    perplexity=50, n_iter=1000).fit_transform(features_norm)

        # Plot
        plt.figure(figsize=(15, 10))
        plt.scatter(tsne[:, 0], tsne[:, 1],
                    c=y, s=8)
        plt.title("Storage_T-SNE")
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999',
                 linestyle='-', alpha=0.2)

        # Save plot
        print("Saving t-SNE plot...")
        plt.savefig("/home/sto/Data/" + folder_name + plot_title + ".png")


    def silhouette_score(self, features, labels):
        return sklearn.metrics.cluster.silhouette_score(features, labels)
    
    def score(self, X, prefix_file, test_labels):
        """
        Outputs accuracy of the SVM classifier on the given testing data.

         @param X Testing set.
         @param y Testing labels.
         @param batch_size Size of batches used for splitting the test data to
                avoid out of memory errors when using CUDA. Ignored if the
                testing set contains time series of unequal lengths.
        """
        print("########## SCORING ON TEST  #######################")
        # generating embeddings for test using trained encoder
        print("TEST Anomalies:", X.shape)
              
        features = self.encode(X, batch_size=self.batch_size, plot_title='TEST')
        print("TEST EMBEDDINGS:", features.shape, features)

        cluster_labels_test = self.classifier.fit_predict(features)
        print("TEST EMBEDDINGS CLUSTER LABELS:", cluster_labels_test.shape, cluster_labels_test)
        print("Number of clusters:",len(set(cluster_labels_test)), set(cluster_labels_test))

        score = self.silhouette_score(features, cluster_labels_test)
        numpy.save(prefix_file + '_' + 'Test_' + 'SilhouetteScore_' + str(score) + '_' +  str(len(set(cluster_labels_test))) + 'clusters_labels.npy', cluster_labels_test)

        if test_labels is not None:
            print("Labels:", test_labels.shape, test_labels)
            print("---------------------------------------------------")
            return sklearn.metrics.cluster.adjusted_rand_score(test_labels, cluster_labels)
        else:
            return self.silhouette_score(features, cluster_labels_test)

    def clusters_complete_dataset(self, dataset, prefix_file):
        print("########## COMPLETE DATASET #######################")
        # print("Train:", X.shape)
        # print("Test:", Y.shape)
        # dataset = numpy.concatenate((X,Y))
        print("Dataset:", dataset.shape)

        print("Number of Anomalies:", dataset.shape[0])

        features = self.encode(dataset, batch_size=self.batch_size)
        print("COMPLETE DATASET EMBEDDINGS:", features.shape, features)

        cluster_labels_dataset = self.classifier.fit_predict(features)
        print("COMPLETE DATASET EMBEDDINGS CLUSTER LABELS:", cluster_labels_dataset.shape, cluster_labels_dataset)
        print(type(cluster_labels_dataset))
        print("Number of clusters:",len(set(cluster_labels_dataset)), set(cluster_labels_dataset))

        score = self.silhouette_score(features, cluster_labels_dataset)
        print("Silhouette Score on COMPLETE DATASET:", score)

        numpy.save(prefix_file + '_' + 'TrainedOnWholeDataset_' + 'SilhouetteScore_' + str(score) + '_' +  str(len(set(cluster_labels_dataset))) + 'clusters_labels.npy', cluster_labels_dataset)

class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps a causal CNN encoder of time series as a PyTorch module and a
    SVM classifier on top of its computed representations in a scikit-learn
    class.

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
                 penalty=1, early_stopping=None, channels=10, depth=3,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, out_channels, cuda, gpu
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size
    
    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        print("INSIDE __create_encoder:")
        print("########## CausalCNN ENCODER Architecture ############")
        print(encoder)
        return encoder

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

    def encode_sequence(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder,
        from the start of the time series to each time step (i.e., the
        evolution of the representations of the input time series with
        repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), wich
        ensures that its output at time step i only depends on time step i and
        previous time steps.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = X
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # First applies the causal CNN
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    # Then for each time step, computes the output of the max
                    # pooling layer
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count * batch_size: (count + 1) * batch_size, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count: count + 1, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1

        self.encoder = self.encoder.train()
        return features

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


class LSTMEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps an LSTM encoder of time series as a PyTorch module and a SVM
    classifier on top of its computed representations in a scikit-learn
    class.

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
    @param cuda Transfers, if True, all computations to the GPU.
    @param in_channels Number of input channels of the time series.
    @param gpu GPU index to use, if CUDA is enabled.
    """

    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, in_channels=1, cuda=False,
                 gpu=0):
        super(LSTMEncoderClassifier, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(cuda, gpu), {}, in_channels, 160, cuda, gpu
        )
        assert in_channels == 1
        self.architecture = 'LSTM'

    def __create_encoder(self, cuda, gpu):
        encoder = networks.lstm.LSTMEncoder()
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

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
            'in_channels': self.in_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }
