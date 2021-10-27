import shap
import torch
import json
import numpy
from sklearn.preprocessing import MinMaxScaler
import sys
import time
import h5py
import random

# versions
print("Torch:",torch.__version__)
print("Numpy:",numpy.__version__)
print("Shap:", shap.__version__)

# paths
data_path = '/home/sto/Data/2-nonValidatedAnomalies_reducedFeatures_sergeValidated_ZERO/'
dataset = '2-nonValidatedAnomalies_reducedFeatures_sergeValidated_ZERO'

model_py_path = '/home/sto/PyScripts/1-sto-cl-copy-autoencoder/'
model_path = '/home/sto/Data/1-sto-estimators-autoencoder/'
model_name = '1-allAnomalies_reducedFeatures_sergeValidated_ZERO_CausalCNN_Autoencoder_autoencoder.pth'

shap_values_save_path = '/home/sto/Data/2-SHAP/FINAL_FINAL_FINAL/'
shap_values_save_name = 'shap-values_all-anomalies_samples-to-explain_CLUSTER_background-data_25'

cluster_labels_path = "/home/sto/Data/2-sto-estimators-encoder/may01-whole_data/"
cluster_labels = "2-nonValidatedAnomalies_reducedFeatures_sergeValidated_TrainedOn-WHOLE-Dataset_SilhouetteScore_0.8879986216174587_3clusters_labels.npy"

print("################################ PATHS #############################################")
print("Data Path:", data_path)
print("Dataset Name:", dataset)
print("------------------------------------------------------------------------------------")
print("Model.py Path:", model_py_path)
print("Model Path:", model_path)
print("Model Name:", model_name)
print("------------------------------------------------------------------------------------")
print("Save Path Shap Values:", shap_values_save_path)
print("Name Shap Values File:", shap_values_save_name)
print("------------------------------------------------------------------------------------")
print("Cluster Labels Path:", cluster_labels_path)
print("Cluster Labels:", cluster_labels)
print("####################################################################################")

# helper functions
def load_dataset(path, dataset):
    with open(path + dataset + "_TRAIN.npy", 'rb') as f:
        train = numpy.load(f, allow_pickle=True)
    print("Train shape:", train.shape)
    print(train)
    with open(path + dataset + "_TEST.npy", 'rb') as f:
        test = numpy.load(f, allow_pickle=True)
    print("Test shape:", test.shape)
    print(test)
    #scaling
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train.reshape(-1,train.shape[-1])).reshape(train.shape)
    test = scaler.fit_transform(test.reshape(-1,test.shape[-1])).reshape(test.shape)
    return train, test

def load_autoencoder_model(model_py_path, model_path, model_name):
    sys.path.insert(0, model_py_path)
    model = torch.load(model_path + model_name)
    print(model['model'])
    return model['model']

def assemble_dataset(train, test):
    print("Train:", train.shape)
    #print(train)
    print("Test:", test.shape)
    #print(test)
    dataset = numpy.concatenate((train,test))
    print("Complete Dataset:", dataset.shape, dataset)
    return dataset

def load_cluster_labels(path, cluster_labels):
    print("Silhouette Score:", cluster_labels.split('SilhouetteScore_')[1].split('_')[0])
    cluster_labels = numpy.load(path + cluster_labels)
    print("Number of clusters:", len(set(cluster_labels)))
    print("Clusters:", set(cluster_labels))
    print("Cluster Labels:",cluster_labels)
    return cluster_labels

def get_anomaly_cluster_association(cluster_labels, cluster):
    # get position (index) of each anomaly belonging to a cluster
    cluster_indices = [index for index, value in enumerate(cluster_labels) if value == cluster]
    print(len(cluster_indices),"samples in cluster", cluster)
    return cluster_indices

train, test = load_dataset(data_path, dataset)
dataset = assemble_dataset(train, test)
dataset = torch.from_numpy(dataset)
print("Dataset:", dataset.shape)
print(dataset)

model = load_autoencoder_model(model_py_path, model_path, model_name)

print("Clusters...")
cluster_labels = load_cluster_labels(cluster_labels_path, cluster_labels) 

# for i in set(cluster_labels:
# i is the label of the cluster

i = 2

print("Cluster", i)
cluster_indices = get_anomaly_cluster_association(cluster_labels, i)
print(cluster_indices)

# dataset containing samples from only this cluster
cluster_data = []
for x in cluster_indices:
    cluster_data.append(dataset[x].detach().numpy())
cluster_data = numpy.array(cluster_data)

print("Cluster Data", i, cluster_data.shape)

if cluster_data.shape[0] > 25:
    x = cluster_data.shape[0] - 24
    background_data_start = random.randint(1,x)
    print("Random generated cut start:", background_data_start)
    background_data = cluster_data[background_data_start:background_data_start+25]

else:
    print("<25 samples. Use all cluster data as background")
    background_data = cluster_data

print("Free up sapce from dataset variable...")
dataset = []
print("Dataset:", len(dataset))

print("Shap Values Computation for Samples from Cluster... ")
s = time.time()
counter = 0

for sample in cluster_data:
    print("Cluster", i)
    print("Sample #", cluster_indices[counter])
    print("#######################################################")
    print("Sample Shape:", sample.shape)

    if (type(sample) != numpy.ndarray):
        sample = sample.detach().numpy()
    if (type(background_data) != numpy.ndarray):
        background_data = background_data.detach().numpy()

    print(type(sample))
    print(type(background_data))

    if numpy.any(numpy.all(background_data == sample, axis=(1, 2))):
        index_sample = numpy.where(numpy.all(background_data == sample, axis=(1, 2)))[0][0]
        print("Index of Sample that is included in Background Data:", index_sample)
        
        background_data_24 = numpy.delete(background_data, 0, axis=0)
        print("USED BACKGROUND DATA:",background_data_24.shape)
        background_data_24 = torch.from_numpy(background_data_24)
        
        start = time.time()
        explainer = shap.DeepExplainer(model, background_data_24)
        end = time.time()
        
        processing_time = end - start
        print("Processing Time DeepExplainer Object Creation = {}".format(processing_time))
        pt_exp = {'DeepExplainer Object Creation Time': str(processing_time)}
        with open(shap_values_save_path + '_24_BACKGROUND_SAMPLE_DeepExplainer_Object_Creation_TIME'+str(processing_time)+'.json', 'w') as f:
            json.dump(pt_exp, f)
    else:
        background_data = torch.from_numpy(background_data)
        print("USED BACKGROUND DATA:",background_data.shape)

        start = time.time()
        explainer = shap.DeepExplainer(model, background_data)
        e = time.time()
        
        processing_time = e-s
        print("Processing Time DeepExplainer Object Creation = {}".format(processing_time))
        pt_exp = {'DeepExplainer Object Creation Time': str(processing_time)}
        with open(shap_values_save_path + 'DeepExplainer_Object_Creation_TIME'+str(processing_time)+'.json', 'w') as f:
            json.dump(pt_exp, f)

    sample = numpy.expand_dims(sample, axis=0)
    sample = torch.from_numpy(sample)
    print("Data to Explain for Shap Values Computation:", sample.shape)
    shap_values = explainer.shap_values(sample)
    # Numpy Save
    with open(shap_values_save_path + shap_values_save_name + '_Anomaly__'+str(cluster_indices[counter])+'Cluster_'+str(i)+'.npy', 'wb') as f:
        numpy.save(f, shap_values)
    print('SAVED:',shap_values_save_path + shap_values_save_name + '_Anomaly__'+str(cluster_indices[counter])+'Cluster_'+str(i))
    counter = counter + 1

e = time.time()
processing_time = e-s
print("Processing Time Shap Values Computation = {}".format(processing_time))

pt_shap = {'SHAP Values Computation Time': str(processing_time)}
with open(shap_values_save_path + 'SHAP_Values_Computation_TIME'+str(processing_time)+'.json', 'w') as f:
    json.dump(pt_shap, f)
