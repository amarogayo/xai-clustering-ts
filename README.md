This repo provides code to: 1) cluster anomalies captured from variable length, high-dimensional sensor metrics (time series), annd 2) explainn clusters based on feature and temporal importance scores. A method tailored to learn 'good' representations of the input time series is used to ensure separability of the clusters. A triplet loss function enhanced with a clustering loss is learned by the DL model, while optimizing for representation learning and the clustering objective simultaneously. 

The general idea of the approach is the following: 

<img width="402" alt="genidea" src="https://user-images.githubusercontent.com/93252225/139062273-eb0ff317-543c-4ad2-bbb2-4c3dc2a70294.png">

The architecture of the causal encoder that optimizes for representation learning and clustering simultaneously is shown below:
<img width="360" alt="architecture" src="https://user-images.githubusercontent.com/93252225/139062610-d775f70f-4a8c-437a-ae51-12ee3f72a846.png">

Evaluation was conduced on UCR, UEA and storage-specific sensor metrics data. Results are better or comparable to state-of-the-art models. More details can be found here:
[clustering-ts.pdf](https://github.com/amarogayo/xai-clustering-ts/files/7425986/clustering-ts.pdf)
and
[explaining-anomaly-clusters.pptx](https://github.com/amarogayo/xai-clustering-ts/files/7425959/explaining-anomaly-clusters.pptx)
