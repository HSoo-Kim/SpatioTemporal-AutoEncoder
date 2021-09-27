# SpatioTemporal-AutoEncoder
Pytorch implementation of SpatioTemporal AutoEncoder

Paper: **Abnormal Event Detection in Videos using Spatiotemporal Autoencoder**

![image](https://user-images.githubusercontent.com/76432990/134844516-725fdaaa-983c-4b72-ae2a-cb478fac67ba.png)

I tried to implement the ST_AE with Pytorch, referring to keras code from https://github.com/aninair1905/Abnormal-Event-Detection.
There could be incorrect implementation in my code, so feel free to announce me about error.

## Missing Components
In referred Keras's code(https://github.com/aninair1905/Abnormal-Event-Detection),
there are dropout and recurrent_drop out in ConvLSTM block.
I cannot add them to my ConvLSTM.py. 
