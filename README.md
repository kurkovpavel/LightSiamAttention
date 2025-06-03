# LightSiamAttention
This LightSiamAttention visual tracker is based on siamse family visual trackers but with custom backbone. Backbone has 128 channel self-attention block which extract features better then convoulutional blocks the same dimension. 
Pytorch model was traing during 15 epoch on got10k dataset.

Run with usb camera: python demo.py

Train: change path to your got10k dataset: python trainCustom.py
