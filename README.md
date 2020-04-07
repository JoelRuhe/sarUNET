# sarUNET
The sarUNET is a deep learning UNET model developed for improving 3D Conebeam CT image quality so they will get the quality of a normal CT image. This UNET model consists of a number of convolution layers, a bottleneck and then a number of deconvolutional layers. At the end of the readme, an illustration of the UNET model is shown.

## Instructions
Below are some instructions given to run the 3D UNET model:

```
cd 3D/
```

```
python main.py --dataset_root /nfs/managed_datasets/LIDC-IDRI/npy/average/ --scratch_path /scratch/USER/ --image_size 32 --batch_size 1 --gpu --horovod
```

This will make the 3D model run. Further information about the flags is given below:

**--dataset_root** = Path to the dataset (Required).  
**--image_size** = Size of the image to be trained. Can be 32, 64, 128, 256 and 512 where the depth is divided by four (eg: 32x32x8) (Required).  
**--batch_size** = Batch size.  
**--gpu** = Use gpu configuration for training the model.  
**--horovod** = Use horovod for data parallelism over multiple Nodes and gpu's. Default is ```False```.      
**--scratch_path** = Path to where the data should be copied to.  




![Image of UNET model](https://github.com/JoelRuhe/sarUNET/blob/master/images/UNET%20model.png)
