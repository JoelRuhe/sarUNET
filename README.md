# sarUNET
The sarUNET is a deep learning UNET model developed for improving 3D cone beam CT image quality and brain tumor segmentation purposes. This UNET model consists of a number of convolution layers, a bottleneck and then a number of deconvolutional layers. At the end of the readme, an illustration of the UNET model is shown. The dataset that has been used for the cone beam CT improvment is the LIDC-IDRI data set. The data set that has been used for the brain tumor segmentation is the brats2015 data set.
<br><br>
This UNET model has been developed as part of my thesis and has been developed for SURFsara where I did my graduation internship.
## Cone beam CT improvement
This section describes what instructions are needed to run the 3D unet model and some results of the are shown.

### Instructions
Below are some instructions given to run the 3D UNET model:

```
cd 3D/
```

```
python main.py --dataset_root /nfs/managed_datasets/LIDC-IDRI/npy/average/ --scratch_path /scratch/USER/ --image_size 32 --batch_size 1 --gpu --horovod
```

This will make the 3D model run. Further information about the flags is given below:

**--dataset_root** = Path to the dataset (Required).  
**--scratch_path** = Path to where the data should be copied to (Required).  
**--image_size** = Size of the image to be trained. Can be 32, 64, 128, 256 and 512 where the depth is divided by four (eg: 32x32x8) (Required).  
**--batch_size** = Batch size (Required). 
**--gpu** = Use gpu configuration for training the model.  
**--horovod** = Use horovod for data parallelism over multiple Nodes and gpu's. Default is ```False```.      

### Results
Below are 3 images. The first is the original, second is with added noise and the third is the denoised one (remake). 

![Original image](https://github.com/JoelRuhe/sarUNET/blob/master/Results/CBCT%20Improvement/original.png)
![Image with added noise](https://github.com/JoelRuhe/sarUNET/blob/master/Results/CBCT%20Improvement/noise.png)
![Remade image](https://github.com/JoelRuhe/sarUNET/blob/master/Results/CBCT%20Improvement/remake.png)


![Image of UNET model](https://github.com/JoelRuhe/sarUNET/blob/master/Results/UNETmodel.png)

## Brain Tumor Segmentation
This section describes the instructions that are needed to run the Segmentation task.

### Instructions
```
cd Segmentation/
```

```
python main.py --dataset_root /path/to/npy/directory/ --scratch_path /scratch/USER/ --batch_size 1 --gpu --horovod
```

**--dataset_root** = Path to the npy dataset (Required).  
**--scratch_path** = Path to where the data should be copied to (Required).  
**--batch_size** = Batch size (Required). <br />
**--image_type** = The input image type. Can be 'T1', 'T1c', 'Flair', 'T2' (Default = T1c).<br />
**--image_tissue** = LGG tumors are slow growing tumors where as HGG are fast growing. Choises are 'HGG' or 'LGG' (Default = HGG). <br />
**--gpu** = Use gpu configuration for training the model.  
**--horovod** = Use horovod for data parallelism over multiple Nodes and gpu's. Default is ```False```.  

### Results
Below are 3 images. The first is the original T1c image, the second is the predicted segmentation and the third is the actual segmentation.

![Original image](https://github.com/JoelRuhe/sarUNET/blob/master/Results/Segmentation/original_T1c.png)
![Image with added noise](https://github.com/JoelRuhe/sarUNET/blob/master/Results/Segmentation/prediction.png)
![Remade image](https://github.com/JoelRuhe/sarUNET/blob/master/Results/Segmentation/segmentation_OT.png)
