## Brain parcellation demo
![Brain parcellation](./example_outputs/screenshot.png)

Visualisation of segmentation results generated by this demo.


#### Overview
This demo employs a high resolution 3D network using the method
described in
```
Li et al., On the Compactness, Efficiency, and Representation of 3D
Convolutional Networks: Brain Parcellation as a Pretext Task,
In: Information Processing in Medical Imaging (IPMI) 2017.
```
DOI: [10.1007/978-3-319-59050-9_28](http://doi.org/10.1007/978-3-319-59050-9_28)

The following sections provide instructions for downloading an MR volume and a
trained network model, and then using the NiftyNet inference program to
generate brain parcellation.


#### Prepare data and run segmentation network
0) **Create a folder for the demo, as an example `/home/niftynet/demo/` is used here:**
```bash
# tentative path for demo purpose
export demopath=/home/niftynet/demo
mkdir ${demopath}
```

1) **Download an MR volume (originally from [OASIS](http://www.oasis-brains.org/) dataset; about 7MB)
and a trained model of HighRes3DNet (about 64MB):**

```bash
# wget is used as an example, but could be replaced by other downloaders
wget -c https://www.dropbox.com/s/rxhluo9sub7ewlp/parcellation_demo.tar.gz -P ${demopath}
# extract the downloaded volume
cd ${demopath}; tar -xvf parcellation_demo.tar.gz
```
 * To demonstrate the generalisation ability of the segmentation
   network, this dataset is not used in training.
 * The MR volume is located at /home/niftynet/demo/OASIS


2) **Update data input folders in the demo folder**

Change `path_to_search` in the configuration file to the downloaded data,
in this example, it will be in the file `/home/niftynet/demo/highres3dnet_config_eval.ini`:

changing from
```ini
path_to_search=OASIS
```
to
```ini
path_to_search=/home/niftynet/demo/OASIS
```

3) **Run NiftyNet inference program**

Using pip installed NiftyNet:
```bash
pip install NiftyNet
cd ${demopath};
net_segment inference -c ${demopath}/highres3dnet_config_eval.ini \
        --save_seg_dir ${demopath}/results
```

or using NiftyNet cloned from [GitHub](https://github.com/NifTK/NiftyNet) or [CMICLab](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet):
```bash
cd NiftyNet/
# additional parameters to set absolute paths of required files
python net_segment.py inference -c ${demopath}/highres3dnet_config_eval.ini \
        --histogram_ref_file ${demopath}/databrain_std_hist_models_otsu.txt \
        --model_dir ${demopath} \
        --save_seg_dir ${demopath}/results
```
The parcellation output will be stored in `/home/nifty/demo/results`.


_Please Note:_

* To achieve an efficient parcellation, a GPU with at least 10GB memory is required.

* Please change the environment variable `CUDA_VISIBLE_DEVICES` to an appropriate value if necessary (e.g., `export CUDA_VISIBLE_DEVICES=0` will allow NiftyNet to use the `0`-th GPU).