# Reflectance Adaptive Filtering Improves Intrinsic Image Estimation
This is the code release for the research project of "Reflectance Adaptive Filtering Improves Intrinsic Image Estimation", which appears at [CVPR 2017](<http://cvpr2017.thecvf.com>) and can be found at <https://arxiv.org/abs/1612.05062>.

It is released under the MIT License, please see `LICENSE.md`.

If you find this useful, please consider citing

```
@inproceedings{nestmeyer2017reflectanceFiltering,
  title={Reflectance Adaptive Filtering Improves Intrinsic Image Estimation},
  author={Nestmeyer, Thomas and Gehler, Peter V},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```


## Direct CNN prediction
To get the reflectance estimation with our trained CNN, the script `decompose_with_trained_CNN.py` needs the necessary parameters:
`python decompose_with_trained_CNN.py --filename_in=/path/to/read/file/from/name.png --path_out=/folder/where/output/should/go/to`

It should only depend on `caffe` (<https://github.com/BVLC/caffe>), `numpy` and `cv2` (OpenCV).

To give an explicit example, I am able to run:
`python ~/Repositories/intrinsic/src/reflectance_filtering/decompose_with_trained_CNN.py --filename_in=/lustre/home/tnestmeyer/Repositories/intrinsic/src/malabar/python_layers/unittests/118495.png --path_out=/lustre/home/tnestmeyer/Results/tmp_folder`

The input image is expected to be a regular gamma-corrected (meaning sRGB) image, while the output gives linear response.


## Reflectance Filtering
Applies the piecewise constant reflectance assumption by using a bilateral/guided filter.

An example usage which should be instructive in how to pass the parameters:
`python filter_reflectance.py --filter_type=bilateral --sigma_color=20 --sigma_spatial=22 --filename_in=/lustre/home/tnestmeyer/Results/tmp_folder/118495-r.png --guidance_in=/lustre/home/tnestmeyer/Repositories/intrinsic/src/malabar/python_layers/unittests/118495.png --path_out=/lustre/home/tnestmeyer/Results/tmp_folder`

It depends on `numpy` and OpenCV with the ximgproc (Extended Image Processing) module (<https://github.com/opencv/opencv_contrib/tree/master/modules/ximgproc>)

## WHDR computation
To evaluate the WHDR of our and other methods, we used the `compute_whdr` function of the released code in the [IIW dataset](<http://opensurfaces.cs.cornell.edu/publications/intrinsic/>) under `iiw-dataset/whdr.py` after loading the image with
`load_image(filename_reflectance, is_srgb=False)`
or (to evaluate other methods which might be saved as sRGB)
`load_image(filename_reflectance, is_srgb=True)`
respectively.

If you want to reproduce our results in Figure 5, most of the previous methods can be downloaded from [Sean Bell's project page](<http://opensurfaces.cs.cornell.edu/publications/intrinsic/>) under `Pre-computed decompositions (release 0, 4.5M)`.
Methods not available there:
   - [Zoran et al. 2015] provide the following images for their test set: <https://people.csail.mit.edu/danielzoran/images.tar.gz>
   - [Bi et al. 2015] provide their results on <http://cseweb.ucsd.edu/~bisai/project/siggraph2015.html> for their full intrinsic decomposition pipeline.

Additionaly, we provide the following results:
   - L1 flattening: `bi2015_l1_only_flattening_linear` is the result of the pure L1 flattening step we also recap in our paper (not their full pipeline above).
    [Download link](<http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/results/bi2015_l1_only_flattening_linear.tar.gz>)
   - Direct CNN prediction: `nestmeyer2016_whdrCNN_1109_rDirectly_from_1108_single_channel_wdm_12_06_then_13_08_linear` is what our 1x1 CNN predicts as reflectance
    [Download link](<http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/results/nestmeyer2016_whdrCNN_1109_rDirectly_from_1108_single_channel_wdm_12_06_then_13_08_linear.tar.gz>)
   - BF(CNN, CNN): `nestmeyer2016_whdrCNN_1109_smoothed_1x_with_ours_c20s22_rDirectly_from_1108_single_channel_wdm_12_06_then_13_08_linear` is after bilateral filtering of our Direct CNN prediction with itself as features
    [Download link](<http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/results/nestmeyer2016_whdrCNN_1109_smoothed_1x_with_ours_c20s22_rDirectly_from_1108_single_channel_wdm_12_06_then_13_08_linear.tar.gz>)
   - GF(CNN, flat): `ours_guided_c3.0s45.0_bi_flat_linear` is after using the guided filter on the Direct CNN prediction with `flat` as features
    [Download link](<http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/results/ours_guided_c3.0s45.0_bi_flat_linear.tar.gz>)
   - Rescaling to [0.55, 1]: `baseline_translated_0.54_linear`
    [Download link](<http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/results/baseline_translated_0.54_linear.tar.gz>)
   - BF(Bi et al. 2015, flat): `bi2015_l1_final_linear_smoothed_with_Bi_flat_linear`
    [Download link](<http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/results/bi2015_l1_final_linear_smoothed_with_Bi_flat_linear.tar.gz>)
   - BF(Zoran et al. 2015, flat)*: `zoran2015_ordinal_onlyHisTest_smoothed_1x_with_Bi_flat_c15s28_linear`
    [Download link](<http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/results/zoran2015_ordinal_onlyHisTest_smoothed_1x_with_Bi_flat_c15s28_linear.tar.gz>)
   - 3x GF(Zoran et al. 2015, flat)*: `zoran_guided_c3.0s45.0_bi_flat_linear_guided_c3.0s45.0_bi_flat_linear_guided_c3.0s45.0_bi_flat_linear`
    [Download link](<http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/results/zoran_guided_c3.0s45.0_bi_flat_linear_guided_c3.0s45.0_bi_flat_linear_guided_c3.0s45.0_bi_flat_linear.tar.gz>)
 


## Training the CNN for the direct reflectance prediction
The code to train the CNN described in the paper is not that clean. In the following, there should be hints for how to use it. If you encounter problems which you cannot solve yourself or missing files, come back to me at
thomas. nestmeyer at tuebingen. mpg. de (remove spaces, at -> @ )
or any other channel you might find in case this address might be out of service.

1. Create numpy array files containing the train / validation / test split of IIW including the annotations with
`createNumpyArrayWithComparisons.py`:
   - Adapt the necessary paths (`DATA_FOLDER` for where to find the IIW dataset and `SAVE_TO_FOLDER` for where the created numpy arrays should be saved to).
   - With `AUGMENT_DATA` you can decide if you want to use the augmented comparisons as described in section 1.1 of the supplementary material. Since it does not lead to better performance, but creation takes really long, better keep it to be `False` if you do not explicitly want to try it.
   - `CREATE = ['trainValTest']` creates the IIW split we describe in the paper. You can use `'dummy'` first to check that everything works well, because for 'trainValTest' you need a lot of memory and creation takes quite some time.
   - You can try `PARALLEL = True` to create the intermediate files (especially useful for augmenting the data), but in the end you need to use `PARALLEL = False` to create the final output due to some race condition (but it always tries to read from the intermediate files).
   - If you have trouble creating the arrays (e.g. due to memory limitations), you can download them (including the augmented comparisons) at:
        * <http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/data/trainValTest_train_256_256_linear.npz>
        * <http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/data/trainValTest_val_256_256_linear.npz>
        * <http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/data/trainValTest_test_256_256_linear.npz>


2. Adapt in
    `data_handling.py`
    where to find the `*.npz` files created/downloaded in 1.

3. Install the latest caffe:
    <https://github.com/BVLC/caffe>
    
    If you have problems, use malabar, our caffe "flavor", already prepared for the next step:
    <http://files.is.tue.mpg.de/tnestmeyer/public/reflectance-filtering/malabar/malabar.tar.gz>
    
    I installed malabar via `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_python=ON -DCUDA_TOOLKIT_ROOT_DIR=/is/software/nvidia/cuda-8.0.44 -DUSE_CUDNN=True -DCUDNN_LIBRARY=/is/software/nvidia/cudnn-5.1/lib64/libcudnn.so -DCUDNN_INCLUDE=/is/software/nvidia/cudnn-5.1/include -DCUDA_ARCH_NAME=Kepler ..`
4. Install barrista, a more convenient python interface to caffe: <https://classner.github.io/barrista/>

5. Install other dependencies (some of which might also be easy to be replaced/removed if unwanted): `simplejson`, `tqdm`, `scipy`.
6. In `train_with_barrista.py` adapt the paths where to find barrista and caffe (or malabar)

7. To start the actual training, run `barrista_train_test.py` with the parameters of your choice.
   - If you want to only estimate the reflectance intensity (grayscale, as e.g. [Zoran et al. 2015]), use `--RS_est_mode=rDirectly`, otherwise if you want to recover RGB reflectance and shading like [Bell et al. 2014], use `--RS_est_mode=rRelMean`.
   - A training command to train a network with 4 layers, 2^4 filters each, being 1x1 convolutions (0 padding) looks for example like this:
        `python training/train_with_barrista.py --stage=fit --iterations=10000 --batch_size=10 -exp=experiment_name --numLayers=4 --num_filters_log=4 --kernel_pad=0 --RS_est_mode=rDirectly`
   - Results are saved to ~/Results/experiment_name
   - If you want to use one of the trained caffemodels to decompose an image, use (adapt the network parameters to what you used in training):
        `python training/train_with_barrista.py --stage=predict --iterations=1 --batch_size=1 -exp=experiment_name2 --numLayers=4 --num_filters_log=4 --kernel_pad=0 --RS_est_mode=rDirectly --decompose=/path/to/image/to/decompose.png --predictCaffemodel=/path/to/trained/weights.caffemodel`
