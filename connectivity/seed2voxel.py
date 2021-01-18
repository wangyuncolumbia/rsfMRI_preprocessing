
import nilearn as nl 
from nilearn import input_data

from nilearn.input_data import NiftiLabelsMasker
import pandas as pd 
import numpy as np
from nilearn import plotting
import os
from nibabel import funcs



def staticFC(seedIMG,seedNAME,funcdata,output):
    '''
    For example:
    seedIMG='Thalamus_atlas_regions.nii'
    seedNAME="Thalamus_atlas_regions.txt"
    '''

    masker = NiftiLabelsMasker(labels_img=seedIMG, standardize=True)
    my_file = pd.read_table(seedNAME,header=None)
    seedname= list(my_file[0])
    seed_time_series = masker.fit_transform(funcdata)

    # remember to check the TR in this code. 
    brain_masker = input_data.NiftiMasker(
        smoothing_fwhm=6,
        detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2,
        memory='nilearn_cache', memory_level=1, verbose=2)
    
    brain_time_series = brain_masker.fit_transform(funcdata)

    #print("Seed time series shape: (%s, %s)" % seed_time_series.shape)
    #print("Brain time series shape: (%s, %s)" % brain_time_series.shape)


    seed_to_voxel_correlations = (np.dot(brain_time_series.T, seed_time_series) /
                                  seed_time_series.shape[0]
                                  )

    # print("Seed-to-voxel correlation shape: (%s, %s)" %
    #       seed_to_voxel_correlations.shape)
    # print("Seed-to-voxel correlation: min = %.3f; max = %.3f" % (
    #     seed_to_voxel_correlations.min(), seed_to_voxel_correlations.max()))

    seed_to_voxel_correlations_img = brain_masker.inverse_transform(
        seed_to_voxel_correlations.T)

    seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)

    # Finally, we can tranform the correlation array back to a Nifti image
    # object, that we can save.
    seed_to_voxel_correlations_fisher_z_img = brain_masker.inverse_transform(
        seed_to_voxel_correlations_fisher_z.T)

    #seed_to_voxel_correlations_fisher_z_img.to_filename('seed_correlation_z.nii.gz')

    IM=funcs.four_to_three(seed_to_voxel_correlations_fisher_z_img)
    for i,M in enumerate(IM):
        outdir=f'{output}/{seedname[i]}/seedFC'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        subname=os.path.basename(funcdata)[:-4]
        if not os.path.exists(f'{outdir}/{subname}_SFC_Z.nii.gz'):
            M.to_filename(f'{outdir}/{subname}_SFC_Z.nii.gz')
        else:
            print('Existed')