import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high, median, median_grouped, mode, mean
from math import e, floor
from sklearn.model_selection import train_test_split
import random

from utils import is_dir_path,segment_lung
from pylidc.utils import consensus
from PIL import Image

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset','MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_MASK'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

#Train-test split
SPLIT_NUMBER = 2 / 3



class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        empty_annotatation_df = pd.DataFrame(index=[],columns=['Patient_id','Nodule_no', 'Annotation_no',
                                                              'Internal structure', 'Calcification', 'Subtlety',
                                                              'Margin', 'Sphericity', 'Lobulation', 'Spiculation',
                                                              'Texture', 'Internal structure_entropy','Calcification_entropy',
                                                              'Subtlety_entropy', 'Margin_entropy', 'Sphericity_entropy',
                                                              'Lobulation_entropy', 'Spiculation_entropy',
                                                              'Texture_entropy', 'Malignancy_entropy',
                                                              'Internal structure_mode','Calcification_mode',
                                                              'Subtlety_mode', 'Margin_mode', 'Sphericity_mode',
                                                              'Lobulation_mode', 'Spiculation_mode',
                                                              'Texture_mode', 'Malignancy_mode',
                                                              'Internal structure_mean','Calcification_mean',
                                                              'Subtlety_mean', 'Margin_mean', 'Sphericity_mean',
                                                              'Lobulation_mean', 'Spiculation_mean',
                                                              'Texture_mean', 'Malignancy_mean',
                                                              'Internal structure_median','Calcification_median',
                                                              'Subtlety_median', 'Margin_median', 'Sphericity_median',
                                                              'Lobulation_median', 'Spiculation_median',
                                                              'Texture_median', 'Malignancy_median',
                                                              'Internal structure_median_high','Calcification_median_high',
                                                              'Subtlety_median_high', 'Margin_median_high', 'Sphericity_median_high',
                                                              'Lobulation_median_high', 'Spiculation_median_high',
                                                              'Texture_median_high', 'Malignancy_median_high'])

        self.meta_annotation_train = empty_annotatation_df
        self.meta_annotation_test = empty_annotatation_df

        self.meta_nodule = pd.DataFrame(index=[],columns=['Patient_id','Nodule_no', 'Internal structure',
                                                          'Calcification', 'Subtlety', 'Margin', 'Sphericity',
                                                          'Lobulation', 'Spiculation', 'Texture'])

    def get_train_test_split(self, size, split = SPLIT_NUMBER):
        random.seed(42)
        indeces = list(range(0,size - 1))
        train_indeces = random.sample(indeces, floor(split * size))
        test_indeces = [index for index in indeces if index not in train_indeces]
        return train_indeces, test_indeces

    def calculate_variable_mode(self, values, nrow):
        mode_val = mode(values)
        mode_column = pd.Series(data=[mode_val for _ in range(nrow)])
        return mode_column
    def calculate_variable_mean(self, values, nrow):
        mean_val = mean(values)
        mean_column = pd.Series(data=[mean_val for _ in range(nrow)])
        return mean_column

    def calculate_variable_median_highs(self, values, nrow):
        median_high_val = median_high(values)
        median_high_column = pd.Series(data=[median_high_val for _ in range(nrow)])
        return median_high_column
    def calculate_variable_medians(self, values, nrow):
        median_val = median(values)
        median_column = pd.Series(data=[median_val for _ in range(nrow)])
        return median_column

    def calculate_variable_entropies(self, values, nrow):
        vc = pd.Series(values).value_counts(normalize=True, sort=False)
        base = 2
        entropy = -(vc * np.log(vc)/np.log(base)).sum()

        entropy_column = pd.Series(data=[entropy for _ in range(nrow)])
        return entropy_column
    def calculate_malignancy(self, malignancy_list):
        malignancy = median_high(malignancy_list)
        if malignancy > 3:
            return [malignancy, True]
        elif malignancy < 3:
            return [malignancy, False]
        else:
            return [malignancy, 'Ambiguous']

    def calculate_summary_statistics(self, tmp_annotations_df):
        relevant_variables = ['Internal structure','Calcification',
                              'Subtlety', 'Margin', 'Sphericity', 'Lobulation',
                              'Spiculation', 'Texture', 'Malignancy']

        summary_statistics_annotation_df = pd.DataFrame(data=[], columns=['Internal structure_entropy','Calcification_entropy',
                                                               'Subtlety_entropy', 'Margin_entropy', 'Sphericity_entropy',
                                                               'Lobulation_entropy', 'Spiculation_entropy',
                                                               'Texture_entropy', 'Malignancy_entropy',
                                                              'Internal structure_mode','Calcification_mode',
                                                              'Subtlety_mode', 'Margin_mode', 'Sphericity_mode',
                                                              'Lobulation_mode', 'Spiculation_mode',
                                                              'Texture_mode', 'Malignancy_mode',
                                                              'Internal structure_mean','Calcification_mean',
                                                              'Subtlety_mean', 'Margin_mean', 'Sphericity_mean',
                                                              'Lobulation_mean', 'Spiculation_mean',
                                                              'Texture_mean', 'Malignancy_mean',
                                                              'Internal structure_median','Calcification_median',
                                                              'Subtlety_median', 'Margin_median', 'Sphericity_median',
                                                              'Lobulation_median', 'Spiculation_median',
                                                              'Texture_median', 'Malignancy_median',
                                                              'Internal structure_median_high','Calcification_median_high',
                                                              'Subtlety_median_high', 'Margin_median_high', 'Sphericity_median_high',
                                                              'Lobulation_median_high', 'Spiculation_median_high',
                                                              'Texture_median_high', 'Malignancy_median_high'])

        nrow = tmp_annotations_df.shape[0]

        for label, values in tmp_annotations_df.items():
            if label in relevant_variables:
                entropy_label = label + "_entropy"
                entropy_column = self.calculate_variable_entropies(values, nrow)
                summary_statistics_annotation_df[entropy_label] = entropy_column

                mode_label = label + "_mode"
                mode_column = self.calculate_variable_mode(values, nrow)
                summary_statistics_annotation_df[mode_label] = mode_column

                mean_label = label + "_mean"
                mean_column = self.calculate_variable_mean(values, nrow)
                summary_statistics_annotation_df[mean_label] = mean_column

                median_label = label + "_median"
                median_column = self.calculate_variable_medians(values, nrow)
                summary_statistics_annotation_df[median_label] = median_column

                median_high_label = label + "_median_high"
                median_high_column = self.calculate_variable_median_highs(values, nrow)
                summary_statistics_annotation_df[median_high_label] = median_high_column

        meta_annotation_df = pd.concat((tmp_annotations_df, summary_statistics_annotation_df), axis=1)
        return meta_annotation_df

    def save_meta(self, meta_df, is_nodule, is_train):
        if is_nodule:
            self.meta_nodule = self.meta_nodule.append(meta_df,ignore_index=True)
        else:
            if is_train:
                self.meta_annotation_train = self.meta_annotation_train.append(meta_df, ignore_index=True)
            else:
                self.meta_annotation_test = self.meta_annotation_test.append(meta_df, ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        nPatients = len(self.IDRI_list)
        print("Number of patients: " + str(nPatients))
        train_indeces, test_indeces = self.get_train_test_split(nPatients)

        IDRI_list_train = [self.IDRI_list[i] for i in train_indeces]
        IDRI_list_test = [self.IDRI_list[i] for i in test_indeces]

        print("Training patient ids: " + str(IDRI_list_train))

        for patient in tqdm(self.IDRI_list):
            pid = patient #LIDC-IDRI-0001~
            print(pid)
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            patient_id_dataset = pid[-4:]
            nodules_annotation = scan.cluster_annotations()

            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    tmp_nodule = []
                    tmp_annotations = []

                    annotation_number = 0

                    for ann in nodule:
                        annotation_number += 1

                        meta_annotation_list = [patient_id_dataset, nodule_idx + 1, annotation_number,
                                                ann.internalStructure, ann.calcification, ann.subtlety, ann.margin,
                                                ann.sphericity, ann.lobulation, ann.spiculation, ann.texture, ann.malignancy]

                        # Remove annotation number as feature in nodule dataset
                        curr_nodule_list = meta_annotation_list[:2] + meta_annotation_list[3:]
                        tmp_nodule.append(curr_nodule_list)
                        tmp_annotations.append(meta_annotation_list)

                    tmp_nodule_df = pd.DataFrame(data=tmp_nodule, columns=['Patient_id','Nodule_no', 'Internal structure','Calcification',
                                                                 'Subtlety', 'Margin', 'Sphericity', 'Lobulation',
                                                                 'Spiculation', 'Texture', 'Malignancy'])

                    tmp_annotations_df = pd.DataFrame(data=tmp_annotations, columns=['Patient_id','Nodule_no', "Annotation_no",
                                                                                     'Internal structure','Calcification',
                                                                                     'Subtlety', 'Margin', 'Sphericity', 'Lobulation',
                                                                                     'Spiculation', 'Texture', 'Malignancy'])

                    if not tmp_annotations_df.empty:
                        meta_annotation_df = self.calculate_summary_statistics(tmp_annotations_df)
                        if patient in IDRI_list_train:
                            self.save_meta(meta_annotation_df, is_nodule=False, is_train=True)
                        else:
                            self.save_meta(meta_annotation_df, is_nodule=False, is_train=False)

                    # if not tmp_nodule_df.empty:
                    #     meta_nodule_list = self.calculate_summary_statistics(tmp_nodule_df)
                    #     meta_nodule_df = pd.DataFrame(data=meta_nodule_list, columns=['Patient_id','Nodule_no', 'Internal structure','Calcification',
                    #                                                            'Subtlety', 'Margin', 'Sphericity', 'Lobulation',
                    #                                                            'Spiculation', 'Texture', 'Malignancy'])
                    #     self.save_meta(meta_nodule_df, is_nodule=True)

        print("Saved Meta data")
        self.meta_annotation_train.to_csv(self.meta_path+'meta_annotation_info_train.csv',index=False)
        self.meta_annotation_test.to_csv(self.meta_path+'meta_annotation_info_test.csv',index=False)
        # self.meta_nodule.to_csv(self.meta_path+'meta_nodule_info.csv',index=False)



if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()
    LIDC_IDRI_list = LIDC_IDRI_list[1: len(LIDC_IDRI_list)]

    test = MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR,mask_threshold,padding,confidence_level)
    test.prepare_dataset()