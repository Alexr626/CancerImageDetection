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
from statistics import median_high, median, median_grouped, mode

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
        self.meta_annotation = pd.DataFrame(index=[],columns=['Patient_id','Nodule_no', 'Annotation_no',
                                                              'Internal structure', 'Calcification', 'Subtlety',
                                                              'Margin', 'Sphericity', 'Lobulation', 'Spiculation',
                                                              'Texture', 'Is_cancer'])
        self.meta_nodule = pd.DataFrame(index=[],columns=['Patient_id','Nodule_no', 'Internal structure',
                                                          'Calcification', 'Subtlety', 'Margin', 'Sphericity',
                                                          'Lobulation', 'Spiculation', 'Texture', 'Is_cancer'])

    def calculate_categorical_grouping(self, categorical_data):
        columns = categorical_data.columns
        list_grouped_categorical_columns = []
        for name, feature in categorical_data.iteritems():
            feature_mode = mode(feature)
            list_grouped_categorical_columns.append(feature_mode)

        return list_grouped_categorical_columns

    def calculate_ordinal_grouping(self, ordinal_data):
        columns = ordinal_data.columns
        list_grouped_ordinal_columns = []
        for name, feature in ordinal_data.iteritems():
            feature_median = median(feature)
            list_grouped_ordinal_columns.append(feature_median)

        return list_grouped_ordinal_columns

    def calculate_malignancy(self, malignancy_list):
        malignancy = median_high(malignancy_list)
        if malignancy > 3:
            return [malignancy, True]
        elif malignancy < 3:
            return [malignancy, False]
        else:
            return [malignancy, 'Ambiguous']

    def calculate_summary_statistics(self, current_nodule_annotations):
        list_categorical_data = self.calculate_categorical_grouping(
            current_nodule_annotations[["Internal structure", "Calcification"]])

        list_ordinal_data = self.calculate_ordinal_grouping(current_nodule_annotations[['Subtlety', 'Margin',
                                                                   'Sphericity', 'Lobulation', 'Spiculation', 'Texture']])

        list_malignancy_cancer = self.calculate_malignancy(current_nodule_annotations['Malignancy'])

        patient_nodule_info = current_nodule_annotations[['Patient_id', 'Nodule_no']].iloc[0].tolist()

        meta_nodule_list = patient_nodule_info + list_categorical_data + list_ordinal_data + list_malignancy_cancer

        # print("Current nodule data after summarization: ")
        # print(meta_nodule_list)

        return meta_nodule_list

    def save_meta(self, meta_list, is_nodule):
        """Saves the information of nodule to csv file"""

        if is_nodule:
            tmp_nodule = pd.Series(data=meta_list, index=['Patient_id','Nodule_no', 'Internal structure',
                                                               'Calcification', 'Subtlety', 'Margin', 'Sphericity',
                                                               'Lobulation', 'Spiculation', 'Texture', 'Malignancy',
                                                               'Is_cancer'])
            self.meta_nodule = self.meta_nodule.append(tmp_nodule,ignore_index=True)

        else:
            tmp_annotation = pd.Series(data=meta_list, index=['Patient_id','Nodule_no', 'Annotation_no',
                                                                   'Internal structure', 'Calcification',  'Subtlety',
                                                                   'Margin', 'Sphericity',
                                                                   'Lobulation', 'Spiculation', 'Texture', 'Malignancy',
                                                                   'Is_cancer'])
            self.meta_annotation = self.meta_annotation.append(tmp_annotation, ignore_index=True)

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

                    annotation_number = 0

                    for ann in nodule:
                        annotation_number += 1

                        is_cancer = self.calculate_malignancy([ann.malignancy])

                        meta_annotation_list = [patient_id_dataset, nodule_idx + 1, annotation_number,
                                                ann.internalStructure, ann.calcification, ann.subtlety, ann.margin,
                                                ann.sphericity, ann.lobulation, ann.spiculation, ann.texture] + is_cancer
                        self.save_meta(meta_annotation_list, is_nodule=False)

                        curr_nodule_list = meta_annotation_list[:2] + meta_annotation_list[3:]
                        tmp_nodule.append(curr_nodule_list)

                    tmp_nodule_df = pd.DataFrame(data=tmp_nodule, columns=['Patient_id','Nodule_no', 'Internal structure','Calcification',
                                                                 'Subtlety', 'Margin', 'Sphericity', 'Lobulation',
                                                                 'Spiculation', 'Texture', 'Malignancy', 'Is_cancer'])

                    if not tmp_nodule_df.empty:
                        meta_nodule_list = self.calculate_summary_statistics(tmp_nodule_df)
                        self.save_meta(meta_nodule_list, is_nodule=True)

        print("Saved Meta data")
        self.meta_annotation.to_csv(self.meta_path+'meta_annotation_info.csv',index=False)
        self.meta_nodule.to_csv(self.meta_path+'meta_nodule_info.csv',index=False)



if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()
    LIDC_IDRI_list = LIDC_IDRI_list[1: len(LIDC_IDRI_list)]

    test = MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR,mask_threshold,padding,confidence_level)
    test.prepare_dataset()