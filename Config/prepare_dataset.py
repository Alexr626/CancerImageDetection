import os
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high, median, mode, mean
from math import e, floor
import random

from utils import is_dir_path

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

#Train-test split
SPLIT_NUMBER = 2 / 3


class MakeDataSet:
    def __init__(self, LIDC_Patients_list, META_DIR):
        self.IDRI_list = LIDC_Patients_list
        self.meta_path = META_DIR
        empty_annotatation_df = pd.DataFrame(index=[],columns=['Patient_id','Nodule_no', 'Nodule_id', 'Annotation_no',
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
                                                              'Texture_median_high', 'Malignancy_median_high', 'Is_cancer'])

        self.meta_annotation_train = empty_annotatation_df
        self.meta_annotation_test = empty_annotatation_df

        empty_nodule_df = pd.DataFrame(index=[],columns=['Patient_id','Nodule_no', 'Nodule_id', 'Is_cancer'])
        self.meta_nodule_train = empty_nodule_df
        self.meta_nodule_test = empty_nodule_df

        self.IDRI_list_train, self.IDRI_list_test = self.get_train_test_split(size = len(self.IDRI_list))



    def create_response(self, malignancy_df, nrow):
        malignancy_entropy = malignancy_df["Malignancy_entropy"][0]
        malignancy_mean = malignancy_df["Malignancy_mean"][0]
        if malignancy_entropy >= 1.5:
            is_cancer = "No_consensus"
        elif malignancy_mean >= 3.5:
            is_cancer = "True"
        elif malignancy_mean <= 2.5:
            is_cancer = "False"
        else:
            is_cancer = "Ambiguous"

        is_cancer_column = pd.Series(data=[is_cancer for _ in range(nrow)])

        return is_cancer_column


    # Input:
    def get_train_test_split(self, size, split = SPLIT_NUMBER):
        random.seed(42)
        indeces = list(range(0,size - 1))
        train_indeces = random.sample(indeces, floor(split * size))
        test_indeces = [index for index in indeces if index not in train_indeces]
        IDRI_list_train = [self.IDRI_list[i] for i in train_indeces]
        IDRI_list_test = [self.IDRI_list[i] for i in test_indeces]
        patient_id_train_test_df = pd.DataFrame(index=[], columns=['Patient_id','In_train'])
        for index in indeces:
            if index in train_indeces:
                in_train = 1
            else:
                in_train = 0
            curr_row = [self.IDRI_list[index], in_train]
            curr_row_df = pd.Series(data=curr_row, index=['Patient_id','In_train'])
            patient_id_train_test_df = patient_id_train_test_df.append(curr_row_df, ignore_index = True)

        patient_id_train_test_df.to_csv(self.meta_path+'patient_id_train_list.csv',index=False)
        return IDRI_list_train, IDRI_list_test

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
        malignancy_summary_df = summary_statistics_annotation_df[["Malignancy_mean", "Malignancy_entropy"]]
        meta_annotation_df["Is_cancer"] = self.create_response(malignancy_summary_df, nrow)

        return meta_annotation_df

    def save_meta(self, meta_df, is_nodule, is_train):
        if is_train:
            if is_nodule:
                self.meta_nodule_train = self.meta_nodule_train.append(meta_df,ignore_index=True)
            else:
                self.meta_annotation_train = self.meta_annotation_train.append(meta_df, ignore_index=True)
        else:
            if is_nodule:
                self.meta_nodule_test = self.meta_nodule_test.append(meta_df,ignore_index=True)
            else:
                self.meta_annotation_test = self.meta_annotation_test.append(meta_df, ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        nodule_id = 0

        for patient in tqdm(self.IDRI_list):
            pid = patient #LIDC-IDRI-0001~
            print(pid)
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            patient_id_dataset = pid[-4:]
            nodules_annotation = scan.cluster_annotations()

            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    nodule_id += 1
                    tmp_nodule = []
                    tmp_annotations = []

                    annotation_number = 0

                    tmp_nodule_df = pd.DataFrame(data=[], columns=['Patient_id','Nodule_no', 'Nodule_id', 'Is_cancer'])

                    for ann in nodule:
                        annotation_number += 1

                        meta_annotation_list = [patient_id_dataset, nodule_idx + 1, nodule_id, annotation_number,
                                                ann.internalStructure, ann.calcification, ann.subtlety, ann.margin,
                                                ann.sphericity, ann.lobulation, ann.spiculation, ann.texture, ann.malignancy]

                        # Remove annotation number as feature in nodule dataset
                        curr_nodule_list = meta_annotation_list[:3] + meta_annotation_list[4:]
                        tmp_nodule.append(curr_nodule_list)
                        tmp_annotations.append(meta_annotation_list)

                    tmp_annotations_df = pd.DataFrame(data=tmp_annotations, columns=['Patient_id','Nodule_no',
                                                                                     'Nodule_id', "Annotation_no",
                                                                                     'Internal structure','Calcification',
                                                                                     'Subtlety', 'Margin', 'Sphericity', 'Lobulation',
                                                                                     'Spiculation', 'Texture', 'Malignancy'])

                    if not tmp_annotations_df.empty:
                        meta_annotation_df = self.calculate_summary_statistics(tmp_annotations_df)
                        tmp_nodule_df = meta_annotation_df[['Patient_id','Nodule_no',
                                                            'Nodule_id', 'Is_cancer']].iloc[0]

                        if patient in self.IDRI_list_train:
                            self.save_meta(meta_annotation_df, is_nodule=False, is_train=True)
                        else:
                            self.save_meta(meta_annotation_df, is_nodule=False, is_train=False)

                    if patient in self.IDRI_list_train:
                        self.save_meta(tmp_nodule_df, is_nodule=True, is_train=True)
                    else:
                        self.save_meta(tmp_nodule_df, is_nodule=True, is_train=False)
                    # if not tmp_nodule_df.empty:
                    #     meta_nodule_list = self.calculate_summary_statistics(tmp_nodule_df)
                    #     meta_nodule_df = pd.DataFrame(data=meta_nodule_list, columns=['Patient_id','Nodule_no', 'Internal structure','Calcification',
                    #                                                            'Subtlety', 'Margin', 'Sphericity', 'Lobulation',
                    #                                                            'Spiculation', 'Texture', 'Malignancy'])
                    #     self.save_meta(meta_nodule_df, is_nodule=True)

        print("Saved Meta data")
        self.meta_annotation_train.to_csv(self.meta_path+'meta_annotation_info_train.csv',index=False)
        self.meta_annotation_test.to_csv(self.meta_path+'meta_annotation_info_test.csv',index=False)
        self.meta_nodule_train.to_csv(self.meta_path+'meta_nodule_info_train.csv',index=False)
        self.meta_nodule_test.to_csv(self.meta_path+'meta_nodule_info_test.csv',index=False)
        # self.meta_nodule.to_csv(self.meta_path+'meta_nodule_info.csv',index=False)



if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()
    LIDC_IDRI_list = LIDC_IDRI_list[1: len(LIDC_IDRI_list)]

    test = MakeDataSet(LIDC_IDRI_list,META_DIR)
    test.prepare_dataset()