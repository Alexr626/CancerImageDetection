import pylidc as pl
import numpy as np
from imageio import imsave
import pandas as pd
import dicom
import glob
from sklearn.model_selection import StratifiedKFold

np.random.seed(0)
# Function to extract nodules from LIDC dataset
# Input: None
# Output: List of tuples containing (case, nodule, malignancy, diameter, malignancy_th, centroid)
def extract_nodules():
    qu = pl.query(pl.Scan)
    im_size = 48
    total_nodules = 0
    # table = get_trans_table(qu)
    df = pd.read_csv('list3.2.csv', dtype={'case': str})
    train_patients = pd.read_csv('/Users/alex/dev/STAT 447B/Project/Data/Meta/patient_id_train_list.csv')
    train_patients["Patient_id_shortened"] = train_patients["Patient_id"].str[-4:]

    nodules = []
    for _, row in df.iterrows():
        # if row['eq. diam.'] > 30: continue

        case = row['case']

        #if case not in table: continue
        scan_id = row["scan"]
        print(case)
        #print(scan_id)
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == ("LIDC-IDRI-" + str(case))).first()
        if scan is None:
            print("Scan value: " + str(scan))
            continue
        try:
            print(scan.get_path_to_dicom_files())
        except:
            continue
        
        try:
            
            dcm = get_any_file(scan.get_path_to_dicom_files())
        except:
            continue
        
        if dcm == "Fail":
            print("Dcm value: " + str(dcm))
            continue
        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        nodules_names = row[8:][row[8:].notnull()].values

        if len(nodules_names) < 3: continue

        annotations = [ann for ann in scan.annotations if ann._nodule_id in nodules_names]

        if len(annotations) == 0: continue

        total_nodules += 1
        malignancy = np.mean([ann.malignancy for ann in annotations])

        # 0 - benign, 1 - ambiguous, 2 - malignant
        if malignancy >= 3.5:
            malignancy_th = 2
        elif malignancy <= 2.5:
            malignancy_th = 0
        else:
            malignancy_th = 1
        #print(dir(annotations[0]))
        annotations = [annt for annt in annotations if annt.bbox_dims(1).max() <= 31]

        if len(annotations) == 0:
            continue
        ann = annotations[0]
        #print(dir(ann))
        vol, seg = ann.uniform_cubic_resample(side_length=im_size - 1, verbose=0)
        view2d = vol
        view2d = hu_normalize(view2d, slope, intercept)
        #view2d *= seg
        nodules.append((case, view2d, malignancy, row['eq. diam.'], malignancy_th, ann.centroid))

    return nodules
# Function go from LIDC fomrat to PNG
# Input: Output directory
def lidc2Png(out_dir):
    nodules = extract_nodules()
    f = open(out_dir + '/labels.csv', 'w')
    for c, nodule, malignancy, diameter, malignancy_th  in enumerate(nodules):
        imsave("{0}/{1}.png".format(out_dir, c), nodule)
        line = "{0},{1},{2},{3}\n".format(c, malignancy, diameter, malignancy_th)
        f.write(line)
    f.close()

# Fucntion to go from LIDC format to NPY
# Input: Output directory

def Lidc2Voxel(out_dir):
    nodules = extract_nodules()
    # folds = get_kfold([n[3] for n in nodules], 10)
    f = open(out_dir + '/labels.csv', 'w')
    f.write('id,malignancy,diameter,malignancy_th,testing,x,y,z\n')
    # load in train patients and test
    train_patients = pd.read_csv('/Users/alex/dev/STAT 447B/Project/Data/Meta/patient_id_train_list.csv')
    train_patients["Patient_id_shortened"] = train_patients["Patient_id"].str[-4:]
    for c, (case, nodule, malignancy, diameter, malignancy_th,(x,y,z)) in enumerate(nodules):
        curr_in_train = int(train_patients[train_patients["Patient_id_shortened"] == case]["In_train"])
        np.save("{0}/{1}.npy".format(out_dir, c), nodule)
        line = "{0},{1},{2},{3},{4},{5},{6},{7}\n".format(
            c, malignancy, diameter, malignancy_th, curr_in_train,x,y,z)
        f.write(line)
    f.close()

# Create CrossValidation Folds
# Input: Labels, number of folds
def get_kfold(labels, k):
    stFold = StratifiedKFold(k, True)
    folds = [0]*len(labels)
    for i, (x,y) in enumerate(stFold.split(labels, labels)):
        for item in y:
            folds[item] = i
    return folds

# Get any file from a directory
# Input: Path to directory
def get_any_file(path):
    files = glob.glob(path + "/*.dcm")
    if len(files) < 1:
        return None
    try:
        dicom_file = dicom.read_file(files[0])
        #print("Patient dicom retrieved")
        return dicom_file
    except Exception as e:
        #print(e)
        print("Patient failed: " + files[1])
        return "Fail"


def get_trans_table(qu):
    table = {}

    
    for scan in qu:
        try:
            path = scan.get_path_to_dicom_files()
            dcm = get_any_file(path)
            
        except:
            continue
        if dcm is None:
            continue
        if dcm == "Fail":
            continue
        table[int(dcm.PatientID[10:])] = scan.id
    return table


def hu_normalize(im, slope, intercept):
    """normalize the image to Houndsfield Unit
    """
    im = im * slope + intercept
    im[im > 400] = 400
    im[im < -1000] = -1000

    im = (255 - 0)/(400 - (-1000)) * (im - 400) + 255

    return im.astype(np.uint8)


if __name__ == '__main__':
    import sys
    Lidc2Voxel("/Users/alex/dev/STAT 447B/Project/Data/Meta/vision_preprocess_output")
    # print(sys.argv)
    # if len(sys.argv) == 2:
    #     print(sys.argv[1])
    #     Lidc2Voxel(sys.argv[1])
    # else:
    #     print("run \"python3 LIDC.py <path to output directory>\"")