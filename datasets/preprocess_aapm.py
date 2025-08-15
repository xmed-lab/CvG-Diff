import os
import numpy as np
import SimpleITK as sitk

def ima2array(ima_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(ima_path))

if __name__ == '__main__':
    # Input your directory to unzipped AAPM16 dataset here, download link: https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144226105715
    # Unzip the FD_1mm.zip to full_1mm
    patients_dir = 'Dataset_dir_to_downloaded_AAMP16/Patient_Data/Training_Image_Data/1mm B30/full_1mm/'
    patients = sorted(os.listdir(patients_dir))
    data_len_dict = {}
    for patient in patients:
        patient_dir = os.path.join(patients_dir, patient, 'full_1mm')
        num_slices = len(os.listdir(patient_dir))
        data_len_dict[patient] = num_slices
    # print(f'Data from Patient {patient} contain {num_slices} slices.')

    test_patient = ['L506']
    train_val_patient = []
    num_test = 0
    num_train_val = 0
    for patient in patients:
        if patient not in test_patient:
            train_val_patient.append(patient)
            num_train_val += data_len_dict[patient]
        else:
            num_test += data_len_dict[patient]

    print(f'In total, there are {num_train_val + num_test} slices from {len(patients)} patients.')
    print(f'The divided dataset consists of {num_train_val} train/val image pairs, making up {num_train_val/(num_train_val + num_test) * 100}%.')
    print(f'The divided dataset consists of {num_test} test image pairs, making up {num_test/(num_train_val + num_test) * 100}%.')

    

    target_root = 'Your_saved_dataset_directory/aapm16'
    target_tag = {
        'train': 'train',
        'test':  'test'
    }
    patients = {
        'train': train_val_patient,
        'test': test_patient
    }
    for type in target_tag:
        target_dir = os.path.join(target_root, f'{target_tag[type]}_img')
        os.makedirs(target_dir, exist_ok=True)
        target_patient = patients[type]
        for patient in target_patient:
            source_dir = os.path.join(patients_dir, patient, 'full_1mm')
            ima_files = sorted([_ for _ in os.listdir(source_dir) if '.IMA' in _])
            for ima_file in ima_files:
                arr_file = ima_file.replace('.IMA', '.npy')
                ima_path = os.path.join(source_dir, ima_file)
                arr_path = os.path.join(target_dir, arr_file)

                if os.path.exists(ima_path) and not os.path.exists(arr_path):
                    arr = ima2array(ima_path)                
                    np.save(arr_path, arr)
                    print('.npy file saved: ', arr_file)