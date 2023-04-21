import os
import shutil

# def seperate_train_test(data_dir, test_set, destination_dir,  ):

#     for case in os.listdir(data_dir):
#         data_path = os.path.join(data_dir, case)
#         print(case[-11:-7])
        
#         if case[-11:-7] in test_set:
#             print(case[-11:-7])
#             shutil.move(data_path, destination_dir)
#                 #mergefolders(os.path.join(img_directory, case), os.path.join(directory, phase))

def seperate_train_test(data_dir, test_set, destination_dir,  extension = "_0000.nii.gz"):

    for case in os.listdir(data_dir):
        data_path = os.path.join(data_dir, case)
        print(case.replace(extension, '')[-4:] )
        if case.replace(extension, '')[-4:] in test_set:
            shutil.move(data_path, destination_dir)
                #mergefolders(os.path.join(img_directory, case), os.path.join(directory, phase))

def rename(data_dir, added_string = "_0000" ):

    for case in os.listdir(data_dir):
        data_path = os.path.join(data_dir, case)
        filename, file_extension = os.path.splitext(data_path)
        new_path = os.path.join(filename+added_string+file_extension )
        os.rename(data_path, new_path)

def rename_nifti(data_dir, added_string = "_0000" ):

    for case in os.listdir(data_dir):
        data_path = os.path.join(data_dir, case)
        new_path = data_path[:-7]+added_string+data_path[-7:]
        os.rename(data_path, new_path)

def remove(data_dir, added_string = "_0000" ):

    for case in os.listdir(data_dir):
        data_path = os.path.join(data_dir, case)
        new_path = data_path.replace(added_string, '')
        os.rename(data_path, new_path)

def delete(data_dir, string_to_contain="_Pancreas" ):

    for case in os.listdir(data_dir):
        data_path = os.path.join(data_dir, case)
        if string_to_contain in data_path:
            os.remove(data_path)