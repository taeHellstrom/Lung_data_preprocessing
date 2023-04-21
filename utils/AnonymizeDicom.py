import argparse
import SimpleITK as sitk
import sys
import os
import csv
import xml.etree.ElementTree as ET

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )
        print(f"Created output directory: {path}")

def recursive_search( input_dir, output_dir, tags ):
    """ finds subfolders """

    print("Starting recursive process..")
    for subdir, _, _ in os.walk(r"{}".format(input_dir)):
        output_path = os.path.join(output_dir, subdir.replace(input_dir, "")[1:])
        anonymize_contentXML(subdir, output_path)
        anonymize_dicom(subdir, output_path, tags)

def anonymize_contentXML(data_path, output_path):

    sectra = os.path.join(data_path, "SECTRA", "CONTENT.XML")
    tags_to_replace = ['datetime', 'utc_date', 'utc_time']      #add sectra content tags to remove here
    data = None
    if (os.path.exists(sectra)): 
        with open(sectra, encoding='latin-1') as f:
            tree = ET.parse(f)
            root = tree.getroot()

            for series in root.findall('patient/request/study/series'):
                for image in series.findall('image'):
                    for elem in list(image.iter()):
                        
                        if (elem.tag in tags_to_replace):
                            try:
                                elem.text = ""
                            except AttributeError:
                                print(f"Failed to replace {elem.tag}")
                                pass

            data = ET.tostring(root, encoding='UTF-8', xml_declaration=True)    

        makedirs(os.path.join(output_path, "SECTRA"))
        new_path = os.path.join(output_path, "SECTRA","CONTENT.xml")     

        myfile = open(new_path, "wb")
        myfile.write(data)
        print(f"Content.XML written to new file. {new_path}")

def anonymize_dicom(data_path, output_path, tags):
    

    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_path)

    if not series_IDs:
        print("The given directory \""+data_path+"\" does NOT contain a DICOM series.\n")
        return
    else:
        print("The given directory \""+data_path+"\" does contain a DICOM series.")

    print("Starting anonymizing process..")
    makedirs(output_path)

    for i, series_ID in enumerate(series_IDs):
        
        print(f"Starting with series: {i}, name: {series_ID}")
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_path, series_ID, useSeriesDetails=False) #useSeriesDetails ?
        series_reader = sitk.ImageSeriesReader()
        
        series_reader.SetFileNames(series_file_names)
        
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()

        # load Dicom series
        try:
            print("Reading")
            imgs = series_reader.Execute()
            
        except Exception as e:
            print (f"--> Fundamental error in image layer: {e} ")
            continue
        
        # step through each slice in the series and check header/metadata
        for i, image_name in enumerate(series_file_names):
            nreader = sitk.ReadImage(image_name, imageIO="GDCMImageIO")
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
        
            # Replace tags with empty strings ie. remove value of tag
            image_slice = nreader
            for tag in tags:
                # print("Removing tag: ", tag)
                # print(imgs.GetMetaData(tag))
                image_slice.SetMetaData(tag, "")
            

            filename = os.path.join(output_path, image_name.split("/")[-1]+'.dcm' )
            
            writer.SetFileName(filename)
            writer.Execute(image_slice)
            os.rename(filename, filename[:-4])
            
        print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='AnonymizeDicom')
    parser.add_argument('--input', type=str, help='directory containing dicom series')
    parser.add_argument('--output', type=str, help='directory to store anonymized dicoms')
    parser.add_argument('--tags', nargs='+', type=str, default=["0008|0012", "0008|0013"], help='[Date, Time] list of tags to remove from series')
    parser.add_argument('--recursive', dest='recursive', default=False, action='store_true', help='recursively search subdirectories for dicom series')
    
    opt = parser.parse_args()

    if opt.recursive==True:
        recursive_search(opt.input, opt.output, opt.tags)
    else:
        anonymize_dicom(opt.input, opt.output, opt.tags)



