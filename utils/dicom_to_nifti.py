#!/usr/bin/env python
import itk
import glob, os, sys 
import numpy as np
import re
import utils.calc_ROI as cr
import SimpleITK as sitk
import rt_utils

def Boundingboxed(data_path, label_path, output_dir, bb_dir, number, phase):

    #annotations = ['Gastroduodenalis', 'AMS', 'Aorta', 'Pancreas', 'Splenic vein', 'Truncus', 'Vena Cava', 'Vena porta', 'VMI', 'Tumour']
    annotations = ['Tumour', 'Pancreas']

    bb_dir = os.path.join(bb_dir, number, phase)
    bb_file = os.path.join(bb_dir, "*.mrml")
    
    for file in glob.glob(bb_file):
        bb_file = file

    if not os.path.exists(bb_file):
        return

    if not os.path.exists(label_path):
        return

    PixelType = itk.ctype('signed short')
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]
    # get bounding box for series
    min_coord, max_coord = cr.calc_ROI(bb_file)

    print(min_coord)

    print(max_coord)

    if(min_coord[2].item() <= 0 ):
        min_coord[2] = np.array([0])

    start = itk.Index[Dimension]()
    start[0] = min_coord[0].item() # startx
    start[1] = min_coord[1].item() # starty
    start[2] = min_coord[2].item() # start along Z

    end = itk.Index[Dimension]()
    end[0] = max_coord[0].item() # endx
    end[1] = max_coord[1].item() # endy
    end[2] = max_coord[2].item() # size along Z

    # roi_start = itk.Index[2]()
    # roi_start[0] = min_coord[0].item() # startx
    # roi_start[1] = min_coord[1].item() # starty

    # roi_end = itk.Index[2]()
    # roi_end[0] = max_coord[0].item() # endx
    # roi_end[1] = max_coord[1].item() # endy

    region = itk.ImageRegion[Dimension]()
    region.SetIndex(start)
    region.SetUpperIndex(end)

    # region2D = itk.ImageRegion[2]()
    # region2D.SetIndex(roi_start)
    # region2D.SetUpperIndex(roi_end)

    output = os.path.join(output_dir,  number, phase)
    output_label = os.path.join(output, 'label')
    output_image = os.path.join(output, 'image')

    if not os.path.exists(output_label):
        os.makedirs(output_label)
    if not os.path.exists(output_image):
        os.makedirs(output_image)

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(data_path)

    seriesUID = namesGenerator.GetSeriesUIDs()
    series_name = phase + '_' + number

    if len(seriesUID) < 1:
        print('No DICOMs in: ' + data_path)
        return

    seriesFound = False
    for uid in seriesUID:
        seriesIdentifier = uid

        print('Reading: ' + seriesIdentifier)

        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()
        #reader.Update()


        # reader_header = dicomIO.GetMetaDataDictionary()
        # dicom_header = itk.MetaDataObject()

        # start_dict = dictionary.Begin()
        # end_dict = dictionary.End()

        size = itk.size(reader.GetOutput())
        print("reader")
        print(size)
        print(reader.GetOutput())

        if size[2] <=2:
            break
        # Apply region of interest
        # ROI = itk.ExtractImageFilter[ImageType, ImageType].New()
        # ROI.SetExtractionRegion(region)
        # ROI.SetInput(reader.GetOutput())

        ROI = itk.RegionOfInterestImageFilter[ImageType, ImageType].New()
        ROI.SetInput(reader.GetOutput())
        ROI.SetRegionOfInterest(region)
        #ROI.SetMetaDataDictionary(reader.GetMetaDataDictionary())
        ROI.Update()

        print("ROI")
        size = itk.size(ROI)
        print(size)
        print(ROI.GetOutput())



        all_labels = []

        for ann in annotations:

            print('ann:', ann)

            labelfilelist = [file for file in os.listdir(label_path) if ann.lower() in file.lower()]

            if len(labelfilelist) > 0:

                labelfile = labelfilelist[0]

                print('labelfile: ', labelfile)
                labelpath = os.path.join(label_path, labelfile)

                
                MeshType = itk.Mesh[itk.SS,3]
                
                meshReader = itk.MeshFileReader[MeshType].New()
                meshReader.SetFileName(labelpath)
                meshReader.Update()

                #print('Meshreaderout: ', meshReader.GetOutput())

                #ImageType = itk.Image[itk.F, 3]

                filter = itk.TriangleMeshToBinaryImageFilter[MeshType, ImageType].New()
                filter.SetInput(meshReader.GetOutput())
                filter.SetInfoImage(reader.GetOutput())
                filter.Update()

                # size = itk.size(filter)
                # print("filter")
                # print(size)
                # print(filter.GetOutput())

                MeshROI = itk.ExtractImageFilter[ImageType, ImageType].New()
                MeshROI.SetExtractionRegion(region)
                MeshROI.SetInput(filter.GetOutput())
                MeshROI.Update()
        
                size = itk.size(MeshROI)
                print("MeshROI")
                print(size)
                #print(MeshROI.GetOutput())

                #print('filterout: ', filter.GetOutput())
                image = np.array(itk.array_from_image(MeshROI.GetOutput())).astype(np.bool)
                all_labels.append(image)

            else:
                
                imsize = ROI.GetOutput().GetLargestPossibleRegion().GetSize()
                print(f"Failed to find {ann} ")
                image = np.zeros((imsize[2], imsize[1], imsize[0])).astype(np.bool)

                all_labels.append(image)

        # process label
        print([ls.shape for ls in all_labels])

        labels = np.stack(all_labels)
        labelmap = labels.astype(np.uint8)
        roi_size = itk.size(ROI)
        
        new_labelmap = np.concatenate((np.zeros((1,roi_size[2],roi_size[1],roi_size[0])).astype(np.uint8), labelmap), axis=0)
        
        labelmap = new_labelmap.argmax(axis=0)
        labelmap = labelmap.astype(np.short)
        labelmap[-1, :,:] = 0.

        labelType = ImageType
        labelmap = np.ascontiguousarray(labelmap)
        itk_image = itk.GetImageFromArray(labelmap)
        itk_image.SetMetaDataDictionary(ROI.GetMetaDataDictionary())
        itk_image.Update()

        header = itk.ChangeInformationImageFilter[ImageType].New()
        header.SetInput(itk_image)
        #print("Header filter")
        header.SetOutputSpacing(ROI.GetOutput().GetSpacing())
        header.ChangeSpacingOn()
        header.SetOutputOrigin(ROI.GetOutput().GetOrigin())
        header.ChangeOriginOn()
        header.SetOutputDirection(ROI.GetOutput().GetDirection())
        header.ChangeDirectionOn()
        header.UpdateOutputInformation()
        header.Update()
        #header.ChangeNone()
        #print(header.GetOutput())

        writer = itk.ImageFileWriter[labelType].New()
        writer.SetInput(header.GetOutput())
        segmentation_name = os.path.join(output_label, series_name + '.nii.gz')
        print(segmentation_name)
        writer.SetFileName(segmentation_name)
        writer.Update()


        metadata = ROI.GetMetaDataDictionary()
        print('metadata: ', metadata)

        data_writer = itk.ImageFileWriter[ImageType].New()
        outFileName = os.path.join(output_image, series_name + '_0000.nii.gz')
        data_writer.SetFileName(outFileName)
        data_writer.UseCompressionOn()
        #data_writer.UseInputMetaDataDictionaryOn ()
        data_writer.SetInput(ROI.GetOutput())
        

        print('Writing: ' + outFileName)
        
        data_writer.Update()

        if seriesFound:
            break

def full3D_all_labels(data_path, label_path, output_dir, number, phase):

    annotations = ['Gastroduodenalis', 'AMS', 'Aorta', 'Pancreas', 'Splenic vein', 'Truncus', 'Vena Cava', 'Vena porta', 'VMI', 'Tumour']
    # annotations = ['Gastroduodenalis', 'Splenic vein', 'VMI', 'Vena Cava', 'Aorta',  'Truncus', "AMS", 'Vena porta', 'Pancreas',  'Tumour']
    # annotations = ['Gastroduodenalis', 'Splenic vein', 'VMI', 'Vena Cava', 'Aorta',  'Truncus', "AMS", 'Pancreas',  'Vena porta','Tumour'] # Vessel involvement
    # annotations = ['Pancreas', 'Tumour']
    # annotations = []

    if not os.path.exists(label_path):
        return

    PixelType = itk.ctype('signed short')
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]


    output = os.path.join(output_dir,  number, phase)
    output_label = os.path.join(output, 'labelsTr')
    output_image = os.path.join(output, 'imagesTr')

    if not os.path.exists(output_label):
        os.makedirs(output_label)
    if not os.path.exists(output_image):
        os.makedirs(output_image)

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(data_path)

    seriesUID = namesGenerator.GetSeriesUIDs()
    series_name = phase + '_' + number

    if len(seriesUID) < 1:
        print('No DICOMs in: ' + data_path)
        return

    seriesFound = False
    for uid in seriesUID:
        seriesIdentifier = uid

        print('Reading: ' + seriesIdentifier)

        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()
        #reader.Update()


        size = itk.size(reader.GetOutput())
        print("reader")
        print(size)
        print(reader.GetOutput())

        if size[2] <=2:
            break

        all_labels = []
        if len(annotations) != 0:
        
            for count, ann in enumerate( annotations):

                print('ann:', ann)

                labelfilelist = [file for file in os.listdir(label_path) if ann.lower() in file.lower()]

                if len(labelfilelist) > 0:

                    labelfile = labelfilelist[0]

                    print('labelfile: ', labelfile)
                    labelpath = os.path.join(label_path, labelfile)

                    
                    MeshType = itk.Mesh[itk.SS,3]
                    
                    meshReader = itk.MeshFileReader[MeshType].New()
                    meshReader.SetFileName(labelpath)
                    meshReader.Update()

                    #print('Meshreaderout: ', meshReader.GetOutput())

                    #ImageType = itk.Image[itk.F, 3]

                    filter = itk.TriangleMeshToBinaryImageFilter[MeshType, ImageType].New()
                    filter.SetInput(meshReader.GetOutput())
                    filter.SetInfoImage(reader.GetOutput())
                    filter.Update()


                    # print('filterout: ', filter.GetOutput())

                    image = np.array(itk.array_from_image(filter.GetOutput())).astype(np.bool) * (count+1)
                    all_labels.append(image)

                else:
                    
                    imsize = reader.GetOutput().GetLargestPossibleRegion().GetSize()
                    print(f"Failed to find {ann} ")
                    image = np.zeros((imsize[2], imsize[1], imsize[0])).astype(np.bool)

                    all_labels.append(image)
        

        # process label
        print([ls.shape for ls in all_labels])

        labels = np.stack(all_labels)
        all_labels = None
        labelmap = labels.astype(np.uint8)

        reader_size = itk.size(reader) ## changed this from filter
        
        new_labelmap = np.concatenate((np.zeros((1,reader_size[2],reader_size[1],reader_size[0])).astype(np.uint8), labelmap), axis=0)
        #new_labelmap = np.zeros((1,reader_size[2],reader_size[1],reader_size[0])).astype(np.uint8)
        labelmap = new_labelmap.argmax(axis=0)
        labelmap = labelmap.astype(np.short)
        

        labelType = ImageType
        labelmap = np.ascontiguousarray(labelmap)
        itk_image = itk.GetImageFromArray(labelmap)
        itk_image.SetMetaDataDictionary(reader.GetMetaDataDictionary())
        itk_image.Update()

        header = itk.ChangeInformationImageFilter[ImageType].New()
        header.SetInput(itk_image)
        #print("Header filter")
        header.SetOutputSpacing(reader.GetOutput().GetSpacing())
        header.ChangeSpacingOn()
        header.SetOutputOrigin(reader.GetOutput().GetOrigin())
        header.ChangeOriginOn()
        header.SetOutputDirection(reader.GetOutput().GetDirection())
        header.ChangeDirectionOn()
        header.UpdateOutputInformation()
        #header.Update()

        writer = itk.ImageFileWriter[labelType].New()
        writer.SetInput(header.GetOutput())
        segmentation_name = os.path.join(output_label, series_name + '.nii.gz')
        print(segmentation_name)
        writer.SetFileName(segmentation_name)
        writer.Update()


        # metadata = reader.GetMetaDataDictionary()
        # print('metadata: ', metadata)

        data_writer = itk.ImageFileWriter[ImageType].New()
        outFileName = os.path.join(output_image, series_name + '_0000.nii.gz')
        data_writer.SetFileName(outFileName)
        data_writer.UseCompressionOn()
        data_writer.UseInputMetaDataDictionaryOn ()
        data_writer.SetInput(reader.GetOutput())
        

        print('Writing: ' + outFileName)
        
        data_writer.Update()

        if seriesFound:
            break 


def seperate_labels(data_path, label_path, output_dir, number, phase):
    annotations = ['Gastroduodenalis', 'AMS', 'Aorta', 'Pancreas', 'Splenic vein', 'Truncus', 'Vena Cava', 'Vena porta', 'VMI', 'Tumour',  'Abb', 'Hep', 'CBD', 'PD']

    if not os.path.exists(label_path):
        return

    PixelType = itk.ctype('signed short')
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]


    output = os.path.join(output_dir,  number, phase)
    output_label = os.path.join(output, 'labels')
    output_image = os.path.join(output, 'images')

    if not os.path.exists(output_label):
        os.makedirs(output_label)
    if not os.path.exists(output_image):
        os.makedirs(output_image)

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(data_path)

    seriesUID = namesGenerator.GetSeriesUIDs()
    series_name = phase + '_' + number

    if len(seriesUID) < 1:
        print('No DICOMs in: ' + data_path)
        return

    seriesFound = False
    for uid in seriesUID:
        seriesIdentifier = uid

        print('Reading: ' + seriesIdentifier)

        fileNames = namesGenerator.GetFileNames(seriesIdentifier)
        
        if uid == '1.3.12.2.1107.5.1.4.50337.30000014062607203665600006934.42512512':
            print('This case has slicefiles in the wrong order. Reordering files...')
            # Since the default function just looks at images 0 and 1 to determine slice thickness
            # and the images are often not correctly alphabetically sorted this part will resort 
            # the file, much slower processing
            # Did not find an easy way to check if the files are incorrectly ordered in general 
            # so I only did this for the one case where I noticed the issue
            pimg_list = [(sitk.ReadImage(x).GetOrigin(), x) for x in fileNames]
            fileNames = [path for _, path in sorted(pimg_list, key = lambda x: x[0][2])] # sort by z

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()
        #reader.Update()
        size = itk.size(reader.GetOutput())
        print("reader")
        print(size)
        print(reader.GetOutput())

        original_labeled_files = os.listdir(label_path) 
        print(original_labeled_files)

        if size[2] <=2:
            print(f'DICOM series only has 1 or 2 slices. Did not process series {series_name}.')
            break
        

        if len(annotations) != 0:
        
            for count, ann in enumerate( annotations):

                print('ann:', ann)

                labelfilelist = [file for file in original_labeled_files if ann.lower() in file.lower()] # definily not the most efficient way to do this
                original_labeled_files = [x for x in original_labeled_files if x not in labelfilelist] # ensure an annotation is only used once

                if len(labelfilelist) == 1:

                    labelfile = labelfilelist[0]

                    print('labelfile: ', labelfile)
                    labelpath = os.path.join(label_path, labelfile)

                    
                    MeshType = itk.Mesh[itk.SS,3]
                    meshReader = itk.MeshFileReader[MeshType].New()
                    meshReader.SetFileName(labelpath)
                    meshReader.Update()


                    filter = itk.TriangleMeshToBinaryImageFilter[MeshType, ImageType].New()
                    filter.SetInput(meshReader.GetOutput())
                    filter.SetInfoImage(reader.GetOutput())
                    filter.Update()

                    image = np.array(itk.array_from_image(filter.GetOutput())).astype(np.short) 
                    labelmap = np.ascontiguousarray(image)

                    itk_image = itk.GetImageFromArray(labelmap)
                    itk_image.SetMetaDataDictionary(reader.GetMetaDataDictionary())
                    itk_image.Update()

                    header = itk.ChangeInformationImageFilter[ImageType].New()
                    header.SetInput(itk_image)
                    header.SetOutputSpacing(reader.GetOutput().GetSpacing())
                    header.ChangeSpacingOn()
                    header.SetOutputOrigin(reader.GetOutput().GetOrigin())
                    header.ChangeOriginOn()
                    header.SetOutputDirection(reader.GetOutput().GetDirection())
                    header.ChangeDirectionOn()
                    header.UpdateOutputInformation()
                    #header.Update()

                    writer = itk.ImageFileWriter[ImageType].New()
                    writer.SetInput(header.GetOutput())
                    segmentation_name = os.path.join(output_label, series_name + '_' +ann+ '.nii.gz')
                    print(segmentation_name)
                    writer.SetFileName(segmentation_name)
                    writer.Update()

                elif len(labelfilelist) > 1:
                    print(f"Multiple labels for {ann} ")

                else:
                    print(f"Failed to find {ann} ")

        metadata = reader.GetMetaDataDictionary()
        print('metadata: ', metadata)

        data_writer = itk.ImageFileWriter[ImageType].New()
        outFileName = os.path.join(output_image, series_name + '.nii.gz')
        data_writer.SetFileName(outFileName)
        data_writer.UseCompressionOn()
        data_writer.UseInputMetaDataDictionaryOn ()
        data_writer.SetInput(reader.GetOutput())

        print('Writing: ' + outFileName)
        data_writer.Update()

        if seriesFound:
            break 


def scan_only(data_path, output_dir, number, phase):


    PixelType = itk.ctype('signed short')
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]

    output = os.path.join(output_dir,  number, phase)
    output_image = os.path.join(output, 'images')


    if not os.path.exists(output_image):
        os.makedirs(output_image)

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(data_path)

    seriesUID = namesGenerator.GetSeriesUIDs()
    series_name = phase + '_' + number

    if len(seriesUID) < 1:
        print('No DICOMs in: ' + data_path)
        return

    seriesFound = False
    for uid in seriesUID:
        seriesIdentifier = uid

        print('Reading: ' + seriesIdentifier)

        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()
        #reader.Update()

        size = itk.size(reader.GetOutput())
        print("reader")
        print(size)
        print(reader.GetOutput())

        if size[2] <=2:
            print(f'DICOM series only has 1 or 2 slices. Did not process series {series_name}.')
            break

        
        metadata = reader.GetMetaDataDictionary()
        print('metadata: ', metadata)

        data_writer = itk.ImageFileWriter[ImageType].New()
        outFileName = os.path.join(output_image, series_name + '.nii.gz')
        data_writer.SetFileName(outFileName)
        data_writer.UseCompressionOn()
        data_writer.UseInputMetaDataDictionaryOn ()
        data_writer.SetInput(reader.GetOutput())

        print('Writing: ' + outFileName)
        data_writer.Update()

        if seriesFound:
            break 


def single_label(data_path, label_path, output_dir, number, phase, ann_name, ann):

    annotations = ['Gastroduodenalis', 'AMS', 'Aorta', 'Pancreas', 'Splenic vein', 'Truncus', 'Vena Cava', 'Vena porta', 'VMI', 'Tumour',  'Abb', 'Hep', 'CBD', 'PD']
    
    if not os.path.exists(label_path):
        return

    PixelType = itk.ctype('signed short')
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]


    output = os.path.join(output_dir,  number, phase)
    output_label = os.path.join(output, 'labels')
    output_image = os.path.join(output, 'images')

    if not os.path.exists(output_label):
        os.makedirs(output_label)
    if not os.path.exists(output_image):
        os.makedirs(output_image)

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(data_path)

    seriesUID = namesGenerator.GetSeriesUIDs()
    series_name = phase + '_' + number

    if len(seriesUID) < 1:
        print('No DICOMs in: ' + data_path)
        return

    seriesFound = False
    for uid in seriesUID:
        seriesIdentifier = uid
        print('Reading: ' + seriesIdentifier)
        
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()
        #reader.Update()

        size = itk.size(reader.GetOutput())
        print("reader")
        print(size)
        print(reader.GetOutput())

        original_labeled_files = os.listdir(label_path) 
        print(original_labeled_files)
        
        if size[2] <=2:
            print(f'DICOM series only has 1 or 2 slices. Did not process series {series_name}.')
            break
        
        

        
        print('ann:', ann)

        labelfilelist = [file for file in original_labeled_files if ann.lower() in file.lower()] # definily not the most efficient way to do this
        original_labeled_files = [x for x in original_labeled_files if x not in labelfilelist] # ensure an annotation is only used once
        print(f'Simliar files in folder: \n{labelfilelist}')
        
        labelpath = os.path.join(label_path, ann_name)
        if os.path.exists(labelpath):

            labelfile = labelfilelist[0]

            print('labelfile: ', labelpath)

            
            MeshType = itk.Mesh[itk.SS,3]
            meshReader = itk.MeshFileReader[MeshType].New()
            meshReader.SetFileName(labelpath)
            meshReader.Update()


            filter = itk.TriangleMeshToBinaryImageFilter[MeshType, ImageType].New()
            filter.SetInput(meshReader.GetOutput())
            filter.SetInfoImage(reader.GetOutput())
            filter.Update()

            image = np.array(itk.array_from_image(filter.GetOutput())).astype(np.short) 
            labelmap = np.ascontiguousarray(image)

            itk_image = itk.GetImageFromArray(labelmap)
            itk_image.SetMetaDataDictionary(reader.GetMetaDataDictionary())
            itk_image.Update()

            header = itk.ChangeInformationImageFilter[ImageType].New()
            header.SetInput(itk_image)
            header.SetOutputSpacing(reader.GetOutput().GetSpacing())
            header.ChangeSpacingOn()
            header.SetOutputOrigin(reader.GetOutput().GetOrigin())
            header.ChangeOriginOn()
            header.SetOutputDirection(reader.GetOutput().GetDirection())
            header.ChangeDirectionOn()
            header.UpdateOutputInformation()
            #header.Update()

            writer = itk.ImageFileWriter[ImageType].New()
            writer.SetInput(header.GetOutput())
            segmentation_name = os.path.join(output_label, series_name + '_' +ann+ '.nii.gz')
            print(segmentation_name)
            writer.SetFileName(segmentation_name)
            writer.Update()

        else:
            sys.exit(f"File does not exist: {labelpath} ")

        metadata = reader.GetMetaDataDictionary()
        print('metadata: ', metadata)

        data_writer = itk.ImageFileWriter[ImageType].New()
        outFileName = os.path.join(output_image, series_name + '.nii.gz')
        data_writer.SetFileName(outFileName)
        data_writer.UseCompressionOn()
        data_writer.UseInputMetaDataDictionaryOn ()
        data_writer.SetInput(reader.GetOutput())

        print('Writing: ' + outFileName)
        data_writer.Update()

        if seriesFound:
            break 


def combine_to_single_label(data_path, label_path, output_dir, number, phase, ann_name_1, ann_name_2, ann):
    '''
    Combine two labels to one nifty file. 
    '''
    annotations = ['Gastroduodenalis', 'AMS', 'Aorta', 'Pancreas', 'Splenic vein', 'Truncus', 'Vena Cava', 'Vena porta', 'VMI', 'Tumour',  'Abb', 'Hep', 'CBD', 'PD']
    
    if not os.path.exists(label_path):
        return

    PixelType = itk.ctype('signed short')
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]


    output = os.path.join(output_dir,  number, phase)
    output_label = os.path.join(output, 'labels')
    output_image = os.path.join(output, 'images')

    if not os.path.exists(output_label):
        os.makedirs(output_label)
    if not os.path.exists(output_image):
        os.makedirs(output_image)

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(data_path)

    seriesUID = namesGenerator.GetSeriesUIDs()
    series_name = phase + '_' + number

    if len(seriesUID) < 1:
        print('No DICOMs in: ' + data_path)
        return

    seriesFound = False
    for uid in seriesUID:
        seriesIdentifier = uid
        print('Reading: ' + seriesIdentifier)
        
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()
        #reader.Update()

        size = itk.size(reader.GetOutput())
        print("reader")
        print(size)
        print(reader.GetOutput())

        original_labeled_files = os.listdir(label_path) 
        print(original_labeled_files)
        
        if size[2] <=2:
            print(f'DICOM series only has 1 or 2 slices. Did not process series {series_name}.')
            break
        
        

        
        print('ann:', ann)

        labelfilelist = [file for file in original_labeled_files if ann.lower() in file.lower()] # definily not the most efficient way to do this
        original_labeled_files = [x for x in original_labeled_files if x not in labelfilelist] # ensure an annotation is only used once
        print(f'Simliar files in folder: \n{labelfilelist}')
        
        labelpath_1 = os.path.join(label_path, ann_name_1)
        labelpath_2 = os.path.join(label_path, ann_name_2)
        if os.path.exists(labelpath_1) and os.path.exists(labelpath_2):

            # labelfile = labelfilelist[0]

            print('labelfile 1: ', labelpath_1)
            print('labelfile 2: ', labelpath_2)
            print('\n Combining images...')

            
            MeshType = itk.Mesh[itk.SS,3]
            meshReader1 = itk.MeshFileReader[MeshType].New()
            meshReader1.SetFileName(labelpath_1)
            meshReader1.Update()


            filter1 = itk.TriangleMeshToBinaryImageFilter[MeshType, ImageType].New()
            filter1.SetInput(meshReader1.GetOutput())
            filter1.SetInfoImage(reader.GetOutput())
            filter1.Update()

            image1 = np.array(itk.array_from_image(filter1.GetOutput())).astype(np.short) 



            meshReader2 = itk.MeshFileReader[MeshType].New()
            meshReader2.SetFileName(labelpath_2)
            meshReader2.Update()


            filter2 = itk.TriangleMeshToBinaryImageFilter[MeshType, ImageType].New()
            filter2.SetInput(meshReader2.GetOutput())
            filter2.SetInfoImage(reader.GetOutput())
            filter2.Update()

            image2 = np.array(itk.array_from_image(filter2.GetOutput())).astype(np.short) 

            combined_image = image1 + image2

            if np.max(combined_image) > 1:                  
                # If there are overlapping structures these will have a value higher than 1. 
                # Change these pixel values to 1. 
                print(f'Overlapping structures! Max value of combined image larger than 1 (max: {np.max(combined_image)}). Adapting label values...')
                val_index = np.where(combined_image > 1)
               
                for i in range(len(val_index[0])):
                    combined_image[val_index[0][i], val_index[1][i], val_index[2][i]] = 1
                print('New max val:', np.max(combined_image))

            labelmap = np.ascontiguousarray(combined_image)




            itk_image = itk.GetImageFromArray(labelmap)
            itk_image.SetMetaDataDictionary(reader.GetMetaDataDictionary())
            itk_image.Update()

            header = itk.ChangeInformationImageFilter[ImageType].New()
            header.SetInput(itk_image)
            header.SetOutputSpacing(reader.GetOutput().GetSpacing())
            header.ChangeSpacingOn()
            header.SetOutputOrigin(reader.GetOutput().GetOrigin())
            header.ChangeOriginOn()
            header.SetOutputDirection(reader.GetOutput().GetDirection())
            header.ChangeDirectionOn()
            header.UpdateOutputInformation()
            #header.Update()

            writer = itk.ImageFileWriter[ImageType].New()
            writer.SetInput(header.GetOutput())
            segmentation_name = os.path.join(output_label, series_name + '_' +ann+ '.nii.gz')
            print(segmentation_name)
            writer.SetFileName(segmentation_name)
            writer.Update()

        else:
            sys.exit(f"File does not exist: {labelpath_1} or \n {labelpath_2}")

        metadata = reader.GetMetaDataDictionary()
        print('metadata: ', metadata)

        data_writer = itk.ImageFileWriter[ImageType].New()
        outFileName = os.path.join(output_image, series_name + '.nii.gz')
        data_writer.SetFileName(outFileName)
        data_writer.UseCompressionOn()
        data_writer.UseInputMetaDataDictionaryOn ()
        data_writer.SetInput(reader.GetOutput())

        print('Writing: ' + outFileName)
        data_writer.Update()

        if seriesFound:
            break 

def dicom_read_LIDC_CT_and_label(CT_path, label_path, label_name):



    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(CT_path)
    reader.SetFileNames(dicom_names)

    CT_image = reader.Execute()

    label_file_path = os.path.join(label_path, label_name)#, '1-1.dcm')

    label_image = sitk.ReadImage(label_file_path)

    assert label_image.GetSpacing() == CT_image.GetSpacing(), f'Spacing of images needs to be the same. \nPrediction spacing: {label_image.GetSpacing()}\nOutput spacing: {CT_image.GetSpacing()}'

    Resize_filter = sitk.ResampleImageFilter()
    Resize_filter.SetOutputSpacing(label_image.GetSpacing())
    Resize_filter.SetOutputOrigin(CT_image.GetOrigin())
    Resize_filter.SetSize(CT_image.GetSize())
    Resize_filter.SetOutputDirection(label_image.GetDirection())
    Resize_filter.SetDefaultPixelValue(0)
    full_size_label = Resize_filter.Execute(label_image)
    
    return CT_image, full_size_label


def dicom_read_LIDC_CT(CT_path, output_dir = None, case_num = None, save_ims=False):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(CT_path)
    reader.SetFileNames(dicom_names)

    CT_image = reader.Execute()
    
    if save_ims:
        filepath_im = os.path.join(output_dir, case_num + '_image.nii.gz')
        print(f'Saving resized file to {output_dir}')
        writer = sitk.ImageFileWriter()
        writer.SetFileName(filepath_im)
        writer.Execute(CT_image)
    else:
        return CT_image


def dicom_read_LIDC_label(CT_path, label_path, label_name, output_dir = None, case_num=None, save_ims=False):
    '''
    CT_path:        Absolute path to CT scan in nifty format
    label_path:     Absolute path to label image in dicom format
    label_name:     Name based on nodule number and annotation number 'Nodule_{num1}_ann_{num2}'
    output_dir:     Absolute dir to location of file save location. Default: None
    case_num:       Case number of file between 0001 and 1012. Default: None
    save_ims:       Boolean operator deciding whether files will be saved to output_dir. Default: False
    '''


    CT_image = sitk.ReadImage(CT_path)

    label_file_path = os.path.join(label_path, '1-1.dcm')

    label_image = sitk.ReadImage(label_file_path)

    # assert label_image.GetSpacing() == CT_image.GetSpacing(), f'Spacing of images needs to be the same. \nPrediction spacing: {label_image.GetSpacing()}\nOutput spacing: {CT_image.GetSpacing()}'
    if not label_image.GetSpacing() == CT_image.GetSpacing():
        # print(f'Spacing of images needs to be the same. \nPrediction spacing: {label_image.GetSpacing()}\nOutput spacing: {CT_image.GetSpacing()}')
        if not round(label_image.GetSpacing()[0], 5) == round(CT_image.GetSpacing()[0], 5) or not round(label_image.GetSpacing()[1], 5) == round(CT_image.GetSpacing()[1], 5) or not round(label_image.GetSpacing()[2], 5) == round(CT_image.GetSpacing()[2], 5):
            print(round(label_image.GetSpacing()[2], 5), round(CT_image.GetSpacing()[2], 5))
            assert label_image.GetSpacing() == CT_image.GetSpacing(), f'Spacing of images needs to be the same. \nPrediction spacing: {label_image.GetSpacing()}\nOutput spacing: {CT_image.GetSpacing()}'

    Resize_filter = sitk.ResampleImageFilter()
    Resize_filter.SetOutputSpacing(CT_image.GetSpacing())
    Resize_filter.SetOutputOrigin(CT_image.GetOrigin())
    Resize_filter.SetSize(CT_image.GetSize())
    Resize_filter.SetOutputDirection(CT_image.GetDirection())
    Resize_filter.SetDefaultPixelValue(0)
    full_size_label = Resize_filter.Execute(label_image)
    
    
    if save_ims:

        filepath_label = os.path.join(output_dir, case_num + '_' + label_name + '.nii.gz')
        print(f'Saving resized file to {output_dir}')
        writer = sitk.ImageFileWriter()
        writer.SetFileName(filepath_label)
        writer.Execute(full_size_label)
    else:
        return full_size_label

def dicom_to_nifty_LIDC(CT_path, label_path, label_name, output_dir = None, save_ims = False):



    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(CT_path)
    reader.SetFileNames(dicom_names)

    CT_image = reader.Execute()

    label_file_path = os.path.join(label_path, label_name, '1-1.dcm')

    label_image = sitk.ReadImage(label_file_path)

    assert label_image.GetSpacing() == CT_image.GetSpacing(), f'Spacing of images needs to be the same. \nPrediction spacing: {label_image.GetSpacing()}\nOutput spacing: {CT_image.GetSpacing()}'

    Resize_filter = sitk.ResampleImageFilter()
    Resize_filter.SetOutputSpacing(label_image.GetSpacing())
    Resize_filter.SetOutputOrigin(CT_image.GetOrigin())
    Resize_filter.SetSize(CT_image.GetSize())
    Resize_filter.SetOutputDirection(label_image.GetDirection())
    Resize_filter.SetDefaultPixelValue(0)
    full_size_label = Resize_filter.Execute(label_image)

    if save_ims:

        filepath_im = os.path.join(output_dir, 'CT_image.nii.gz')
        print(f'Saving resized file to {output_dir}')
        writer = sitk.ImageFileWriter()
        writer.SetFileName(filepath_im)
        writer.Execute(CT_image)


        filepath_label = os.path.join(output_dir, 'nodule_label.nii.gz')
        print(f'Saving resized file to {output_dir}')
        writer = sitk.ImageFileWriter()
        writer.SetFileName(filepath_label)
        writer.Execute(full_size_label)

def read_RT_struct(mask_dir, ct_dir, CT_sitk, output_dir=None, save_num=None):
    if (not output_dir == None) and  not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rtstruct = rt_utils.RTStructBuilder.create_from(
    dicom_series_path=ct_dir, 
    rt_struct_path=mask_dir
    )
    struct_names = rtstruct.get_roi_names()
    im_orig = CT_sitk.GetOrigin()
    im_spacing = CT_sitk.GetSpacing()
    im_direction = CT_sitk.GetDirection()
    im_size = CT_sitk.GetSize()
    for struct in struct_names:
        struct_array = rtstruct.get_roi_mask_by_name(struct).astype(int)    # Include .astype(int) to convert from bool to int
        
        
        struct_sitk = sitk.GetImageFromArray(struct_array)
        
        struct_sitk.SetOrigin(im_orig)
        struct_sitk.SetSpacing(im_spacing)
        struct_sitk.SetDirection(im_direction)
        if not output_dir == None:
            if 'Nodule' in struct:
                break
                struct = 'Nodule_seg_1'
            elif 'Lung' in struct:
                struct = struct[:5] + '_lung_seg'
            filename = save_num + "_" + struct + ".nii.gz"
            filepath = os.path.join(output_dir, filename)
            j=2
            while os.path.exists(filepath):
                struct = 'Nodule_seg_' + str(j)
                filename = save_num + "_" + struct + ".nii.gz"
                filepath = os.path.join(output_dir, filename)
                j+=1

            print(f'Saving struct {struct} to {filepath}')
            writer = sitk.ImageFileWriter()
            writer.SetFileName(filepath)
            writer.Execute(struct_sitk)



if __name__ == "__main__":
    pass