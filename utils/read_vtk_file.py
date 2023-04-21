import os
import sys

import numpy as np
import itk

def find_leafes( root ):
    """ finds folders with no subfolders """
    for root, dirs, files in os.walk(root):
        if not dirs: # can't go deeper
            return root

root_dir = r'G:\Datasets\pancreas_cze\Bounding boxes\BB Tumor(+), Vaat(-)\1053\par'
image_path = os.path.join(root_dir, '706 Unnamed Series.nrrd')

annotations = ['Gastroduodenalis', 'AMS', 'Aorta', 'Pancreas', 'Splenic vein', 'Truncus', 'Vena Cava', 'Vena porta', 'VMI', 'Tumour']
annDir = r'G:\Datasets\pancreas_cze\Detail annotaties\Detail Vaat(-)\1053\pvp\Patient 1053_1053\No study description\Patient 1053 706 S__(11-03-2020_11-10-48-9462)'

#dirName = find_leafes(root)
#print('dirName', dirName)
PixelType = itk.ctype('signed short')
Dimension = 3
ImageType = itk.Image[itk.F, 3]
#ImageType = itk.Image[PixelType, Dimension]

#namesGenerator = itk.GDCMSeriesFileNames.New()
#namesGenerator.SetUseSeriesDetails(True)
#namesGenerator.AddSeriesRestriction("0008|0021")
#namesGenerator.SetGlobalWarningDisplay(False)
#namesGenerator.SetDirectory(dirName)

#seriesUID = namesGenerator.GetSeriesUIDs()

#if not len(seriesUID) == 1:
#    print('too few or too many DICOMs in: ' + dirName)
#    sys.exit(1)
#for uid in seriesUID:
#    seriesIdentifier = uid
#    if len(sys.argv) > 3:
#        seriesIdentifier = sys.argv[3]
#        seriesFound = True
#    print('Reading: ' + seriesIdentifier)
#    fileNames = namesGenerator.GetFileNames(seriesIdentifier)
#

#
#    reader = itk.ImageSeriesReader[ImageType].New()
#    dicomIO = itk.GDCMImageIO.New()
#    reader.SetImageIO(dicomIO)
#    reader.SetFileNames(fileNames)
#    reader.ForceOrthogonalDirectionOff()

reader = itk.ImageFileReader[itk.Image[itk.F, 3]].New(FileName=image_path)
reader.Update()

"""
    writer = itk.ImageFileWriter[ImageType].New()
    outFileName = os.path.join(dirName, seriesIdentifier + '.nii.gz')
    if len(sys.argv) > 2:
        outFileName = sys.argv[2]
    writer.SetFileName(outFileName)
    writer.UseCompressionOn()
    writer.SetInput(reader.GetOutput())
    print('Writing: ' + outFileName)
    writer.Update()
"""

all_labels = []
for ann in annotations:
    print('ann:', ann)
    labelfilelist = [file for file in os.listdir(annDir) if ann.lower() in file.lower()]
    if len(labelfilelist) > 0:
        labelfile = labelfilelist[0]
        print('labelfile: ', labelfile)
        labelpath = os.path.join(annDir, labelfile)

        MeshType = itk.Mesh[itk.F, 3]
        meshReader = itk.MeshFileReader[MeshType].New()
        meshReader.SetFileName(labelpath)
        meshReader.Update()

        #print('Meshreaderout: ', meshReader.GetOutput())

        ImageType = itk.Image[itk.F, 3]

        filter = itk.TriangleMeshToBinaryImageFilter[MeshType, ImageType].New()
        filter.SetInput(meshReader.GetOutput())
        filter.SetInfoImage(reader.GetOutput())
        filter.Update()

        #print('filterout: ', filter.GetOutput())
        image = np.array(itk.array_from_image(filter.GetOutput())).astype(np.bool)
        all_labels.append(image)

    else:
        imsize = reader.GetOutput().GetLargestPossibleRegion().GetSize()

        image = np.zeros((imsize[2], imsize[1], imsize[0])).astype(np.bool)

        all_labels.append(image)


#metadata = reader.GetMetaDataDictionary()
#del reader
#del filter


print([ls.shape for ls in all_labels])
labels = np.stack(all_labels)

labelmap = labels.astype(np.uint8)
labelmap = labelmap.argmax(axis=0)
labelmap = labelmap.astype(np.float32)
#labelmap[:,:,] = 0.
labelmap[-1, :,:] = 0.


print(labelmap.shape)

labelType = itk.Image[itk.F, 3]
labelmap = np.ascontiguousarray(labelmap)
itk_image = itk.GetImageFromArray(labelmap)
itk_image.SetMetaDataDictionary(reader.GetMetaDataDictionary())
itk_image.Update()

metadata = reader.GetMetaDataDictionary()
print('metadata: ', metadata)


writer = itk.ImageFileWriter[labelType].New()
writer.SetInput(itk_image)
writer.SetFileName(os.path.join(root_dir, 'segm.nrrd'))
writer.Update()





