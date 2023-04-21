import numpy as np
import xml.etree.ElementTree as ET 

def calc_ROI(roi_dir):
    root = ET.parse(roi_dir).getroot()

    spacing = None
    origin = None
    direction = None
    for series in root.findall('Volume'):
        spacing = series.get('spacing').split()
        origin = series.get('origin').split()
        direction = series.get('ijkToRASDirections').split()
        print(spacing)
        print(origin)
        print(direction)

    coords = None

    for series in root.findall('AnnotationROI'):
        coords = series.get('ctrlPtsCoord').split("|")
        
        print(coords)

    np.c_[np.diag(np.array(spacing, dtype = float)), np.array(origin, dtype = float)]
    A = np.r_[np.c_[np.diag(np.array(spacing, dtype = float)), np.array(origin, dtype = float)], np.expand_dims(np.array([0, 0, 0, 1]), axis=0)]

    Direction = np.diag(np.array(direction, dtype = float).reshape(3,3))

    roi_origin = np.array(coords[0].split(), dtype=float)
    roi_size = np.array(coords[1].split(), dtype=float)

    roi_origin_1 = np.r_[roi_origin, np.array([1])]
    roi_voxel_center = np.linalg.inv(A)@roi_origin_1

    min_coord = (np.multiply(roi_voxel_center[:3], Direction) - np.diag(roi_size/A[:3, :3])).astype(int)
    max_coord = (np.multiply(roi_voxel_center[:3], Direction) + np.diag(roi_size/A[:3, :3])).astype(int)

    return min_coord, max_coord
