import numpy as np
import SimpleITK as sitk
# from typing import List, Union

# import numpy as np
# from pydicom.dataset import FileDataset

# from rt_utils.utils import ROIData
# from . import ds_helper, image_helper


def correct_struct_image(im_array, CT_image):
    '''
    Change of dimentions and flipping of structs from ISP dataset. 
    params:: im_array: numpy array of struct
    params:: CT_image: SimpleITK image of corresponding CT scan
    '''
    struct_new = np.swapaxes(im_array, 0,2)
    struct_new_2 = np.swapaxes(struct_new, 1,2)
    # struct_new_3 = np.flip(struct_new_2, 0)

    struct_sitk = sitk.GetImageFromArray(struct_new_2.astype(np.float32))

    struct_sitk.SetOrigin(CT_image.GetOrigin())
    struct_sitk.SetSpacing(CT_image.GetSpacing())
    struct_sitk.SetDirection(CT_image.GetDirection())

    return struct_sitk




# class RTStruct:
#     """
#     Wrapper class to facilitate appending and extracting ROI's within an RTStruct
#     """

#     def __init__(self, series_data, ds: FileDataset, ROIGenerationAlgorithm=0):
#         self.series_data = series_data
#         self.ds = ds
#         self.frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[
#             -1
#         ].FrameOfReferenceUID  # Use last strucitured set ROI

#     def set_series_description(self, description: str):
#         """
#         Set the series description for the RTStruct dataset
#         """

#         self.ds.SeriesDescription = description

#     def add_roi(
#         self,
#         mask: np.ndarray,
#         color: Union[str, List[int]] = None,
#         name: str = None,
#         description: str = "",
#         use_pin_hole: bool = False,
#         approximate_contours: bool = True,
#         roi_generation_algorithm: Union[str, int] = 0,
#     ):
#         """
#         Add a ROI to the rtstruct given a 3D binary mask for the ROI's at each slice
#         Optionally input a color or name for the ROI
#         If use_pin_hole is set to true, will cut a pinhole through ROI's with holes in them so that they are represented with one contour
#         If approximate_contours is set to False, no approximation will be done when generating contour data, leading to much larger amount of contour data
#         """

#         # TODO test if name already exists
#         self.validate_mask(mask)
#         roi_number = len(self.ds.StructureSetROISequence) + 1
#         roi_data = ROIData(
#             mask,
#             color,
#             roi_number,
#             name,
#             self.frame_of_reference_uid,
#             description,
#             use_pin_hole,
#             approximate_contours,
#             roi_generation_algorithm,
#         )

#         self.ds.ROIContourSequence.append(
#             ds_helper.create_roi_contour(roi_data, self.series_data)
#         )
#         self.ds.StructureSetROISequence.append(
#             ds_helper.create_structure_set_roi(roi_data)
#         )
#         self.ds.RTROIObservationsSequence.append(
#             ds_helper.create_rtroi_observation(roi_data)
#         )

#     def validate_mask(self, mask: np.ndarray) -> bool:
#         if mask.dtype != bool:
#             raise RTStruct.ROIException(
#                 f"Mask data type must be boolean. Got {mask.dtype}"
#             )

#         if mask.ndim != 3:
#             raise RTStruct.ROIException(f"Mask must be 3 dimensional. Got {mask.ndim}")

#         if len(self.series_data) != np.shape(mask)[2]:
#             raise RTStruct.ROIException(
#                 "Mask must have the save number of layers (In the 3rd dimension) as input series. "
#                 + f"Expected {len(self.series_data)}, got {np.shape(mask)[2]}"
#             )

#         if np.sum(mask) == 0:
#             print("[INFO]: ROI mask is empty")

#         return True

#     def get_roi_names(self) -> List[str]:
#         """
#         Returns a list of the names of all ROI within the RTStruct
#         """

#         if not self.ds.StructureSetROISequence:
#             return []

#         return [
#             structure_roi.ROIName for structure_roi in self.ds.StructureSetROISequence
#         ]
    
#     def get_struct(self, struct_num) -> np.ndarray:
        
#         """
#         Returns the 3D binary mask of the ROI

#         """

#         for i, structure_roi in enumerate(self.ds.StructureSetROISequence):
#             if i == struct_num:
#                 contour_sequence = ds_helper.get_contour_sequence_by_roi_number(
#                     self.ds, structure_roi.ROINumber
#                 )
#                 return image_helper.create_series_mask_from_contour_sequence(
#                     self.series_data, contour_sequence
#                 )

#         raise RTStruct.ROIException(f"ROI of number `{struct_num}` does not exist in RTStruct")