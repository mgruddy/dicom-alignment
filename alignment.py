import os
import pydicom as dicom
import warnings
import pandas as pd
import cv2
from contour import *
from scipy.interpolate import RegularGridInterpolator

# here object (dcm) refers to a pydicom.dataset.FileDataset object

def get_dicom_files(path, modality=None):
    """
    Returns all DICOM files in a directory; user can specify a specific modality
    
    Arugments:
        path (str): path to directory with desired DICOM files
    Outputs:
        dicom_files (list of dcms): list of DICOM dataset objects contained in given directory
        OR
        dicom_dataset (dcm): A single DICOM dataset object if there is only DICOM file 
            (of a given modality, if specified) in the directory
    """
    # handle `/` missing
    if path[-1] != '/':
        path += '/'
    # get .dcm files
    fpaths = [path + f for f in os.listdir(path) if '.dcm' in f]
    if len(fpaths) == 0:
        print("There are no dicom files found in directory")
        return None
    # return all dicom files if no modality is given
    if not modality:
        return [dicom.read_file(fpath) for fpath in fpaths]
    # get all dicom files of a certain modality
    dicom_files = []
    for fpath in fpaths:
        f = dicom.read_file(fpath)
        if f.Modality == modality:
            dicom_files.append(f)

    if (modality in ['RTDOSE', 'RTPLAN', 'RTSTRUCT']) and (len(dicom_files) > 1):
        warnings.warn(f"There are multiple {modality} files, returning the last one!")
        return dicom_files
    elif len(dicom_files) == 0:
        print(f"No {modality} file(s) found in directory.")
        return None
    else:
        dicom_dataset = dicom_files[0]
        return dicom_dataset
    
def Rx_dose_from_plan(plan_file, target_names=['TARGET']):
    """
    Given a plan file (a DICOM dataset object with modality RTPLAN) find
        the prescription dose in Grays. Prints the Prescription Description if
        no DoseReferenceSequence attribute is found.
        
    Arugments:
        plan_file (dcm): a DICOM dataset object with modality RTPLAN
        target_names (list of strs): an optional argument to specify the name of the
            Dose Reference Type for the target volume
    Outputs:
        Rx (float): The prescription dose in Grays of the target volume.
    """
    try:
        dose_ref_seq = plan_file.DoseReferenceSequence
    except:
        warnings.warn("No Dose Reference Sequence, returning Prescription Description...")
        try:
            Rx_descrip = plan_file.PrescriptionDescription
        except:
            print("No Precription Description either!")
            return None
        return Rx_descrip
    
    for organ in dose_ref_seq:
        if organ["300a", '0020'].value in target_names:
            Rx = float(organ["300a", '0026'].value)
            return Rx
    print("No target volume found...")
    print("Check that Dose Reference Type is labeled 'TARGET', and edit target_names argument.")

def get_CT_ordered_slices(path):
    """
    Returns a list of DICOM datasets corresponding to slices of a CT scan, ordered by
        slice number
        
    Arugments:
        path (str): path to directory with CT slice DICOM files
    Outputs:
        sorted_slices (list of dcms): list of DICOM dataset objects corresponding
            to slices of a CT scan, ordered by slice number
    """
    slices = get_dicom_files(path, modality="CT")
    assert slices == list, "Only one CT file found!"
    slices = [(s, s.ImagePositionPatient[-1]) for s in slices]
    slices = sorted(slices, key=operator.itemgetter(1))
    sorted_slices = [slc[0] for slc in slices]
    return sorted_slices

def get_CT_array(path):
    """
    Constructs a 3D CT scan from DICOM files corresponding to slices
        of the CT scan, in the form of a numpy array. The output array is of
        shape (Columns, Rows, Depth) in accordance with standard DICOM format

    Arguments:
        path (str): path to directory with CT slice DICOM files
    Outputs:
        CT_array (numpy array): A numpy array corresponding to a 3D CT scan
    """
    slices = [slc.pixel_array for slc in get_ordered_CT_slices(path)]
    CT_array = np.array(CT_slices)
    CT_array = np.transpose(CT_array,(1,2,0))
    return CT_array

def get_dose_array(path, Rx_normalization=None):
    """
    Constructs a 3D dose delivery array from a DICOM file, in the form of a numpy array.
        The output array is of shape (Columns, Rows, Depth) in accordance with standard DICOM format.
        If given a prescription dose value in Grays, will normalize the dose array with
        respect to the prescription dose, i.e. 1.0 refers to value of the prescription dose in Grays.

    Arguments:
        path (str): path to directory with CT slice DICOM files
        Rx_normaliztion (float): an optional argument to normalize the dose array with respect to
            the prescription dose in Grays
    Outputs:
        dose_array (numpy array): a numpy array corresponding to 3D dose delivery
    """
    dose_data = get_dicom_files(path, modality="RTDOSE")
    dose_array = dose_data.pixel_array
    # cols, rows, depth!
    dose_array = np.transpose(dose_array, (1,2,0))
    if Rx_normalization:
        Gy_scale_factor = dose_data.DoseGridScaling
        dose_array * Gy_scale_factor
        dose_array / Rx_normalization
    return dose_array

def spacing_from_dicom(dicom_dataset):
    """
    Get the 3D spacing information from a DICOM file
    
    Arguments:
        dcm (dcm): a DICOM dataset object
    Outputs:
        spacing(list of floats): Spacing of the voxels of the corresponding
            3D array (in millimeters) given in columns, rows, depth
    """
    spacings = [float(dicom_dataset.PixelSpacing[0]),
                float(dicom_dataset.PixelSpacing[1]),
                float(dicom_dataset.SliceThickness)]
    return spacings

def origin_from_dicom(dicom_dataset):
    """
    Get the physical coordinates for the first (upper-left-shallowest) voxel from a DICOM file
    
    Arguments:
        dcm (dcm): a DICOM dataset object
    Outputs:
        origin (list of floats): The 3D coordinates of the first voxel of the corresponding
            DICOM files (in millimeters) given in columns, rows, depth
    """
    origin = [float(i) for i in dicom_dataset.ImagePositionPatient]
    return origin

def orientation_from_dicom(dicom_dataset):
    """
    Get the Image Orientation from a DICOM file
    
    Arguments:
        dcm (dcm): a DICOM dataset object
    Outputs:
        orientation (list of floats): The Image Orientation for the corresponding 3D array
    """
    orientation = [float(i) for i in dicom_dataset.ImageOrientationPatient]
    return orientation

def get_CT_voxel_data(path):
    """
    Get physical information about the voxels of a 3D CT scan
    
    Arguments:
        path (str): path to the directory with the CT scan slices
    Outputs:
        CT_shape (triple of ints): The shape of the 3D array (columns, rows, depth)
        CT_origin (list of floats): Spacing of the voxels of the corresponding
            CT scan 3D array (in millimeters) given in columns, rows, depth
        CT_spacing (list of floats):  The 3D coordinates of the first voxel of the corresponding
            CT scan 3D array (in millimeters) given in columns, rows, depth
    """
    slices = get_ordered_CT_slices(path)
    
    assert all([(orientation_from_dicom(slc) ==
                 [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) for slc in slices]), "Not yet implemented for this ImageOrientation."
    
    assert all([(spacing_from_dicom(slc) ==
                     spacing_from_dicom(slices[0])) for slc in slices]), "Slices have different spacings!"
    CT_spacing = spacing_from_dicom(slices[0])
    
    # check for slice thickness errors
    assert all([((origin_from_dicom(slices[i])[-1] - origin_from_dicom(slices[i-1])[-1]) ==
                 CT_spacing[-1]) for i in range(1,len(slices))]), "Slice thickness is incorrect."
    
    CT_origin = origin_from_dicom(slices[0])
    
    xy_shape = slices[0].pixel_array.shape
    CT_shape = (xy_shape[0], xy_shape[1], len(slices))
  
    return CT_shape, CT_origin, CT_spacing

def get_dose_voxel_data(path):
    """
    Get physical information about the voxels of a 3D dose delivery scan
    
    Arguments:
        path (str): path to the directory with the dose dataset
    Outputs:
        dose_shape (triple of ints): The shape of the 3D array (columns, rows, depth)
        dose_origin (list of floats): Spacing of the voxels of the corresponding
            3D dose delivery array (in millimeters) given in columns, rows, depth
        dose_spacing (list of floats):  The 3D coordinates of the first voxel of the corresponding
            3D dose delivery array (in millimeters) given in columns, rows, depth
    """
    dose_data = get_dicom_files(path, modality="RTDOSE")
    
    assert (orientation_from_dicom(dose_data)
              == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), "Not yet implemented for this ImageOrientation."
    
    dose_origin = origin_from_dicom(dose_data)
    dose_spacing = spacing_from_dicom(dose_data)
    dose_shape = np.transpose(dose_data.pixel_array, (1,2,0)).shape
    
    return dose_shape, dose_origin, dose_spacing

def align_3d_arrays(trasforming_array_origin, transforming_array_spacing, transforming_array,
                    fixed_array_origin, fixed_array_spacing, fixed_array_shape, fill_value=0.0)
    trsfm_shape = transforming_array.shape
    '''
    Aligns the values of one 3D array to the voxels of another 3D array via interpolation.
    NOTE: This function is dimension agnostic. As long as *ALL* of the spacings and origins match,
        the function will work. For example, it will interpolate (x,y,z) or (z,y,x) as long
        as the inputs all match.
    
    Arguments:
        trasforming_array_origin (list of floats): The 3D physical coordinates of the
            3D array (columns, rows, depth) to be interpolated
        transforming_array_spacing (list of floats): The voxel spacing of the 3D array
            (columns, rows, depth) to be interpolated
        transforming_array (numpy array): The 3D array to use for interpolation
        fixed_array_origin (list of floats): The 3D physical coordinates of the
            3D array (columns, rows, depth) to be interpolated onto
        fixed_array_spacing (list of floats): The voxel spacing of the 3D array
            (columns, rows, depth) to be interpolated onto
        fixed_array_shape (triple of ints): The shape of the 3D array (columns, rows, depth)
            to be interpolated onto
        fill_value (float): this value is filled into voxels that do not overlap
    Outputs:
        transformed_array (numpy array): interpolated array
    '''
    # compute the dose grid in terms of physical coordinates
    x1_values = np.array([float(trasforming_array_origin[0])+i*float(transforming_array_spacing[0])
                          for i in range(trsfm_shape[0])])
    x2_values = np.array([float(trasforming_array_origin[1])+i*float(transforming_array_spacing[1])
                          for i in range(trsfm_shape[1])])
    x3_values = np.array([float(trasforming_array_origin[2])+i*float(transforming_array_spacing[2])
                          for i in range(trsfm_shape[2])])
    
    # lowest/highest values in terms of physical coordinates
    x1_range = (min(x1_values),max(x1_values))
    x2_range = (min(x2_values),max(x2_values))
    x3_range = (min(x3_values),max(x3_values))
    
    # create interpolator
    grid_interpolator = RegularGridInterpolator((x1_values,x2_values,x3_values),transforming_array, 
                                                     bounds_error=False, fill_value=fill_value)
    
    # spacing and origin variables
    dx1, dx2, dx3 = fixed_array_spacing
    px1, px2, px3 = fixed_array_origin
    
    # use mgrid
    new_grid = np.mgrid[px1:px1+(fixed_array_shape[0])*dx1:dx1,
                        px2:px2+(fixed_array_shape[1])*dx2:dx2,
                        px3:px3+(fixed_array_shape[2])*dx3:dx3]
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))
    transformed_array = grid_interpolator(new_grid)
    return transformed_array

def suggest_roi_name(contour_data, name):
    """
    Suggest the name of the contour structure using a standard name
    
    Arguments:
        contour_data (dcm): a DICOM dataset object with modality RTSTRUCT
        name (str): the name of the organ/contour
    Outputs:
        suggestions (list of strs): suggested names for this organ/contour as it
            appears in the ROI names
    """
    names = get_roi_names(contour_data)
    suggestions = []
    name_split = name.split(" ")
    name_split = [word[0].upper()+word[1:] for word in name_split]
    for roi_name in names:
        if any([word in roi_name for word in name_split]) or any([word.lower() in roi_name for word in name_split]):
                suggestions.append(roi_name)
    return suggestions
    
def get_roi_index(contour_data, name):
    """
    Get the ROI index value for a contour dataset from its name
    
    Arguments:
        contour_data (dcm): a DICOM dataset object with modality RTSTRUCT
        name (str): the name of the contour
    Outputs:
        roi_index (int): index value for the desired contour
    """
    names = get_roi_names(contour_data)
    if name not in names:
        warnings.warn(f"Contour {name} is not in contour dataset!")
    else:
        roi_idx = get_roi_names(contour_data).index(name)
        return roi_index

def percent_volume_Rx(organ_mask, normalized_dose_array, Rx_value):
    """
    Get the percentage of a volume receiving Rx_value*100% of the prescription dose
    
    Arguments:
        organ_mask (numpy array): a numpy array corresponding to the filled-in 3D mask
            of a volume
        normalized_dose_array (numpy array): a numpy array corresponding to the 3D dose
            delivery, normalized to the prescription dose
    Outputs:
        percent_volume (float): the percentage of the volume receiving the desired
            percentage of the prescription dose
    """
    assert organ_mask.shape == normalized_dose_array.shape, "Arrays are different sizes!"
    organ_dose = organ_mask*normalized_dose_array
    organ_threshold = np.zeros(organ_mask.shape)
    organ_threshold[organ_dose > Rx_value] = 1
    percent_volume = (np.sum(organ_threshold) / np.sum(organ_mask))
    return percent_volume

def get_center_slice(mask, dim=0):
    """
    Get the center slice of a binary mask with respect to a given dimension
    
    Arguments:
        mask (numpy array): a binary mask of a volume
        dim (0, 1, or 2): the desired dimension
    Outputs:
        slc (int): the slice of the center of the mask with respect to the
            given dimension
    """
    slices = []
    for idx in range(mask.shape[dim]):
        if dim == 0:
            slc = mask[idx]
        if dim == 1:
            slc = mask[:,idx,:]
        if dim == 2:
            slc = mask[:,:,idx]
        if np.sum(slc) > 0.0:
            slices.append(idx)
    first = min(slices)
    last = max(slices)
    slc = int((last + first) / 2)
    return slc
    
    
    