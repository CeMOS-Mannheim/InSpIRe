# ... functions used for the co-registration workflow were provided by Dr. Stefan Schmidt.
import SimpleITK as sitk  # need to install SimpleITK-SimpleElastix
import skimage.exposure as exposure  # (0.19.2)
import numpy as np
import numpy.typing as npt

def register_image_elastix(fixed: npt.NDArray[np.uint8], moving: npt.NDArray[np.uint8], method: str, iterations: int, 
                           fixed_pixel_size: float = 1.0, moving_pixel_size: float = 1.0) -> tuple[npt.NDArray[np.uint8], sitk.ParameterMap]:
    fixed_image = sitk.GetImageFromArray(fixed)
    moving_image = sitk.GetImageFromArray(moving)

    # Data format
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # Pixel size
    # sets spacing (or pixel/voxel resolution)
    fixed_image.SetSpacing((fixed_pixel_size, fixed_pixel_size))
    # sets spacing (or pixel/voxel resolution)
    moving_image.SetSpacing((moving_pixel_size, moving_pixel_size))
    # Mutual information
    elastix_image_filter = sitk.ElastixImageFilter()
    elastix_image_filter.SetFixedImage(fixed_image)
    elastix_image_filter.SetMovingImage(moving_image)

    # ParameterMap
    parameter_map = sitk.GetDefaultParameterMap(method)
    parameter_map['MaximumNumberOfIterations'] = [str(iterations)]

    elastix_image_filter.SetParameterMap(parameter_map)
    elastix_image_filter.Execute()
    sitk.PrintParameterMap(parameter_map)
    print(parameter_map)

    result_image = sitk.Cast(elastix_image_filter.GetResultImage(), sitk.sitkFloat32)
    result = sitk.GetArrayViewFromImage(result_image) # transform to np
    transform_parameter_map = elastix_image_filter.GetTransformParameterMap()
    #to obtain uint8
    result = (exposure.rescale_intensity(result, out_range=(0, 255))).astype('uint8') 
    
    return result, transform_parameter_map


# ... functions used for the co-registration workflow were provided by Dr. Stefan Schmidt.
def transformix_image(image: npt.NDArray[np.uint8], transform_parameter_map: sitk.ParameterMap, transform_pixel_size: float = 1.0) -> npt.NDArray[np.uint8]:
    transformix_image_filter = sitk.TransformixImageFilter()
    transformix_image_filter.SetTransformParameterMap(transform_parameter_map)
    image_transformed = sitk.GetImageFromArray(image)
    image_transformed = sitk.Cast(image_transformed, sitk.sitkFloat64)
    # sets spacing (or pixel/voxel resolution)
    image_transformed.SetSpacing((transform_pixel_size, transform_pixel_size))
    transformix_image_filter.SetMovingImage(image_transformed)
    transformix_image_filter.Execute()
    result_transformed = sitk.Cast(transformix_image_filter.GetResultImage(), sitk.sitkFloat64)
    transformed_image = sitk.GetArrayViewFromImage(result_transformed)

    result = (exposure.rescale_intensity(transformed_image, out_range=(0,255))).astype('uint8')
    return result
