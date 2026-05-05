from functools import wraps
from time import perf_counter
import SimpleITK as sitk

def logger(func):

    @wraps(func)
    def _decorated(*args, **kwargs):
        start = perf_counter()
        res = func(*args, **kwargs)
        end = perf_counter()
        time_elapsed = round(end - start)
        mins = time_elapsed // 60
        secs = time_elapsed % 60
        print(f'{func.__name__}: {mins} min {secs} sec.')
        return res
    return _decorated

def save_nii(arr, p, spacing=(1, 1, 1)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    writer = sitk.ImageFileWriter()
    writer.SetFileName(p)
    writer.Execute(img)
