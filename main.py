import nibabel as nb 
import matplotlib.pyplot as mp
img = nb.load('sample.nii')
data = img.get_fdata()
print("MRI dimensions:", data.shape)
slice_index = data.shape[2]//2
mp.imshow(data[:, :, slice_index], cmap="gray")
mp.title("Brain MRI Slice")
mp.axis("off")
mp.show()