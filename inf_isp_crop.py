import numpy as np

class Crop:
    """
    Parameters:
    ----------
    img: Coloured image (3 channel)
    new_size: 2-tuple with required height and width.
    
    Return:
    ------
    cropped_img: Generated coloured image of required size with same dtype 
    as the input image.
    """
    def __init__(self, img, sensor_info, parm_cro):
        self.img = img
        self.old_size = (sensor_info["height"], sensor_info["width"])
        self.new_size = (parm_cro["new_height"], parm_cro["new_width"])
        self.enable   = parm_cro["isEnable"]
        self.is_debug = parm_cro["isDebug"]

    def crop(self, img, rows_to_crop=0, cols_to_crop=0):
        
        """
        Crop 2D array.
        Parameter:
        ---------
        img: image (2D array) to be cropped.
        rows_to_crop: Number of rows to crop. If it is an even integer, 
                      equal number of rows are cropped from either side of the image. 
                      Otherwise the image is cropped from the extreme right/bottom.
        cols_to_crop: Number of columns to crop. Works exactly as rows_to_crop.
        
        Output: cropped image
        """
        
        if rows_to_crop:
            if rows_to_crop%2==0:
                img = img[rows_to_crop//2:-rows_to_crop//2, :]
            else:
                img = img[0:-1, :]
        if cols_to_crop:         
            if cols_to_crop%2==0:
                img = img[:, cols_to_crop//2:-cols_to_crop//2]
            else:
                img = img[:, 0:-1] 
        return img 

    def execute(self):
        
        print('Crop = ' + str(self.enable))
        
        if self.old_size==self.new_size or self.enable==False:
            print('   - Output size is the same as input size.')
            return self.img
            
        assert self.old_size[0]>self.new_size[0] and self.old_size[1]>self.new_size[1], \
        "Invalid output size {}x{}.\nMake sure output size is smaller than input size!".format( \
        
        self.new_size[0], self.new_size[1]) 
        
        crop_rows = self.old_size[0] - self.new_size[0]
        crop_cols = self.old_size[1] - self.new_size[1]
        cropped_img  = np.empty((self.new_size[0], self.new_size[1], 3), dtype=self.img.dtype)
        
        for i in range(3):
            cropped_img[:, :, i] = self.crop(self.img[:, :, i],crop_rows, crop_cols)
        
        if self.is_debug:
                print('   - Number of rows cropped = ', crop_rows)
                print('   - Number of columns cropped = ', crop_cols)
                print('   - Shape of cropped image = ', cropped_img.shape)
        return cropped_img
##########################################################################