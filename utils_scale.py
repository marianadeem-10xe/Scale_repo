from math import log10, sqrt
import numpy as np
import pandas as pd

class Scale:
    def __init__(self, img, new_size):
        self.img = img.astype("float32")
        self.new_size = new_size
        self.old_size = (img.shape[0], img.shape[1])
    
    def optimal_reduction_factor(self, curr_size, required_size, crop):
        
        """
        Compute the minimum number of rows and columns to be cropped 
        such that after cropping, the size of the array becomes a multiple 
        of the required output size i.e. 
                required_size = q*cropped_size 
        where q is taken from [3/4, 1/5, 2/5, 3/5, 4/5, 5/6, 4/7].
        
        Output: 
        min_crop_val: list with number of rows and columns to be cropped  
        min_red_fact: list with scale fcators for rows and columns. Enteries 
                      of the list are tuples where first entry is the numerator and the second
                      is the denominator of the no-integer scale factor. If reduction factor is 
                      zero, it means no scaling is needed after cropping the array.
        """
        # list of reduction factors: [3/4, 1/5, 2/5, 3/5, 4/5, 5/6, 4/7]
        
        print("Computing reduction factors...")
        numertaors   = [3,3,1,2,3,4,5,4]
        denominators = [2,4,5,5,5,5,6,7]
        min_crop_val, min_fact = [np.inf, np.inf], [1,1]  

        for i in range(2):
            for fraction in zip(numertaors, denominators):
                # crop then scale: (old_size - crop) * reduction factor = new_size
                fact = fraction[0]/fraction[1]
                if crop==False:
                    min_crop_val = [0, 0]
                    if np.round(curr_size[i]*fact)==np.round(required_size[i]):
                        min_fact[i] = fraction  
                else:
                    crop_val = curr_size[i] - (required_size[i]/fact)
                    if crop_val < min_crop_val[i] and crop_val>0: 
                        min_crop_val[i], min_fact[i] = int(crop_val), fraction
        
        # if all factors fail, crop directly
        while np.inf in min_crop_val:
            idx = min_crop_val.index(np.inf)
            min_crop_val[idx] = curr_size[idx] - required_size[idx]
        return min_crop_val, min_fact
    
    def downscale_to_half_ntimes(self):
       
        """Reduce the input img (2D array) to half its size as many times as possible."""
        
        scale_fact = 1
        while (self.old_size[0]%2 == 0 and self.old_size[0]//2 > self.new_size[0]) and \
              (self.old_size[1]%2 == 0 and self.old_size[1]//2 > self.new_size[1]):
              scale_fact*=2
              self.old_size = (self.old_size[0]//2, self.old_size[1]//2)

        if scale_fact==1:
            return self.img
        
        print("Downscaling height and width by ", scale_fact)        
        down_scale = DownScale(self.img, (self.old_size[0], self.old_size[1]))
        self.img   = down_scale.downscale_by_int_factor()
        return self.img

    def resize_by_non_int_fact(self, red_fact, method):
        
        """"
        Resize 2D array by non-inteeger factor n/d.
        Firstly, the array is upsacled n times then downscaled d times.

        Parameter:
        ---------
        red_fact: list with scale factors for height and width.
        method: list with algorithms used for upscaling(at index 0) and downscaling(at index 1).
        Output: scaled img
        """
        
        for i in range(2):
            if red_fact[i]==1:  # means that no scaling is required (for both height and width)
                continue
            else:
                # reduction factor = n/d    --->  Upscale the cropped image n times then downscale d times
                upscale_fact   = red_fact[i][0] 
                downscale_fact = red_fact[i][1]
                
                print("upscaling {} by: ".format("height" if i==0 else "width"), upscale_fact)
                upscale_to_size = (upscale_fact*self.old_size[0], self.old_size[1]) if i==0 else \
                                    (self.old_size[0], upscale_fact*self.old_size[1])
                upscale  = UpScale(self.img, upscale_to_size)
                self.img = upscale.execute(method[0])
                self.old_size = (self.img.shape[0], self.img.shape[1])

                print("downscaling {} by: ".format("height" if i==0 else "width"), downscale_fact)
                downscale_to_size = (int(np.round(self.old_size[0]/downscale_fact)), self.old_size[1]) if i==0 else \
                                    (self.old_size[0], int(np.round(self.old_size[1]//downscale_fact)))    
                downscale     = DownScale(self.img, downscale_to_size)
                self.img      = downscale.execute(method[1])
                self.old_size = (self.img.shape[0], self.img.shape[1])
                
        return self.img

    def execute(self, method=["Nearest_Neighbor", ""], crop=True):
        
        """
        Rescale an input 2D array of size 2592x1944 to one of the following sizes:
        - 2560x1440
        - 1920x1080
        - 1280x720
        - 640x480

        Step1: Downscale array by an even factor.
        Step2: Crop array
        Step3: Downscale by a non-integer factor.

        Output: scaled array. 
        """
        output_h = [1440, 1080, 720, 480]
        output_w = [2560, 1920, 1280, 640]
        
        """assert self.new_size[0] in output_h or self.new_size[1] in output_w,\
            "Output size must be one of the following:\n"\
            "- 2560x1440,\n- 1920x1080,\n- 1280x720,\n- 640x480"
        """   
        
        scaled_img    = self.downscale_to_half_ntimes()
        self.old_size = (scaled_img.shape[0], scaled_img.shape[1])

        # Crop and scale the image further if needed
        if self.old_size!=self.new_size:
            crop_val, red_fact = self.optimal_reduction_factor(list(self.old_size), list(self.new_size), crop)
            print("crop_val, red_fact:", crop_val, red_fact)
        
            # Crop img
            if crop:
                down_scale = DownScale(scaled_img, (self.old_size[0]-crop_val[0], self.old_size[1]-crop_val[1])) 
                self.img   = down_scale.crop(scaled_img, crop_val[0], crop_val[1])
                self.old_size = self.img.shape[0], self.img.shape[1]
                print("cropped img to size: ", self.img.shape)
            
            # Resize if needed.
            if self.old_size!=self.new_size:
               scaled_img = self.resize_by_non_int_fact(red_fact, method)       
            else:
               scaled_img = self.img.copy()

        return scaled_img    
##########################################################################
class UpScale(Scale):
    
    def scale_nearest_neighbor(self):
        
        """
        Upscale/Downscale 2D array by integer scale factor using Nearest Neighbor (NN) algorithm.
        """
        old_height, old_width = self.img.shape[0], self.img.shape[1]
        new_height, new_width = self.new_size[0], self.new_size[1]
        scale_height , scale_width = new_height/old_height, new_width/old_width

        scaled_img = np.zeros((new_height, new_width), dtype = "uint16")

        for y in range(new_height):
            for x in range(new_width):
                y_nearest = int(np.floor(y/scale_height))
                x_nearest = int(np.floor(x/scale_width))
                scaled_img[y,x] = self.img[y_nearest, x_nearest]
        return scaled_img
    
    def scale_bilinear_method(self):
        
        """Upscale/Downscale an image using bilinear interpolation method
         using an integer factor."""
        
        old_height, old_width = self.img.shape[0], self.img.shape[1]
        new_height, new_width = self.new_size[0], self.new_size[1]
        scale_height , scale_width = new_height/old_height, new_width/old_width

        scaled_img = np.zeros((new_height, new_width), dtype = "float32")
    
        for y in range(new_height):
            for x in range(new_width):
                
                # Coordinates in old image
                old_y, old_x = y/scale_height, x/scale_width

                x1 = min(int(np.floor(old_x)), old_width-1)
                y1 = min(int(np.floor(old_y)), old_height-1)
                x2 = min(int(np.ceil(old_x)), old_width-1)
                y2 = min(int(np.ceil(old_y)), old_height-1)
                
                # Get four neghboring pixels
                Q11 = self.img[y1, x1]
                Q12 = self.img[y1, x2]
                Q21 = self.img[y2, x1]
                Q22 = self.img[y2, x2]

                # Interpolating P1 and P2
                P1 = (x2-old_x)*Q11 + (old_x-x1)*Q12
                P2 = (x2-old_x)*Q21 + (old_x-x1)*Q22

                # The case where the new pixel lies between two pixels
                if x1 == x2:
                    P1 = Q11
                    P2 = Q22

                # Interpolating P
                P = (y2-old_y)*P1 + (old_y-y1)*P2    

                scaled_img[y,x] = np.round(P)
        return scaled_img.astype("uint16") 

    def execute(self, method):
        print("upscaling using {}".format(method if method else "Bilinear Interpolation"))
        if method == "Nearest_Neighbor":
            return self.scale_nearest_neighbor()
        return self.scale_bilinear_method()
##########################################################################
class DownScale(Scale):
    
    def crop(self, img, rows_to_crop=0, cols_to_crop=0):
        
        """
        Crop 2D array.

        Parameter:
        ---------
        img: image (2D array) to be cropped.
        rows_to_crop: Number of rows to crop. If it is an even integer, 
                      equal number of rows are cropped from either side of the image. 
                      Otherwise the image is cropped from the extreme right.
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

    def downscale_by_int_factor(self, mode="average"):
        
        """
        Downscale a 2D array by an integer scale factor.
        
        Assumption: new_size is a integer multiple of old image.
        
        Parameters
        ----------
        mode: str, "average" or max
        Method to downscale a window to a single pixel.

        Output: 16 bit Scaled image.  
        """
        
        self.scale_height = self.new_size[0]/self.old_size[0]             
        self.scale_width = self.new_size[1]/self.old_size[1]
        
        """assert self.old_size[0]%self.new_size[0]==0 and self.old_size[1]%self.new_size[1]==0, \
            "scale factor is not an integer."
        """
        box_height = int(np.ceil(1/self.scale_height)) 
        box_width  = int(np.ceil(1/self.scale_width))
        
        scaled_img = np.zeros((self.new_size[0], self.new_size[1]), dtype = "float32")

        for y in range(self.new_size[0]):
            for x in range(self.new_size[1]):
                
                y_old = int(np.floor(y/self.scale_height))
                x_old = int(np.floor(x/self.scale_width))
                
                y_end = min(y_old + box_height, self.old_size[0])
                x_end = min(x_old + box_width, self.old_size[1])
                
                if mode == "max":
                    scaled_img[y,x] = np.amax(self.img[y_old:y_end, x_old:x_end])
                else:     
                    scaled_img[y,x] = np.average(self.img[y_old:y_end, x_old:x_end])
        
        return np.round(scaled_img).astype("uint16")    

    def execute(self, method):
        print("downscaling using {}".format(method if method else "Bilinear Interpolation"))
        if method=="Nearest_Neighbor":
            scale_obj = UpScale(self.img, self.new_size)
            return scale_obj.execute("Nearest_Neighbor")
        return self.downscale_by_int_factor()  

##########################################################################
# Object to compile results in a csv file
class Results:
    def __init__(self):
        self.confusion_pd = pd.DataFrame(np.zeros((1,5)), columns=["Filename", "Scaled from", "Scaled to","MSE", "PSNR"])
    
    def add_row(self,row):
        self.confusion_pd = pd.concat([self.confusion_pd, pd.DataFrame(np.array(row, dtype=object).reshape(1,5), columns=["Filename", "Scaled from", "Scaled to","MSE", "PSNR"])], ignore_index=False)
    
    def save_csv(self, path, filename):
        self.confusion_pd.to_csv(path + "/" +filename + ".csv", index=False)

##########################################################################

def Evaluation(cv2_img, scaled_img):
    error, PSNR = 0, 0
    for ch in range(3):
        error += np.round((np.add(cv2_img.astype("float32"), - scaled_img.astype("float32"))**2).sum()/cv2_img.size, 4)
        try:
            PSNR  += round(20*log10(4095/sqrt(error)), 4)
        except ZeroDivisionError:
            pass
    
    error, PSNR = error/3, PSNR/3
    print("-"*50)
    print("MSE: ", error)
    print("PSNR: ", PSNR)
    print("-"*50)
    return [error, PSNR]
