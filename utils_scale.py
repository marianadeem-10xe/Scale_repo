from math import log10, sqrt
import numpy as np
import pandas as pd

class Scale:
    def __init__(self, img, new_size):
        self.img = img.astype("float32")
        self.new_size = new_size
        self.old_size = (img.shape[0], img.shape[1])
        self.output_sizes = [(1944,2592), (2304,1296), (1920,1080), (1280,720), (640,360)]

    def round_off(self, n):
        if n-int(n)<0.5:
            return np.floor(n)
        return np.ceil(n)    
    
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
    
    def bilinear_interpolation(self):
        old_height, old_width      = self.img.shape[0], self.img.shape[1]
        new_height, new_width      = self.new_size[0], self.new_size[1]
        scale_height , scale_width = new_height/old_height, new_width/old_width
        
        scaled_img  = np.zeros((new_height, new_width), dtype = "float32")
        old_coor    = lambda a, scale_fact: (a+0.5)/scale_fact - 0.5
        
        for y in range(new_height):
            for x in range(new_width):

                # Coordinates in old image
                old_y, old_x = old_coor(y, scale_height), old_coor(x, scale_width)
                
                x1 = 0 if np.floor(old_x)<0 else min(int(np.floor(old_x)), old_width-1)
                y1 = 0 if np.floor(old_y)<0 else min(int(np.floor(old_y)), old_height-1)
                x2 = 0 if np.ceil(old_x)<0 else min(int(np.ceil(old_x)), old_width-1)
                y2 = 0 if np.ceil(old_y)<0 else min(int(np.ceil(old_y)), old_height-1)
                
                # Get four neghboring pixels
                Q11 = self.img[y1, x1]
                Q12 = self.img[y1, x2]
                Q21 = self.img[y2, x1]
                Q22 = self.img[y2, x2]

                # Interpolating P1 and P2
                weight_right = old_x- np.floor(old_x)
                weight_left  = 1-weight_right 
                P1 = weight_left*Q11 + weight_right*Q12
                P2 = weight_left*Q21 + weight_right*Q22

                # The case where the new pixel lies between two pixels
                if x1 == x2:
                    P1 = Q11
                    P2 = Q22

                # Interpolating P
                weight_bottom = old_y - np.floor(old_y)
                weight_top = 1-weight_bottom 
                P = weight_top*P1 + weight_bottom*P2    

                scaled_img[y,x] = self.round_off(P)
        return scaled_img.astype("uint16")

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
                # upscale  = UpScale(self.img, upscale_to_size)
                if method[0]=="Nearest_Neighbor":
                    self.img = self.scale_nearest_neighbor()
                else:
                    self.img = self.bilinear_interpolation()

                self.old_size = (self.img.shape[0], self.img.shape[1])

                print("downscaling {} by: ".format("height" if i==0 else "width"), downscale_fact)
                downscale_to_size = (int(np.round(self.old_size[0]/downscale_fact)), self.old_size[1]) if i==0 else \
                                    (self.old_size[0], int(np.round(self.old_size[1]//downscale_fact)))    
                # downscale     = DownScale(self.img, downscale_to_size)
                if method[1]=="Nearest_Neighbor":
                    self.img = self.scale_nearest_neighbor()
                else:
                    self.img = self.bilinear_interpolation()
                    
                self.old_size = (self.img.shape[0], self.img.shape[1])
                
        return self.img
    
    def get_red_fact(self):
        img_h, img_w = self.old_size[0], self.old_size[1]
        
        red_fcat_h = [(8,9), (5,6), (2,3), (1,2), (1,1)] 
        red_fcat_w = [(2,3), (5,6), (2,3), (1,2), (1,1)]
        
        size_idx = self.output_sizes.index((img_h,img_w))
        
        return [red_fcat_h[size_idx], red_fcat_w(size_idx)]  
    
    def execute(self, method=["Nearest_Neighbor", ""], crop=True):
        
        """
        Rescale an input 2D array of size 2592x1944 to one of the following sizes:
        - 2304x1296
        - 1920x1080
        - 1280x720
        - 640x360

        Output: scaled array. 
        """
        while self.old_size==self.new_size:
            red_fact = self.get_red_fact()
            scaled_img    = self.resize_by_non_int_fact(red_fact, method)
            self.old_size = (scaled_img.shape[0], scaled_img.shape[1])

        return scaled_img    
##########################################################################
class UpScale(Scale):
    
     

    def execute(self, method):
        print("upscaling using {}".format(method if method else "Bilinear Interpolation"))
        if method == "Nearest_Neighbor":
            return self.scale_nearest_neighbor()
        return self.bilinear_interpolation()

##########################################################################
class DownScale(Scale):    

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

##########################################################################

