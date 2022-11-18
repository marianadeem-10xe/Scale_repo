import numpy as np
import modules.crop as crop
from scipy.signal import correlate2d

class Scale:
    def __init__(self, img, sensor_info, parm_sca):
        self.img = img.astype("uint8")
        self.enable = parm_sca['isEnable']
        self.sensor_info = sensor_info
        self.parm_sca = parm_sca

    def execute(self):

        print('Scale = ' + str(self.enable))
        self.is_debug = self.parm_sca["isDebug"]

        if self.old_size==self.new_size or self.enable==False:
            print('   - Output size is the same as input size.') if self.is_debug else None
            return self.img

        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])
        scaled_img = np.empty((self.new_size[0], self.new_size[1], 3), dtype="uint8")
        
        for i in range(3):
            ch_arr = self.img[:,:,i]
            scale_2d = Scale2D(ch_arr, self.sensor_info, self.parm_sca)
            scaled_img[:,:,i] = scale_2d.execute()                
            self.parm_sca["isDebug"]= False

        # convert uint16 img to uint8           
        scaled_img = np.uint8(np.clip(scaled_img, 0, (2**8)-1))
        return scaled_img

class Scale2D:
    def __init__(self, img, sensor_info, parm_sca):
        self.img         = img.astype('float32')
        self.sensor_info = sensor_info
        self.parm_sca    = parm_sca
        self.output_sizes= [(1440, 2560),(1080,1920),\
                            (960,1280),  (480,640)]
    
    def optimal_reduction_factor(self, curr_size, required_size):
        
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
        
        numertaors   = [3,1,2,3,4,5,4]
        denominators = [4,5,5,5,5,6,7]
        
        min_crop_val, min_fact = [np.inf, np.inf], [0,0]
        
        for i in range(2):
            for fraction in zip(numertaors, denominators):
                # crop then scale: (old_size - crop) * reduction factor = new_size
                fact = fraction[0]/fraction[1]
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
        
        print("   -  Bilinear Downscaling height and width by ", scale_fact) if self.is_debug else None

        down_scale = DownScale(self.img, (self.old_size[0], self.old_size[1]))
        self.img   = down_scale.downscale_bilinear()
        return self.img.astype("float32")

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
            if red_fact[i]==0:  # means that no scaling is required (for both height and width)
                continue
            else:
                # reduction factor = n/d    --->  Upscale the cropped image n times then downscale d times
                upscale_fact      = red_fact[i][0] 
                downscale_fact    = red_fact[i][1]
                
                if self.is_debug:
                    print("   - {} upscaling {} by: ".format(method[0],"height" if i==0 else "width"), upscale_fact)
                    
                upscale_to_size   = (upscale_fact*self.old_size[0], self.old_size[1]) if i==0 else \
                                    (self.old_size[0], upscale_fact*self.old_size[1])
                upscale           = UpScale(self.img, upscale_to_size)
                self.img          = upscale.execute(method[0])
                self.old_size     = (self.img.shape[0], self.img.shape[1])

                if self.is_debug:
                    print("   - {} downscaling {} by: ".format(method[1],"height" if i==0 else "width"), downscale_fact)
             
                downscale_to_size = (self.old_size[0]//downscale_fact, self.old_size[1]) if i==0 else \
                                    (self.old_size[0], self.old_size[1]//downscale_fact)    
                downscale         = DownScale(self.img, downscale_to_size)
                self.img          = downscale.execute(method[1])
                self.old_size     = (self.img.shape[0], self.img.shape[1])
                
        return self.img.astype("float32")

    def execute(self):
        
        """
        Rescale an input 2D array of size 2592x1944 to one of the following sizes:
        - 2560x1440
        - 1920x1080
        - 1280x960
        - 640x480

        Step1: Downscale array by an even factor.
        Step2: Crop array
        Step3: Downscale by a non-integer factor.

        Output: scaled array. 
        """
        self.old_size = (self.sensor_info["height"], self.sensor_info["width"])
        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])
        self.is_debug = self.parm_sca["isDebug"]
        self.is_hardware      = self.parm_sca["isHardware"] 
        self.algo             = self.parm_sca["Algo"] 
        self.upscale_method   = self.parm_sca["upscale_method"] 
        self.downscale_method = self.parm_sca["downscale_method"] 
        
        if self.new_size == self.old_size:
           return self.img 
        
        if self.is_hardware:
            assert self.new_size in self.output_sizes, "Invalid output size {}!\n".format(self.new_size)
            
            self.img      = self.downscale_to_half_ntimes()
            self.old_size = self.img.shape

            # Crop and scale the image further if needed
            if self.old_size!=self.new_size:
                crop_val, red_fact = self.optimal_reduction_factor(list(self.old_size), list(self.new_size))
                
                print("   - Crop_val, Red_fact:", crop_val, red_fact) if self.is_debug else None
            
                # Crop img
                crop_obj      = crop.Crop(self.img, self.sensor_info, self.parm_sca ) 
                cropped_img    = crop_obj.crop(self.img, crop_val[0], crop_val[1])
                
                self.img      = cropped_img.copy()
                self.old_size = self.img.shape
                
                # Resize if needed.
                if self.old_size!=self.new_size:
                    self.img = self.resize_by_non_int_fact(red_fact, [self.upscale_method, self.downscale_method])       
            
            self.img = np.uint16(np.clip(self.img, 0, (2**12)-1))
            
        else:
            scale = UpScale(self.img, self.new_size)
            print("   - Scaling with {} method...".format(self.algo)) if self.is_debug else None
            if self.algo=="Nearest_Neighbor":
                self.img = scale.scale_nearest_neighbor()
            else:
                self.img = scale.bilinear_interpolation()    

        print("   - Shape of scaled image for a single channel = ", self.img.shape) if self.is_debug else None
        return self.img
##########################################################################
class UpScale:
    def __init__(self, img, new_size):
        self.img = img.astype("float32")
        self.old_size = img.shape
        self.new_size = new_size
    
    def round_off(self, n):
        if n-int(n)<0.5:
            return np.floor(n)
        return np.ceil(n) 
    
    def scale_nearest_neighbor(self):
        
        """
        Upscale/Downscale 2D array using Nearest Neighbor (NN) algorithm 
        by any (real number) scale factor.
        """
        # print("upscaling with Nearest Neighbor") 
        
        old_height, old_width = self.img.shape[0], self.img.shape[1]
        new_height, new_width = self.new_size[0], self.new_size[1]
        scale_height , scale_width = new_height/old_height, new_width/old_width

        scaled_img = np.zeros((new_height, new_width), dtype = "float32")
    
        for y in range(new_height):
            for x in range(new_width):
                y_nearest = int(np.floor(y/scale_height))
                x_nearest = int(np.floor(x/scale_width))
                scaled_img[y,x] = self.img[y_nearest, x_nearest]
        
        scaled_img = np.uint16(np.clip(scaled_img, 0, (2**12)-1))
        return scaled_img
    
    def bilinear_interpolation(self):
        
        """Upscale/Downscale 2D array using bilinear interpolation method
         using any (real number) factor."""

        # print("upscaling with Bilinear Interpolation")

        old_height, old_width      = self.img.shape[0], self.img.shape[1]
        new_height, new_width      = self.new_size[0], self.new_size[1]
        scale_height , scale_width = new_height/old_height, new_width/old_width
        
        scaled_img  = np.empty((new_height, new_width), dtype = "float32")
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
        
        scaled_img = np.uint16(np.clip(scaled_img, 0, (2**12)-1))
        return scaled_img 

    def execute(self, method):
        if method == "Nearest_Neighbor":
            return self.scale_nearest_neighbor()
        return self.bilinear_interpolation()

##########################################################################
class DownScale:
    """This class is only used to downscale a 2D array by int factor."""
    
    def __init__(self, img, new_size):
        self.img = img.astype("float32")
        self.old_size = img.shape
        self.new_size = new_size
    
    def downscale_nearest_neighbor(self):

        # print("Downscaling with Nearest Neighbor.")

        old_height, old_width = self.img.shape[0], self.img.shape[1]
        new_height, new_width = self.new_size[0], self.new_size[1]
        
        # As new_size is less than old_size, scale factor is defined s.t it is >1 for downscaling
        scale_height , scale_width = old_height/new_height, old_width/new_width

        assert scale_height-int(scale_height)==0 and \
               scale_width-int(scale_width)==0, "Scale factor must by an integer!"

        kernel = np.zeros((int(scale_height), int(scale_width)))
        kernel[0,0] = 1

        scaled_img  = stride_convolve2d(self.img, kernel)
        scaled_img  = np.uint16(np.clip(scaled_img, 0, (2**12)-1))
        return scaled_img 

    def downscale_bilinear(self):
        
        """
        Downscale a 2D array by an integer scale factor using Bilinear method (average) 
        employing 2D convolution.
        
        Assumption: new_size is a integer multiple of old image.
        
        Parameters
        ----------
        new_size: Output size. It may or may not be different than self.new_size. 
        Output: 16 bit scaled image in which each pixel is an average of box nxm 
        determined by the scale factors.  
        """
        # print("Downscaling with Bilinear method.")

        scale_height = self.new_size[0]/self.old_size[0]             
        scale_width  = self.new_size[1]/self.old_size[1]

        box_height = int(np.ceil(1/scale_height)) 
        box_width  = int(np.ceil(1/scale_width))
        
        kernel     = np.ones((box_height, box_width))/(box_height*box_width)
        
        scaled_img = stride_convolve2d(self.img, kernel)
        scaled_img = np.uint16(np.clip(np.round(scaled_img), 0, (2**12)-1))
        
        return scaled_img    

    def execute(self, method):
        if method=="Nearest_Neighbor":
            return self.downscale_nearest_neighbor()
        return self.downscale_bilinear()  

########################################################################## 

def stride_convolve2d(matrix, kernel):
    return correlate2d(matrix, kernel, mode="valid")[::kernel.shape[0], ::kernel.shape[1]]

##########################################################################