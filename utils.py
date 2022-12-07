import numpy as np
from crop import Crop
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
        self.old_size = (self.sensor_info["height"], self.sensor_info["width"])
        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])

        if self.old_size==self.new_size or self.enable==False:
            print('   - Output size is the same as input size.') if self.is_debug else None
            return self.img

        scaled_img = np.empty((self.new_size[0], self.new_size[1], 3), dtype="uint8")

        for i in range(3):
            ch_arr = self.img[:,:,i]
            scale_2d = Scale_2D(ch_arr, self.sensor_info, self.parm_sca)
            scaled_ch = scale_2d.execute()

            # If input size is invalid, the Scale_2D class returns the original image.          
            if scaled_ch.shape==self.old_size:
                return self.img
            else:    
                scaled_img[:,:,i] = scaled_ch                
            
            # Because each channel is scaled in the same way, the isDeug flag is turned 
            # off after the first channel has been scaled.    
            self.parm_sca["isDebug"]= False

        # convert uint16 img to uint8           
        scaled_img = np.uint8(np.clip(scaled_img, 0, (2**8)-1))
        return scaled_img

################################################################################
class Scale_2D:        
    def __init__(self, img, sensor_info, parm_sca):
        self.img = img.astype("float32")
        self.sensor_info = sensor_info
        self.parm_sca = parm_sca
    
    def downscale_nearest_neighbor(self):

        # print("Downscaling with Nearest Neighbor.")

        old_height, old_width = self.img.shape[0], self.img.shape[1]
        new_height, new_width = self.new_size[0], self.new_size[1]
        
        # As new_size is less than old_size, scale factor is defined s.t it is >1 for downscaling
        scale_height , scale_width = old_height/new_height, old_width/new_width

        if scale_height-int(scale_height)!=0 or scale_width-int(scale_width)!=0:
            print("Scale factor must by an integer!")
            return None

        kernel = np.zeros((int(scale_height), int(scale_width)))
        kernel[0,0] = 1

        scaled_img  = stride_convolve2d(self.img, kernel)
        return scaled_img.astype("uint16")

    def scale_nearest_neighbor(self, new_size):
            
            """
            Upscale/Downscale 2D array by integer scale factor using Nearest Neighbor (NN) algorithm.
            """
            
            # print("Nearest Neighbor...")

            old_height, old_width = self.img.shape[0], self.img.shape[1]
            new_height, new_width = new_size[0], new_size[1]
            scale_height , scale_width = new_height/old_height, new_width/old_width

            scaled_img = np.zeros((new_height, new_width), dtype = "uint16")

            for y in range(new_height):
                for x in range(new_width):
                    y_nearest = int(np.floor(y/scale_height))
                    x_nearest = int(np.floor(x/scale_width))
                    scaled_img[y,x] = self.img[y_nearest, x_nearest]
            return scaled_img
        
    def bilinear_interpolation(self, new_size):
            
            # print("Bilinear interpolation...")
            old_height, old_width      = self.img.shape[0], self.img.shape[1]
            new_height, new_width      = new_size[0], new_size[1]
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
                if bool(red_fact[i]):  # means that no scaling is required (for both height and width)
                    
                    # reduction factor = n/d    --->  Upscale the cropped image n times then downscale d times
                    upscale_fact   = red_fact[i][0] 
                    downscale_fact = red_fact[i][1]
                    
                    # print("upscaling {} by: ".format("height" if i==0 else "width"), upscale_fact)
                    upscale_to_size = (upscale_fact*self.img.shape[0], self.img.shape[1]) if i==0 else \
                                      (self.img.shape[0], upscale_fact*self.img.shape[1])
                    # upscale  = UpScale(self.img, upscale_to_size)
                    if method[0]=="Nearest_Neighbor":
                        self.img = self.scale_nearest_neighbor(upscale_to_size)
                    else:
                        self.img = self.bilinear_interpolation(upscale_to_size)

                    # discard self.old_size = (self.img.shape[0], self.img.shape[1])

                    # print("downscaling {} by: ".format("height" if i==0 else "width"), downscale_fact)
                    downscale_to_size = (int(np.round(self.img.shape[0]/downscale_fact)), self.img.shape[1]) if i==0 else \
                                        (self.img.shape[0], int(np.round(self.img.shape[1]//downscale_fact)))    
                    
                    if method[1]=="Nearest_Neighbor":
                        self.img = self.scale_nearest_neighbor(downscale_to_size)
                    else:
                        self.img = self.downscale_by_int_factor(downscale_to_size)
                        
                    # self.old_size = (self.img.shape[0], self.img.shape[1])
                    
            return self.img

    def downscale_by_int_factor(self, new_size):
            
            """
            Downscale a 2D array by an integer scale factor using 2D convolution.
            
            Assumption: new_size is a integer multiple of old image.
            
            Parameters
            ----------
            new_size: Output size. It may or may not be different than self.new_size. 
            Output: 16 bit scaled image in which each pixel is an average of box nxm 
            determined by the scale factors.  
            """

            scale_height = new_size[0]/self.img.shape[0]             
            scale_width  = new_size[1]/self.img.shape[1]

            box_height = int(np.ceil(1/scale_height)) 
            box_width  = int(np.ceil(1/scale_width))
            
            scaled_img = np.zeros((new_size[0], new_size[1]), dtype = "float32")
            kernel     = np.ones((box_height, box_width))/(box_height*box_width)
            
            scaled_img = stride_convolve2d(self.img, kernel)
            return np.round(scaled_img).astype("uint16")

    def apply_scale_algo(self, method):
        
        # scaled_img = np.empty(self.new_size, dtype="float32")
        valid_h = [1080, 1536, 1944]
        sizes   = [OUT_1080x1920, OUT_1536x2592, OUT_1944x2592]
        idx     = valid_h.index(self.old_size[0])
       
        size_obj   = sizes[idx](self.new_size)
        scale_info = size_obj.scale_dict
        
        if scale_info == [[1,0,None], [1,0,None]]:
            print("Invalid output size. Choose from the following:\n"\
                  "   - valid height = {}\n   - valid widths = {}"\
                  .format(size_obj.valid_sizes[0],size_obj.valid_sizes[1]))
            return self.img    
        else:
            # step 1: Downscale by int fcator
            if scale_info[0][0]>1 or scale_info[1][0]>1:
                self.img = self.downscale_by_int_factor((self.old_size[0]//scale_info[0][0], 
                                                        self.old_size[1]//scale_info[1][0]))
                # self.old_size = self.img.shape
                print("   - Shape after downscaling by integer factor {}:  {}".format((scale_info[0][0],\
                        scale_info[1][0]),self.img.shape)) if self.is_debug else None

            # step 2: crop
            if scale_info[0][1]>0 or scale_info[1][1]>0:
                crop_obj = Crop(self.img, self.sensor_info, self.parm_sca)
                self.img = crop_obj.crop(self.img, scale_info[0][1], 
                                                    scale_info[1][1])
                print("   - Shape after cropping {}:  {}".format((scale_info[0][1],\
                        scale_info[1][1]),self.img.shape)) if self.is_debug else None                                    
            
            # step 3: Scale with non-int factor
            if bool(scale_info[0][2])==True or bool(scale_info[1][2])==True:
                self.img = self.resize_by_non_int_fact((scale_info[0][2], 
                                                    scale_info[1][2]), method)
                print("   - Shape after scaling by non-integer factor {}:  {}".format((scale_info[0][2],\
                        scale_info[1][2]),self.img.shape)) if self.is_debug else None
            return self.img

    def execute(self):
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
            self.img = self.apply_scale_algo([self.upscale_method, self.downscale_method])
        else:
            print("   - Scaling with {} method...".format(self.algo)) if self.is_debug else None
            if self.algo=="Nearest_Neighbor":
                self.img = self.scale_nearest_neighbor()
            else:
                self.img = self.bilinear_interpolation()    

        print("   - Shape of scaled image for a single channel = ", self.img.shape) if self.is_debug else None
        return self.img

#############################################################################

def stride_convolve2d(matrix, kernel):
    return correlate2d(matrix, kernel, mode="valid")[::kernel.shape[0], ::kernel.shape[1]]

#############################################################################
# The structure and working of the following three classes is exactlly the same
class OUT_1080x1920:
  """
    The hardware friendly approach can only be used for specific input and output sizes.
    This class checks if the given output size can be achieved by this approach and 
    creates a list with corresponding constants used to execute this scaling approach 
    comprising of the following three steps:
    
    1. Downscale with int factor 
    2. Crop  
    3. Scale with non-integer-factor 

    Class Attributes:
    ----------------
    VALID_SIZES [list]: A nested list with two sublists enlisting valid heights(index 0) and 
    widths(index 1) respectively. 

    SCALE_INFO [dict]: Reference dictionary containing constants used to scale the input to 
    each of the valid output sizes using the above steps. This dictionary is structured
    in correspondence to VALID_SIZES i.e. each key is a nested list similar to VALID_SIZES and 
    index 0 in each of the sublists (for height and width) are the constants which are to be 
    used for scaling the input to the size at index 0 in VALID_SIZES.
    
    The elements in each list correspond to the following constants:
    
    1. Scale factor [int]: (default 1) scale factor for downscaling.   
    2. Crop value [int] : (defaut 0) number of rows or columns to be croped.
    3. Non-int scale factor [tuple with 2 enteries]: (default None) a rational scale factor 
       of form n/d where n is the first entry (index 0) and d is the second entry(index 1)
       of this tuple.   
    
    Instance Attributes:
    -------------------  
    SCALE_list:  a nested list with two sublists containing constants used in order to
    scale height (index 0) and width (index 1) to the given NEW_SIZE using the three 
    steps above.
    
    """  
  
    def __init__(self, new_size):
        self.new_size = new_size
        self.valid_sizes = [[720,480,360], [1280,640]]
        
        if new_size[0] not in self.valid_sizes[0] or new_size[1] not in self.valid_sizes[1]:
            # if size is invalid, no scaling if performed and original image is returned 
            self.scale_dict = [[1,0,None],[1,0,None]]          
        else:
            self.configure()

    def configure(self):
        self.scale_dict = [[], []]
        scale_info = {"int_downscale": [[1, 2, 3], [1, 3]],
                      "crop": [[0,60,0],[0,0]],
                      "non_int_scale":[[(2,3), None, None],[(2,3), None]]}
        for i in range(2):
            idx = self.valid_sizes[i].index(self.new_size[i])
            self.scale_dict[i].append(scale_info["int_downscale"][i][idx])
            self.scale_dict[i].append(scale_info["crop"][i][idx])
            self.scale_dict[i].append(scale_info["non_int_scale"][i][idx])

#############################################################################
class OUT_1536x2592:
    
    def __init__(self, new_size):
        self.new_size = new_size
        self.valid_sizes = [[1080,720,480,360], [1920,1280,640]]
        
        if new_size[0] not in self.valid_sizes[0] or new_size[1] not in self.valid_sizes[1]:
            # if size is invalid, no scaling if performed and original image is returned 
            self.scale_dict = [[1,0,None],[1,0,None]]          
        else:
            self.configure()

    def configure(self):
        self.scale_dict = [[], []]
        scale_info = {"int_downscale": [[1,2,3,4], [1,2,4]],
                      "crop": [[24,48,32,24],[32,16,6]],
                      "non_int_scale":[[(5,7), None, None, None],[(3,4), None, None]]}
        for i in range(2):
            idx = self.valid_sizes[i].index(self.new_size[i])
            self.scale_dict[i].append(scale_info["int_downscale"][i][idx])
            self.scale_dict[i].append(scale_info["crop"][i][idx])
            self.scale_dict[i].append(scale_info["non_int_scale"][i][idx])
#############################################################################
class OUT_1944x2592:
    
    def __init__(self, new_size):
        self.new_size = new_size
        self.valid_sizes = [[1440,1080,960,720,480,360], [2560,1920,1280,640]]
        
        if new_size[0] not in self.valid_sizes[0] or new_size[1] not in self.valid_sizes[1]:
            # if size is invalid, no scaling if performed and original image is returned 
            self.scale_dict = [[1,0,None],[1,0,None]]          
        else:
            self.configure()

    def configure(self):
        self.scale_dict = [[], []]
        scale_info = {"int_downscale": [[1,1,2,2,4,4], [1,1,2,4]],
                      "crop": [[24,54,12,12,6,6],[32,32,16,8]],
                      "non_int_scale":[[(3,4),(4,7), None,(3,4), None, (3,4)],[None,(3,4), None, None]]}
        for i in range(2):
            idx = self.valid_sizes[i].index(self.new_size[i])
            self.scale_dict[i].append(scale_info["int_downscale"][i][idx])
            self.scale_dict[i].append(scale_info["crop"][i][idx])
            self.scale_dict[i].append(scale_info["non_int_scale"][i][idx])
