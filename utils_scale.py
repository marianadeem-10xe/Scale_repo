from unicodedata import decimal
import numpy as np
import colour_demosaicing as cd

class Scale:
    def __init__(self, img, new_size):
        self.img = img.astype("float32")
        self.new_size = new_size
        self.old_size = (img.shape[0], img.shape[1])
    
    def optimal_reduction_factor(self, curr_size, required_size):
        # list of reduction factors: [3/4, 1/5, 2/5, 3/5, 4/5, 5/6, 4/7]
        
        numertaors   = [3,1,2,3,4,5,4]
        denominators = [4,5,5,5,5,6,7]
        
        min_crop_val, min_fact = [np.inf, np.inf], [0,0]
        
        for i in range(2):
            for fraction in zip(numertaors, denominators):
                # crop then scale: (old_size - crop) * reduction factor = new_size
                fact = fraction[0]/fraction[1]
                crop_val = curr_size[i] - (required_size[i]/fact)
                # print(fact, crop_val)
                if crop_val < min_crop_val[i] and crop_val>0: 
                    min_crop_val[i], min_fact[i] = int(crop_val), fraction
        
        # if all factors fail, crop directly
        while np.inf in min_crop_val:
            idx = min_crop_val.index(np.inf)
            min_crop_val[idx] = curr_size[idx] - required_size[idx]
        return min_crop_val, min_fact
    
    def execute(self):
        output_h = [1440, 1080,720,480]
        output_w = [2560, 1920,1280,640]
        
        assert self.new_size[0] in output_h or self.new_size[1] in output_w,\
            "Output size must be one of the following:\n"\
            "- 2560x1440,\n- 1920x1080,\n- 1280x720,\n- 640x480"   

        curr_h, curr_w = self.old_size[0], self.old_size[1]
        # print(self.new_size)
        
        # Check if height and width both can be reduced by half
        down_scale = DownScale(self.img, self.new_size)
        scaled_img = down_scale.img 
        
        while (curr_h%2==0 and curr_h//2>self.new_size[0]) and \
              (curr_w%2==0 and curr_w//2>self.new_size[1]):
            print("reducing image by 1/2\nold_size: {}\nnew_size: {}".format(self.old_size, (curr_h, curr_w)))
            
            down_scale.new_size = (curr_h//2, curr_w//2)
            self.old_size  = (curr_h, curr_w)
            scaled_img     = down_scale.downscale_by_int_factor()        
            curr_h, curr_w = scaled_img.shape[0], scaled_img.shape[1]

        crop_val, red_fact = self.optimal_reduction_factor([curr_h, curr_w], list(self.new_size))
        print("crop_val, red_fact:", crop_val, red_fact)
        
        # Crop img 
        cropped_img = down_scale.crop(scaled_img, crop_val[0], crop_val[1])
        print("cropped img...\nold_size: {}\nnew_size: {}".format((curr_h, curr_w), cropped_img.shape))
        curr_h, curr_w = cropped_img.shape[0], cropped_img.shape[1]
        output_img = cropped_img
        # Resize height then width to the required size.       
        for i in range(2):
            if red_fact[i]==0:  # means that no scaling is required (for both height and width)
                continue
            else:
                # reduction factor = n/d    --->  Upscale the cropped image n times then downscale d times
                upscale_fact = red_fact[i][0] 
                downscale_fact = red_fact[i][1]
                
                print("upscaling {} by: ".format(i), upscale_fact)
                upscale_to_size = (upscale_fact*curr_h, curr_w) if i==0 else (curr_h, upscale_fact*curr_w)
                print(upscale_to_size, curr_h, curr_w)
                upscale = UpScale(output_img, upscale_to_size)
                upscaled_img = upscale.execute()
                print("upsacled to size", upscaled_img.shape)
                print("downscaling {} by: ".format(i), downscale_fact)
                downscale_to_size = (upscaled_img.shape[0]//downscale_fact, curr_w) if i==0 else (curr_h, upscaled_img.shape[1]//downscale_fact)    
                print(downscale_to_size)
                downscale = DownScale(upscaled_img, downscale_to_size)
                print(downscale.img.shape, downscale.old_size)
                downscaled_img = downscale.downscale_by_int_factor()
                curr_h, curr_w = downscaled_img.shape[0], downscaled_img.shape[1]
                output_img = downscaled_img
                print("{} fixed...".format(i), output_img.shape)
        return output_img    
##########################################################################
class UpScale(Scale):
    
    # def __init__(self, method):
    #     super.__init__()
    #     self.method =  method
    
    def scale_nearest_neighbor(self):
        old_height, old_width = self.img.shape[0], self.img.shape[1]
        new_height, new_width = self.new_size[0], self.new_size[1]
        scale_height , scale_width = int(new_height/old_height), int(new_width/old_width)

        print("scale factor",scale_height , scale_width)
        scaled_img = np.zeros((new_height, new_width), dtype = "uint16")

        for y in range(new_height):
            for x in range(new_width):
                # print("x, y: ", x,y, x/scale_width, y/scale_height)
                y_nearest = int(np.floor(y/scale_height))
                x_nearest = int(np.floor(x/scale_width))
                # print("x_nearest, y_nearest: ", x_nearest, y_nearest)
                scaled_img[y,x] = self.img[y_nearest, x_nearest]
        return scaled_img

    def execute(self):
        return self.scale_nearest_neighbor()                     
##########################################################################
class DownScale(Scale):
    
    def crop(self, img, rows_to_crop=0, cols_to_crop=0):
        # old_h, old_w               = img.shape[0], img.shape[1]
        # rows_to_crop, cols_to_crop = old_h%self.new_size[0], old_w%self.new_size[1]
        
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
        """Assumption: new_size is a multiple of old image"""
        
        self.scale_height = self.new_size[0]/self.old_size[0]             
        self.scale_width = self.new_size[1]/self.old_size[1]
        
        assert self.old_size[0]%self.new_size[0]==0 and self.old_size[1]%self.new_size[1]==0, \
            "scale factor is not an integer."
        
        box_height = int(np.ceil(1/self.scale_height)) 
        box_width  = int(np.ceil(1/self.scale_width))
        
        scaled_img = np.zeros(self.new_size, dtype = "float32")

        for y in range(self.new_size[0]):
            for x in range(self.new_size[1]):
                
                y_old = int(np.floor(y/self.scale_height))
                x_old = int(np.floor(x/self.scale_width))
                # print(y_old, x_old)
                y_end = min(y_old + box_height, self.old_size[0])
                x_end = min(x_old + box_width, self.old_size[1])
                # print(y_end, x_end)
                if mode == "max":
                    scaled_img[y,x] = np.amax(self.img[y_old:y_end, x_old:x_end])
                else:     
                    self.img[y_old:y_end, x_old:x_end]
                    scaled_img[y,x] = np.average(self.img[y_old:y_end, x_old:x_end])
        # print(scaled_img)
        return np.round(scaled_img).astype("uint16")    

##########################################################################
# Functions to display RGB image.

def demosaic_raw(img, bayer):

        bpp = 12
        img = np.float32(img) / np.power(2, bpp)
        hs_raw = np.uint8(img*255)
        img = np.uint8(cd.demosaicing_CFA_Bayer_bilinear(hs_raw, bayer))
        return img

def get_color(x, y):
    is_even = lambda x: x%2==0

    if is_even(x) and is_even(y):
        return "R"
    elif not(is_even(x)) and not(is_even(y)):
        return "B"
    else:
        return "G"


def white_balance(img, R_gain, B_gain, G_gain):
    img = img.astype("float32")
    for x in range(img.shape[1]): #col
        for y in range(img.shape[0]):   #row
            color = get_color(y, x)
            if color=="R":
                img[y][x] *= R_gain 
            elif color=="G":
                img[y][x] *= G_gain 
            else:
                img[y][x] *= B_gain
    # img = np.clip(img, 0, 4095)               
    img = ((img/4095)*(4095)).astype("uint16")
    return img 

def gamma(img):
    img = np.float32(img)/255
    img = (img**(1/2.2))*255
    return img.astype("uint8") 