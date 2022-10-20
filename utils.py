from unicodedata import decimal
import numpy as np
import colour_demosaicing as cd

class BiLinear_Scale:
    def __init__(self, img, new_size):
        self.img          = np.float32(img)
        self.old_size     = (img.shape[0], img.shape[1])
        self.new_size     = new_size
        self.scale_height = self.new_size[0]/self.old_size[0]
        self.scale_width  = self.new_size[1]/self.old_size[1]
    
    def horizontal_interpolation(self, y, x):
        """Assumption: y is int value and x is float value."""
        left_pixel  = self.img[y, int(np.floor(x))] 
        right_pixel = self.img[y, int(np.ceil(x))]
        return (right_pixel-left_pixel)*x + left_pixel
    
    def vertical_interpolation(self, y, x):
        """Assumption: x is int value and y is float value."""
        
        top_pixel    = self.img[int(np.floor(y)), x] 
        bottom_pixel = self.img[int(np.ceil(y)) , x]
        return (bottom_pixel-top_pixel)*y + top_pixel 

    def upscale_bilinear(self):
        
        print("scale factor",self.scale_height , self.scale_width)
        scaled_img = np.zeros((self.new_size), dtype = "float32")
        
        # If the new size is assumed to be same as the old one, 
        # each one of the new pixel is added with an increment of increment_h, increment_w 
        increment_h = (self.old_size[0]-1)/(self.new_size[0]-1) 
        increment_w = (self.old_size[1]-1)/(self.new_size[1]-1)

        for y in range(self.new_size[0]): 
            for x in range(self.new_size[1]):

                # index of the pixel w.r.t old width
                proj_y = y*increment_h
                proj_x = x*increment_w 
                
                if proj_y-int(proj_y)==0 and proj_x-int(proj_x)==0:
                    scaled_img[y,x] = self.img[int(proj_y), int(proj_x)]
                
                elif proj_y-int(proj_y)==0:
                    scaled_img[y,x] = self.horizontal_interpolation(int(proj_y), proj_x)
                
                elif proj_x-int(proj_x)==0:
                    scaled_img[y,x] = self.vertical_interpolation(proj_y, int(proj_x))
                
                else:
                    top_pixel    = self.horizontal_interpolation(int(np.floor(proj_y)), proj_x)
                    bottom_pixel = self.horizontal_interpolation(int(np.ceil(proj_y)), proj_x) 
                    scaled_img[y,x] = (bottom_pixel-top_pixel)*proj_y + top_pixel                   
        return np.around(scaled_img).astype("uint16")

##########################################################################
class Downscale:
    def __init__(self, img, new_size):
        self.img = img.astype("float64")
        self.old_size = (self.img.shape[0], self.img.shape[1])
        self.new_size = new_size
        self.scale_height = self.new_size[0]/self.old_size[0]
        self.scale_width  = self.new_size[1]/self.old_size[1]
        self.need_to_crop = (self.new_size[0]%self.old_size[0]!=0 or self.new_size[1]%self.old_size[1]!=0)

    def crop(self, img):
        old_h, old_w               = img.shape[0], img.shape[1]
        rows_to_crop, cols_to_crop = old_h%self.new_size[0], old_w%self.new_size[1]
        
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
        
        if self.need_to_crop:
            cropped_img = self.crop(self.img)
        # print(cropped_img, cropped_img.shape)
        scale_height = self.new_size[0]/cropped_img.shape[0]
        scale_width  = self.new_size[1]/cropped_img.shape[1]
        
        box_height = int(np.ceil(1/scale_height)) 
        box_width  = int(np.ceil(1/scale_width))
        
        scaled_img = np.zeros(self.new_size, dtype = "float32")

        for y in range(self.new_size[0]):
            for x in range(self.new_size[1]):
                
                y_old = int(np.floor(y/scale_height))
                x_old = int(np.floor(x/scale_width))
                # print(y_old, x_old)
                y_end = min(y_old + box_height, self.old_size[0])
                x_end = min(x_old + box_width, self.old_size[1])
                # print(y_end, x_end)
                if mode == "max":
                    scaled_img[y,x] = np.amax(cropped_img[y_old:y_end, x_old:x_end])
                else:     
                    cropped_img[y_old:y_end, x_old:x_end]
                    scaled_img[y,x] = np.average(cropped_img[y_old:y_end, x_old:x_end])
        # print(scaled_img)
        return np.round(scaled_img).astype("uint16")     

                   
##########################################################################    
#   link: https://tech-algorithm.com/articles/bilinear-image-scaling/
    
    def bilinear_formula(self):
        print("scale factor",self.scale_height , self.scale_width)
        scaled_img = np.zeros((self.new_size), dtype = "float32")
        
        # If the new size is assumed to be same as the old one, 
        # each one of the new pixel is added with an increment of increment_h, increment_w 
        increment_h = (self.old_size[0]-1)/(self.new_size[0]-1) 
        increment_w = (self.old_size[1]-1)/(self.new_size[1]-1)

        for y in range(self.new_size[0]): 
            for x in range(self.new_size[1]):

                # index of the pixel w.r.t old width
                proj_y = y*increment_h
                proj_x = x*increment_w 

                A = self.img[int(np.floor(proj_y)), int(np.floor(proj_x))]
                B = self.img[int(np.floor(proj_y)), int(np.ceil(proj_x))]
                C = self.img[int(np.ceil(proj_y)), int(np.floor(proj_x))]
                D = self.img[int(np.ceil(proj_y)), int(np.ceil(proj_x))]

                scaled_img[y,x] = (A*(1-proj_x)*(1-proj_y)) + (B*proj_x*(1-proj_y)) +\
                                  ((C*proj_y)*(1-proj_x)) + (D*proj_x*proj_y)

        return np.around(scaled_img).astype("uint16")                  

##########################################################################

def scale_nearest_neighbor(img, new_size):
    old_height, old_width = img.shape[0], img.shape[1]
    new_height, new_width = new_size[0], new_size[1]
    scale_height , scale_width = new_height/old_height, new_width/old_width

    print("scale factor",scale_height , scale_width)
    scaled_img = np.zeros((new_height, new_width), dtype = "uint16")

    for y in range(new_height):
        for x in range(new_width):
            # print("x, y: ", x,y, x/scale_width, y/scale_height)
            y_nearest = int(np.floor(y/scale_height))
            x_nearest = int(np.floor(x/scale_width))
            # print("x_nearest, y_nearest: ", x_nearest, y_nearest)
            scaled_img[y,x] = img[y_nearest, x_nearest]
    return scaled_img

######################################################################

def scale_nearest_neighbor_v0(img, sacle_fact):
    old_height, old_width = img.shape[0], img.shape[1]
    new_height, new_width = int(old_height*sacle_fact), int(old_width*sacle_fact)

    print("scale factor", sacle_fact)
    scaled_img = np.zeros((new_height, new_width), dtype = "uint16")

    for y in range(new_height):
        for x in range(new_width):
            # print("x, y: ", x,y, x/scale_width, y/scale_height)
            y_nearest = int(np.floor(y/sacle_fact))
            x_nearest = int(np.floor(x/sacle_fact))
            # print("x_nearest, y_nearest: ", x_nearest, y_nearest)
            scaled_img[y,x] = img[y_nearest, x_nearest]
    return scaled_img

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