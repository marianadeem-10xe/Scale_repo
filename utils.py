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
    
    def horizontal_interpolation(self, x1, x2, y1, weight):
        if x1==x2:
            return self.img[y1,x1]
        left_pixel  = self.img[y1,x1]
        right_pixel = self.img[y1,x2]
        return (weight)*left_pixel + (1-weight)*right_pixel
    
    """def vertical_interpolation(self, x1, x2, y1, y2, weight):
        
        top_pixel    = self.img[] 
        bottom_pixel = self.img[int(np.ceil(y)) , x]
        return (1-y)*top_pixel + (y)*bottom_pixel""" 
    
    def scale_bilinear(self):
        
        if self.new_size==self.old_size: return self.img

        print("scale factor",self.scale_height , self.scale_width)
        scaled_img = np.zeros((self.new_size), dtype = "float32")
        
        for y in range(self.new_size[0]):
            for x in range(self.new_size[1]):
                
                # print("y, x: ", y/self.scale_height, x/self.scale_width)
                proj_y, proj_x = y/self.scale_height, x/self.scale_width

                y1 = min(int(np.floor(proj_y)), self.old_size[0]-1) 
                x1 = min(int(np.floor(proj_x)), self.old_size[1]-1) 
                y2 = min(int(np.ceil(proj_y)), self.old_size[0]-1)
                x2 = min(int(np.ceil(proj_x)), self.old_size[1]-1)
                
                # Get the four neighbouring pixels
                # top_left     = self.img[y1,x1]
                # top_right    = self.img[y1,x2]
                # bottom_left  = self.img[y2,x1]
                # bottom_right = self.img[y2,x2]
                
                weight_h = np.ceil(proj_x) - proj_x
                P1 = self.horizontal_interpolation(x1, x2, y1, weight_h)
                P2 = self.horizontal_interpolation(x1, x2, y2, weight_h)

                weight_v = np.ceil(proj_y)-proj_y
                scaled_img[y,x] = (1-weight_v)*P1 + (weight_v)*P2
                # print(weight_h, weight_v)
        return np.around(scaled_img).astype("uint16")

    def box_downsample(self):
        box_width  = int(np.ceil(1/self.scale_width))
        box_height = int(np.ceil(1/self.scale_height))
        scaled_image = np.zeros((self.new_size), dtype="float32")
        for y in range(self.new_size[0]):
            for x in range(self.new_size[1]):
                # Coordinates in old image
                x_ = int(np.floor(x/self.scale_width))
                y_ = int(np.floor(y/self.scale_height))
                
                # min() is used to assure that coordinates aren't out of bounds
                x_end = min(x_ + box_width, self.old_size[0]-1)
                y_end = min(y_ + box_height, self.old_size[1]-1)

                # We average the colors in the box
                pixel = self.img[y_:y_end,x_:x_end].mean()
                
                # We convert results to a tuple of ints
                pixel = np.round(pixel).astype(int)
                
                scaled_image[y,x] = pixel 
        return scaled_image.astype("uint16")         
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