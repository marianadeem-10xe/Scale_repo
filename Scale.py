from multiprocessing.spawn import old_main_modules
from utils import scale_nearest_neighbor, Downscale, optimal_reduction_factor


def Scale(img, new_size):
    output_h = [1440, 1080,720,480]
    output_w = [2560, 1920,1280,640]
    
    assert new_size[0] in output_h or new_size[1] in output_w

    old_h, old_w = img.shape[0], img.shape[1]
    
    down_scale   = Downscale(img, new_size)
    while (old_h%2==0 and old_h//2>new_size[0]) and (old_w%2==0 and old_w//2>new_size[1]):
        down_scale.new_size = (old_h//2, old_w//2)
        scaled_img   = down_scale.downscale_by_int_factor()        
        old_h, old_w = scaled_img.shape[0], scaled_img.shape[1]

    crop_val, red_fact     = optimal_reduction_factor([old_h, old_w], list(new_size))
  
    # Crop img 
    cropped_img = down_scale.crop(scaled_img, crop_val[0], crop_val[1])    # crop with crop val and the reduce using the red fact
         
    for i in range(2):
        if red_fact[i]==0:  # means that no scaling is required
            return cropped_img
        else:
            # reduction factor = n/d    --->  Upscale the cropped image n times then downscale d times
            upscale_fact = red_fact[i][0] 
            upscaled_img = scale_nearest_neighbor(cropped_img, upscale_fact*old_h, upscale_fact*old_w)
            downscale_fact = red_fact[i][1]
            down_scale.img, down_scale.new_size = upscaled_img, (downscale_fact*upscaled_img[0], downscale_fact*upscaled_img[1])
            downscaled_img = down_scale.downscale_by_int_factor()
    
    return downscaled_img








