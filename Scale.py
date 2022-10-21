from multiprocessing.spawn import old_main_modules
from utils import scale_nearest_neighbor, Downscale, optimal_reduction_factor


def Scale(img, new_size):
    output_h = [1440, 1080,720,480]
    output_w = [2560, 1920,1280,640]

    old_h, old_w = img.shape[0], img.shape[1]
    assert new_size[0] in output_h or new_size[1] in output_w

    while (old_h%2==0 and old_h//2<new_size[0]) and (old_w%2==0 and old_w//2<new_size[1]):
        down_scale   = Downscale(img, (new_size[0]//2, new_size[1]//2))
        scaled_img   = down_scale.downscale_by_int_factor()        
        old_h, old_w = scaled_img.shape[0], scaled_img.shape[1]

    crop_val, red_fact     = optimal_reduction_factor([old_h, old_w], list(new_size))
    crop_h, crop_h         = crop_val[0], crop_val[1] 
    red_fact_h, red_fact_w = red_fact[0], red_fact[1]
    
    for i in range(2):
        pass    # crop with crop val and the reduce using the red fact