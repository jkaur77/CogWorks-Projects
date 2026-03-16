import pickle
import os
import numpy as np
import skimage.io as io


def vectorize_photo(path) :
    image = io.imread(str(path))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB
    
    return image


def proc(Person: str) :
    filepaths = []
    
    person_path = "./data/" + Person
    
    filelist = os.listdir(person_path)
    for i in filelist:
        if i.endswith(".jpg" or ".png"):  # You could also add "and i.startswith('f')
            filepaths.append(person_path + "/" + i)
    


    vectorized_images = []

    for photo_path in filepaths :
        image = vectorize_photo(photo_path)
        vectorized_images.append(image)

    save_path = Person + ".npy"

    np.save(save_path, vectorized_images)

