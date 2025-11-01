import numpy as np

def Img_Norm(img: np.array):
    """ 
    Normalization of Image pixel values between [0, 1]
    """
    min = np.min(img)
    max = np.max(img)
    return (img - min) / (max - min)

def Batch_Img_Norm(imgs: np.array):
    """ 
    Normalization of Image pixel values between [0, 1] for a batch of images
    image size = (batch, height, width, channels)
    min / max : (batch, 1, 1, 1)
    """
    min = np.min(imgs, axis=(1, 2, 3), keepdims=True)
    max = np.max(imgs, axis=(1, 2, 3), keepdims=True)
    return (imgs - min) / (max - min)


if __name__ == "__main__":
    img = np.array([1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9], dtype=float)
    print(Img_Norm(img))

    # if we have a batch of images then ? we use the batch normalization
    