def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist