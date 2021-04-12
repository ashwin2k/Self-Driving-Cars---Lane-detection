def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

def get_steer(s,r)
# r = s / (sqrt(2 - 2 * cos(2*a/n))