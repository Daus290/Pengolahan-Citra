import cv2
import numpy as np

def apply_edge_detection(image_path, threshold1=100, threshold2=200): # default slider threshold (untuk canny)
    img = cv2.imread(image_path, 0)

    # Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)

    # Prewitt
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    prewittx = cv2.filter2D(img, -1, kernelx)
    prewitty = cv2.filter2D(img, -1, kernely)
    prewitt = cv2.add(prewittx, prewitty)
    prewitt = cv2.convertScaleAbs(prewitt)

    # Roberts
    robertsx = cv2.filter2D(img, -1, np.array([[1, 0], [0, -1]]))
    robertsy = cv2.filter2D(img, -1, np.array([[0, 1], [-1, 0]]))
    roberts = cv2.add(robertsx, robertsy)
    roberts = cv2.convertScaleAbs(roberts)

    # Canny
    canny = cv2.Canny(img, threshold1, threshold2) # pass argumen dari fungsi threshold default tadi

    return {
        'sobel': sobel,
        'prewitt': prewitt,
        'roberts': roberts,
        'canny': canny
    }
