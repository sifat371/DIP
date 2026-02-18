import cv2
import numpy as np

def show_histogram(image):
    his_img = np.zeros((300, 256, 3), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    his = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(his, his, 0, 300, cv2.NORM_MINMAX)
    
    for x in range(256):
        y = int(his[x][0])
        cv2.line(his_img, (x, 300), (x, 300 - y), (255, 255, 255), 1)
    
    return his_img

def linear1(image):
    r = image / 255.0
    s = 1.0 - r
    s = np.uint8(s * 255)
    return s

def non_linear(image, gamma):
    r = image / 255.0
    c = 1.0
    s = c * np.power(r, gamma)
    s = np.uint8(s * 255)
    return s

def main():
    cam = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()
        
        gamma_img = non_linear(frame, 2.0)
        linear_img = linear1(frame)
        
        row1 = np.hstack((frame, gamma_img, linear_img))
        hist_main = show_histogram(frame)
        hist_gamma = show_histogram(gamma_img)
        hist_linear = show_histogram(linear_img)
        
        row2 = np.hstack((hist_main, hist_gamma, hist_linear))
        row2_resized = cv2.resize(row2, (row1.shape[1], row2.shape[0]))
        
        final_show = np.vstack([row1, row2_resized])
        
        cv2.imshow("Original | Gamma (0.5) | Linear + Histograms", final_show)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()