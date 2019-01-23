import cv2


def deal_rects(rects, rate=0.8):
    rects = sorted(rects, key=lambda item: item[0])
    max_h = 0
    for rect in rects:
        _, _, _, h = rect
        if h > max_h:
            max_h = h

    r_y = 0
    for rect in rects:
        x, y, w, h = rect
        if h > (max_h * rate):
            r_x = x + w

    return r_x


def convert(imgpath):
    img = cv2.imread(imgpath)
    gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 25)
    img2, ctrs, hier = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    x = deal_rects(rects)
    gray = gray[:, :x]
    cv2.imwrite(imgpath, gray)
