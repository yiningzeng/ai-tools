# coding:utf-8
import cv2
import numpy as np

if __name__ == "__main__":
    img_path = "/home/baymin/daily-work/new-work/素材/yunsheng_date/gray/1562124994_359.jpg"
    img = cv2.imread(img_path)
    #获取图片的宽和高
    width,height = img.shape[:2][::-1]
    #将图片缩小便于显示观看
    img_resize = cv2.resize(img,
    (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("img",img_resize)
    # print("img_reisze shape:{}".format(np.shape(img_resize)))

    #将图片转为灰度图
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("img_gray",img_gray)
    cv2.imwrite("/home/baymin/daily-work/new-work/素材/yunsheng_date/gray/a4.jpg", img_gray)
    # print("img_gray shape:{}".format(np.shape(img_gray)))
    # cv2.waitKey()
