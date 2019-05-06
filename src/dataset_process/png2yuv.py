import cv2,os
import numpy as np
path='./DIV2K'
for i,files in enumerate(os.listdir(path)):
    if i<=100:
        img=cv2.imread(os.path.join(path,files))
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        height,width,c=img_yuv.shape
        uv_width = width // 2
        uv_height = height // 2

        Y = np.zeros((height, width), np.uint8, 'C')
        U = np.zeros((uv_height, uv_width), np.uint8, 'C')
        V = np.zeros((uv_height, uv_width), np.uint8, 'C')
        
        Y=img_yuv[:,:,0]
        U=img_yuv[:,:,1]
        V=img_yuv[:,:,2]
        fp = open('./DIV2K_YUV/{}_{}x{}_60.yuv'.format(files.split('.')[0],width,height), 'wb')
        file1='{}_{}x{}_60.yuv'.format(files.split('.')[0],width,height)
        for m in range(height):
            for n in range(width):
                fp.write(Y[ m, n])
        for m in range(uv_height):
            for n in range(uv_width):
                fp.write(U[ 2*m, 2*n])
        for m in range(uv_height):
            for n in range(uv_width):
                fp.write(V[ 2*m, 2*n])
        fp.close()
        content='#======== File I/O ===============\nInputFile                     : {}\nInputBitDepth                 : 8           # Input bitdepth\nFrameRate                     : 30          # Frame Rate per second\nFrameSkip                     : 0          # Number of frames to be skipped in input\nSourceWidth                   : {}         # Input  frame width\nSourceHeight                  : {}         # Input  frame height\nFramesToBeEncoded             : 1        # Number of frames to be coded\n\nLevel                         : 3.1'.format(file1,width,height)
        f=open('./DIV2K_CFG/{}_{}x{}_60.cfg'.format(files.split('.')[0],width,height), "w")
        f.write(content)
        f.close()
        print(file1)