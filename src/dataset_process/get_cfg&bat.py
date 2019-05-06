import os,sys
files = os.listdir('./DIV2K_YUV')
qpl=[27,32,38,45]
#get cfg
'''
for num,file1 in enumerate(files):
    print(num,file1)
    width, height = [int(l) for l in file1.split('_')[1].split('x')]
    content='#======== File I/O ===============\nInputFile                     : {}\nInputBitDepth                 : 8           # Input bitdepth\nFrameRate                     : 30          # Frame Rate per second\nFrameSkip                     : 10           # Number of frames to be skipped in input\nSourceWidth                   : {}         # Input  frame width\nSourceHeight                  : {}         # Input  frame height\nFramesToBeEncoded             : 33         # Number of frames to be coded\n\nLevel                         : 3.1'.format(file1,width,height)
    f=open('E:/DATA/{}.cfg'.format(file1.split('.')[0]), "w")
    f.write(content)

'''
content=''
#get bat
for file1 in files:
    cfg=file1.split('.')[0]+'.cfg'
    yuv=file1.split('.')[0]+'.yuv'
    for qp in qpl:
        content+='TAppEncoder.exe -c encoder_intra_main.cfg -c {} -o ./{}/{} -q {} > result_0_{}.txt\n'.format(cfg,qp,yuv,qp,qp)
f=open('./DIV2K_100.bat', "w")
f.write(content)
f.close()