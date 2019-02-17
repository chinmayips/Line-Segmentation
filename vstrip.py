import cv2
import numpy as np
#import image
imp=[2,3,4,5,6,7,8]
for im1 in imp:
    image = cv2.imread('images/'+str(im1)+'.jpg')
    im = cv2.imread('images/'+str(im1)+'.jpg')
    #image=cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    #cv2.imshow('orig',image)
    #cv2.waitKey(0)

    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    cv2.fastNlMeansDenoising(gray,gray)


    #binary
    ret,image = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    #cv2.imshow('second',image)
    #cv2.waitKey(0)

    #image = image.astype('float32')
    #image = image / 255

    #strips
    (rows,cols)=image.shape
    strips=[]

    #200 is width of strip
    s_width = 100
    s_cnt = 5
    #min gap between two lines
    space_thres  = 10
    #max no of pixel vals in a line to qualify as empty line
    count_thres = 5
    #BLOCK A
    for i in range(0,cols,s_width):
        if(i+s_width<cols): #Not end of the image
            strips.append(image[0:rows,i:i+s_width])
        else: #For the end of the image
            strips.append(image[0:rows,i:cols])


    gaps_arr = []

    count_arr = [] #Array of count_strips

    final_px = [] #Array of arrays to store the row values
    #BLOCK b
    for simg in strips:
        (rows1,cols1)=simg.shape
        x=[]
        count_strip = [] # Count the white px for each row of a strip
        for i in range(1,rows1-1):
            count_w = 0
            nxt_cnt_w = 0
            prev_cnt_w= 0
            for j in range(cols1):
                vals = simg[i][j]
                nxt_vals = simg[i+1][j]
                prev_vals = simg[i-1][j]
                if(vals == 255): #Look for white pixels
                    count_w = count_w +1
                if(nxt_vals == 255):
                    nxt_cnt_w = nxt_cnt_w +1 
                if(prev_vals ==255): #Look for white pixels
                    prev_cnt_w = prev_cnt_w +1
            #print(count_w)
            count_strip.append(count_w)
            if count_w<=s_cnt and (nxt_cnt_w>s_cnt or prev_cnt_w >s_cnt):
                '''if(len(x)!=0):
                    if(x[len(x)-1]==i-1):
                        x.pop()'''
                x.append(i)      	
        count_arr.append(count_strip)
        gaps_arr.append(x)

    mid_arr = []

    #gaps_arr and mid_arr are array of arrays 
    #gaps_arr has the gap of eact strip - gaps of each strip is an array
    #BLOCK c
    for gfs in gaps_arr:
        y=[]
        for gap_i in range(0,len(gfs)-1):
            space_diff = gfs[gap_i+1] - gfs[gap_i]
            if(space_diff>space_thres):
                mid_pt = gfs[gap_i] + space_diff//2
                y.append(mid_pt)
        mid_arr.append(y)

    print(gaps_arr[1])
    print(mid_arr[1])


    # j is indiv strip no
    strips_no = len(mid_arr)
    #BLOCK d
    for j in range(len(mid_arr)):
        z = []
        for i in range(len(mid_arr[j])):
            if(count_arr[j][mid_arr[j][i]]<=count_thres):
                z.append(mid_arr[j][i])
                cv2.line(image,(s_width*j,mid_arr[j][i]),(s_width*(j+1),mid_arr[j][i]),(255,0,0),thickness=1)
        if(len(z)!=0):
            final_px.append(z)

    cv2.imwrite('op/'+str(im1)+'.jpg',image)
    no_of_lines=min([len(x) for x in final_px])
'''
#no_of_lines=len(final_px[0])
final_lines = []
n_img =[ ]

print(no_of_lines)

(rows,cols)=image.shape

j=0
#for i in range(0,mid_arr[j][0]):
#print(final_px[0][0])
prev_final_px=0

for n_line in range(0,no_of_lines-1):
    s=0
    n_img=[]
    for j in range(1, strips_no-1):
        if(s+s_width<cols):
            temp_img = image[prev_final_px:final_px[j][n_line] , s:s+s_width]
            s = s+s_width
        else:
            temp_img = image[prev_final_px:final_px[j][n_line] , s:cols ]
       
        if(n_img==[]):
            n_img=temp_img
        else:
            temp=[]
            for i in range(prev_final_px,final_px[j][n_line]):
                if(i<len(n_img) and i<len(temp_img)):
                    temp.append(np.concatenate((n_img[i] , temp_img[i])))
                else:
                    break
            n_img=np.array(temp)
        prev_final_px=final_px[j][n_line]
        #print(n_img)
        #print('-------------------------------')
    nnn_img = np.array(n_img)
    cv2.imwrite('op/fl.jpg' ,nnn_img)
    if(n_line==0):
        break
   ''' 


