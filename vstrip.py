import cv2
import numpy as np
import statistics
import sys
from PIL import Image
import pickle

#100 is width of strip
s_width = 100
s_cnt = 5
#min gap between two lines
space_thres  = 10
#max no of pixel vals in a line to qualify as empty line
count_thres = 5

def preprocessImage(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.fastNlMeansDenoising(gray,gray)
    #binary
    ret,image = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return image



def calcThresholds():
    s_width = 100 #200 is width of strip
    s_cnt = 5 
    space_thres  = 10  #min gap between two lines 
    count_thres = 5 #max no of pixel vals in a line(per strip) to qualify as empty line


def obtainStrips(image):
    (rows,cols)=image.shape
    strips=[]
    for i in range(0,cols,s_width):
        if(i+s_width<cols): #Not end of the image
            strips.append(image[0:rows,i:i+s_width])
        else: #For the end of the image
            strips.append(image[0:rows,i:cols])
    return strips



def segmentStrips(strips):
    gaps_arr = []
    count_arr = [] #Array of count_strips

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
    #gaps_arr has the gap of each strip - gaps of each strip is an array
    for gfs in gaps_arr:
        y=[]
        for gap_i in range(0,len(gfs)-1):
            space_diff = gfs[gap_i+1] - gfs[gap_i]
            if(space_diff>space_thres):
                mid_pt = gfs[gap_i] + space_diff//2
                y.append(mid_pt)
        mid_arr.append(y)

    final_px = []
    # j is indiv strip no
    #After finding the mid point of each line we eliminate the unnecesary ones.
    for j in range(len(mid_arr)):
        z = []
        for i in range(len(mid_arr[j])):
            if(count_arr[j][mid_arr[j][i]]<=count_thres):
                z.append(mid_arr[j][i])
                cv2.line(image,(s_width*j,mid_arr[j][i]),(s_width*(j+1),mid_arr[j][i]),(255,0,0),thickness=1)
        if(len(z)!=0):
            final_px.append(z)
    return final_px , mid_arr


def GetLinesPxls(final_px): #Cleaning the final_pxls array 
    line_px = []
    medians = [0]
    max_idx = max([len(x) for x in final_px])
    for indx in range(max_idx):
        new_line = []
        for strip in final_px:
            if(len(strip)>indx): 
                new_line.append(strip[indx])
        med = statistics.median(new_line)
        med = int(med)
        medians.append(med)
        print(med , len(new_line))
        for nl in range(len(new_line)-1):
            # if(new_line[nl]>med+50 or new_line[nl]<med-50):
            #     new_line[nl] = 0
            # el
            if(new_line[nl]>med+30 or new_line[nl]<med-30):
                if(nl==0): #First Strip 
                    new_line[nl] = new_line[nl+1]
                elif(nl==len(new_line)-1): #Last Strip
                    new_line[nl] = med
                else:
                    new_line[nl] = (new_line[nl+1]+new_line[nl-1])/2
        line_px.append(new_line)

    #Find avg and sd dev (Median will work best +- A threshold value) for each line and if more over sd dev then remove 
    #And Combine strips even if it is there in only 1 strip, maybe there is only 1 word in that line. 
    for j in line_px:
        print(j)
    return line_px , medians

def calcThresholds(strips):
    mid=len(strips)//2
    (rows1,cols1)=strips[mid].shape
    strip_mid=strips[mid]
    #print(strip_mid)
    tot_count_w = 0
    count_whitepx=[]
    for i in range(1,rows1-1):
        count_w = 0
        for j in range(cols1):
            vals = strip_mid[i][j]
            if(vals == 255): #Look for white pixels
                count_w = count_w +1
        tot_count_w+=count_w
        count_whitepx.append(count_w)
    #max no of pixel vals in a line to qualify as empty line
    count_thres=tot_count_w//rows1
    #200 is width of strip
    s_width = 100
    seen=False
    min_space=0
    min_spaces=[]

    for i in range(1,rows1-1):
        if(count_whitepx[i-1]<=count_thres):
            if(not seen):
                seen=True
                min_space=0
            else:
                min_space+=1

        else:
            if(min_space!=0):
                min_spaces.append(min_space)
                min_space=0
                seen=False

    min_spaces.sort()
    space_thres=min_spaces[round(0.5*len(min_spaces))]//2

    s_cnt = count_thres
    #min gap between two lines
    #space_thres  = 10
    return s_width,space_thres,count_thres

def combineStrips(line_px , medians , image):
    (rows,cols)=image.shape
    print(medians)
    for l_no in range(len(line_px)):
        images_in_line = []
        prev_start_px = 0 if l_no==0 else int(line_px[l_no-1][j])
        for j in range(len(line_px[l_no])):
            s = j*s_width
            if(s+s_width<cols):
                temp_img = image[medians[l_no]:int(line_px[l_no][j]) , s:s+s_width]
            else:
                temp_img = image[medians[l_no]:int(line_px[l_no][j])  , s:cols ]
            images_in_line.append(temp_img)
        
        images_opened = []
        for a in range(len(images_in_line)):
            im_a = np.array(images_in_line[a])
            # print("Image is " , im_a , images_in_line[a])
            # if(len(im_a)==0):
            #     break
            if(len(im_a)!=0):
                cv2.imwrite('op/line'+ str(l_no) + "_" +str(a)+'.jpg', im_a)
                images_opened.append(Image.open('op/line'+ str(l_no) + "_" + str(a)+'.jpg'))
            else:
                temp_im = Image.new('RGB', (100,2))
                images_opened.append(temp_im)

        # print(images_opened)
        widths, heights = zip(*(i.size for i in images_opened))
        total_width = sum(widths) 
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        strip_cnt = 0
        for im in images_opened:
            # np_im = np.array(im)
            nxt_x_offset = x_offset + im.size[0]
            box = (x_offset, 0 , nxt_x_offset ,im.size[1])
            # print(np_im.shape , box)
            new_im.paste(im, box)
            x_offset = nxt_x_offset
            strip_cnt  = strip_cnt+1

        # print("Image is:" , new_im)
        ni = np.array(new_im)
        print(ni.shape , l_no , total_width , heights)
        if(len(ni.shape)== 3):
            new_im.save('op/test/'+img_str+'_l'+str(l_no)+'.jpg')

if __name__ == "__main__":
    img_str = "74D2"
    image = cv2.imread('images/'+img_str+'.jpg')
    im = cv2.imread('images/'+img_str+'.jpg')
    image = preprocessImage(image)
    im = preprocessImage(im)
    strips = obtainStrips(image)
    s_width,space_thres,count_thres=calcThresholds(strips)
    final_px , mid_arr = segmentStrips(strips)

    # pickle_out = open("One.pickle","wb")
    # pickle.dump(final_px, pickle_out)
    # pickle_out.close()
    
    # pickle_in = open("One.pickle","rb")
    # final_px = pickle.load(pickle_in)

    line_px , medians = GetLinesPxls(final_px)

    for j in range(len(line_px)):
            for i in range(len(line_px[j])):
                l_px = int(line_px[j][i])
                cv2.line(im,(s_width*i,l_px),(s_width*(i+1),l_px),(255,0,0),thickness=1)


    combineStrips(line_px , medians , im)
    cv2.imwrite('op/'+img_str+'.jpg',im)