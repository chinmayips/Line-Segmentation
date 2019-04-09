import cv2
import numpy as np
import statistics
import sys
import os

def preprocessImg(img,imgNo,fileName):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #binary
    ret,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
    cv2.imwrite('out/'+imgNo+'/aftermorph.jpg',opening)
    # sure background area
    kernel = np.ones((3,3),np.uint8)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    cv2.imwrite('out/'+imgNo+'/dilate.jpg',sure_bg)
    return sure_bg

def findConnectedComponents(sure_bg):
    # Marker labelling
    ret, markers,stats,centroid = cv2.connectedComponentsWithStats(sure_bg)
    return stats

def drawLines(stats,img,imgNo,fileName):
    for lineCnt in range(len(stats)):
        lineImg=img[stats[lineCnt][1]:stats[lineCnt][3],stats[lineCnt][0]:stats[lineCnt][2]]
        cv2.imwrite("out/"+imgNo+"/line"+str(lineCnt)+".png",lineImg)

def removeNoise(stat):
    totHt=0
    for  line in stat:
        totHt+=line[3]-line[1]
    avgHt=totHt//len(stat)
    filteredLines=[]
    for line in stat:
        if(line[3]-line[1]<(avgHt/2)):
            continue
        filteredLines.append(line)
    return filteredLines

def convertToList(stat):
    outerList=[]
    for listEle in stat:
        innerList=[]
        for ele in listEle:
            innerList.append(ele)
        outerList.append(innerList)
    return outerList

def ignoreSmallComponents(stat):
    fileredList=[]
    for ele in stat:
        if ele[4] < 400:
            continue
        fileredList.append(ele)
    return fileredList


def getCoordinates(line):
    minXval=line[0]
    minYval=line[1]
    maxHt=line[3]
    maxYval=minYval+maxHt
    return minXval,minYval,maxHt,maxYval


def addCoordinatesToLine(lineStat,singleLineStat,minXval,minYval,maxYval,cols):
    singleLineStat.append(minXval)
    singleLineStat.append(minYval)
    singleLineStat.append(cols)
    singleLineStat.append(maxYval)
    #singleLineStat.append(minYval+maxHt)
    lineStat.append(singleLineStat)
    return lineStat,singleLineStat

def inRange(lineVal,minXval,minYval,lineStat):
    totHt=0
    if len(lineStat)!=0:
        for  line in lineStat:
            totHt+=line[3]-line[1]
        avgHt=totHt//len(lineStat)
        if((minXval-15<lineVal[0] and lineVal[0]< minXval+15) and lineVal[1]>(minYval+avgHt)//2):
            return True
    return False

def notSeen(lineVal,minXval,maxXval):
    if(lineVal[0]>minXval and lineVal[0]<maxXval):
        return False
    return True
def getLineBoxes(stat,cols):
    lineStat=[]
    singleLineStat=[]
    minXval,minYval,maxHt,maxYval=getCoordinates(stat[1])
    maxXval=minXval+stat[1][2]
    for i in range(1,len(stat)):
        if((stat[i][1] > minYval+maxHt) or inRange(stat[i],minXval,minYval,lineStat)):
            lineStat,singleLineStat=addCoordinatesToLine(lineStat,singleLineStat,minXval,minYval,maxYval,cols)
            singleLineStat=[]
            minXval,minYval,maxHt,maxYval=getCoordinates(stat[i])
            maxXval=minXval+stat[i][2]
        else:
            minXval=min(minXval,stat[i][0])
            maxXval=max(maxXval,stat[i][0]+stat[i][2])
            minYval=min(minYval,stat[i][1])
            maxHt=max(maxHt,stat[i][3])
            maxYval=max(maxYval,stat[i][1]+stat[i][3])

    lineStat,singleLineStat=addCoordinatesToLine(lineStat,singleLineStat,minXval,minYval,maxYval,cols)
    return lineStat

# if len(sys.argv)!=2:
#     print("Incorrect args passed!")
#     exit()
# img = cv2.imread(sys.argv[1])
# imgNo = sys.argv[1].split('/')[1].split('.')[0]
#images=range(1,11)
path='/home/swetha/7thsem/ocr-kannada/images/Easy_jpg'
listFiles=os.listdir(path)
imgNo=1
for file in sorted(listFiles):
    imgNo=str(imgNo)
    img = cv2.imread(path+'/'+file)
    rows,cols,ch=img.shape
    fileName=file.split('.')[0]
    opPath='out/'+imgNo
    if not os.path.exists(opPath):
        os.makedirs(opPath)

    sure_bg=preprocessImg(img,imgNo,fileName)
    stats=findConnectedComponents(sure_bg)
    stats=convertToList(stats)
    stats=ignoreSmallComponents(stats)
    #sort based on y and x axis
    stats=sorted(stats, key=lambda stats: (stats[1], stats[0]))
    avgHtOfStats=statistics.mean([ht[3] for ht in stats])
    print(stats)

    for i in range(len(stats)):
        x1=stats[i][0]
        y1=stats[i][1]
        x2=x1+stats[i][2]
        y2=y1+stats[i][3]
        cv2.rectangle(sure_bg,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imwrite('dilate2.png',sure_bg)

    stats=getLineBoxes(stats,cols)
    stats=removeNoise(stats)
    drawLines(stats,img,imgNo,fileName)
    imgNo=int(imgNo)+1