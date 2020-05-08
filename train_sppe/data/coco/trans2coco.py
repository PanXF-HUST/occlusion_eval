# import json
# fp = open('keypoints.json','r')
# dict = json.load(fp)
# print(dict.keys())
# images = dict['images']
# img_name=[]
# for i in range (len(images)):
#     img_name.append(images[i]['file_name'])
# print(img_name)

from pycocotools.coco import COCO
import numpy as np
#import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import h5py
from PIL import Image
from PIL import ImageDraw
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# initialize COCO api for person keypoints annotations
# dataDir = '/home/myubuntu/Desktop/human-pose-estimation.pytorch-master/data/coco'
# dataType = 'person_keypoints_train2017'         #'person_keypoints_val2017'
# annFile = '{}/annotations/{}.json'.format(dataDir,dataType)
annFile = 'person_keypoints_train2017.json'
coco_kps=COCO(annFile)

# display COCO categories and supercategories
cats = coco_kps.loadCats(coco_kps.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco_kps.getCatIds(catNms=['person'])
imgIds = coco_kps.getImgIds(catIds=catIds )
print ('there are %d images containing human'%len(imgIds))
#print (imgIds)
def getBndboxKeypointsGT():

    '''
    firstRow = ['imagename','bndbox','nose',
            'left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
            'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
            'left_knee','right_knee','left_ankle','right_ankle']
        keypointsWriter.writerow(firstRow)'''
    h5_imgname = []
    h5_bndbox = []
    h5_keypoints = []
    for i in range(len(imgIds)):
        imageNameTemp = coco_kps.loadImgs(imgIds[i])[0]
        imageName = imageNameTemp['file_name'].encode('raw_unicode_escape')
        img = coco_kps.loadImgs(imgIds[i])[0]
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco_kps.loadAnns(annIds)
        #print(anns)
        personNumber = len(anns)
        #print(personNumber)
        #np.fromstring(imageName, dtype=np.uint8).astype('float64')
        #print (imageName)
        #imageName=imageName.tolist()

        for i in range(personNumber):
            i=str(anns[i]['image_id'])
            #print(i)
            while len(i)<12:
                i='0'+i
            i=i+'.jpg'
            temp3 = []
            for h in range(len(i)):
                jiji=ord(i[h])
                #print(jiji)
                temp3.append(jiji)
            #print(temp3)
            temp3 = np.array(temp3)
            temp3 = temp3.astype(np.float64)
            #print (temp3)
            h5_imgname.append(temp3)

    #print(h5_imgname)
        for j in range(personNumber):
            temp = []
            b=anns[j]['bbox']
            b[2] = int(b[0]+b[2])
            b[3] = int(b[1]+b[3])
            b[0] = int(b[0])
            b[1] = int(b[1])
            #print(b)
            temp.append(b)
            h5_bndbox.append(temp)
            #print(h5_bndbox)

             #h5_keyPoints.append(anns[j]['keypoints'])
            temp1 = []
             #print(anns[j]['keypoints'])
            for k in range (len(anns[j]['keypoints'])):

                if(k%3==0):
                    temp2=[]
                if (k % 3 != 2):
                    temp2.append(anns[j]['keypoints'][k])
                if (k % 3 == 2):
                    temp1.append(temp2)
                #print(temp1)
            h5_keypoints.append(temp1)

    h5_imgname = np.array(h5_imgname)
    h5_bndbox = np.array(h5_bndbox)
    h5_keypoints = np.array(h5_keypoints)
    #print(h5_imgname)
    h5file = h5py.File('annot_coco2017.h5', 'w')
    h5file.create_dataset('imgname', data=h5_imgname)
    h5file.create_dataset('bndbox',data=h5_bndbox)
    h5file.create_dataset('part',data=h5_keypoints)


    h5file.close()

if __name__ == "__main__":
    print ('Writing bndbox and keypoints to h5 files..."')
    getBndboxKeypointsGT()

