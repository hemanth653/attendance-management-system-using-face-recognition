from os import listdir
import cv2
import numpy as np

root_dir="./dataset"
features=[]
labels=[]
i=0
for subfolder in listdir(root_dir):
    # print(subfolder)
    path=f"{root_dir}/{subfolder}"
    print(f"--------{path}------------")
    for file in listdir(path):
        # print(file)
        filepath=f"{path}/{file}"
        image=cv2.imread(filepath,0)
        images=cv2.resize(image,(350,350))

        features.append(images)
        labels.append(i)

        # print(image)
        # cv.imshow("demo",image)
        # cv.waitKey()
    i=i+1

print(f"the number of features = {len(features)} ")
print(f"the number of labels = {len(labels)} ")

print(features,labels)

recong=cv2.face.LBPHFaceRecognizer_create()

recong.train(features,np.array(labels))
recong.save("facemodel.yml")