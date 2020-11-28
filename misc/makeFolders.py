import os
import shutil

root = "../../../SwimData/GeoCodes"

filliste = os.listdir(os.path.join(root,"allGeoCodes"))

#for at lave selve mapperne
# for fil in filliste:
#     sti = os.path.join(root,"classifier4","artTrain",fil[:3])
#     os.mkdir(sti)
    
#for at få noget kunstig data ned i mapperne

for fil in filliste:
    position = os.path.join(root,"allGeoCodes",fil)
    destination = os.path.join(root,"classifier4","artTrain",fil[:3])
    shutil.copy(position,destination)
