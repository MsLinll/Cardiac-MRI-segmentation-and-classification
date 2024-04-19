import os

def rename(path):
    for file in os.listdir(path):
        newName = file.replace("_MYO","")
        os.rename(os.path.join(path,file),os.path.join(path,newName))

path = "E:/finalproject/ACDC_dataset/ACDC_MYO/ACDC_2/Training_Labels"
files = os.listdir(path)
rename(path)
print("successfully!")