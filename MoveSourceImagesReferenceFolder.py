import os
import shutil
import numpy as np


sourceFile = "HandExpansionCoordinates.txt"
sourceDirectory = "Hands"
rootDirectory = "/Users/chekumis/Desktop/PalmarBBGtest/"
inputrootDirectory = "/Users/chekumis/Desktop/Palmar/"
targetDirectory = "/Users/chekumis/Desktop/PalmarBBGtest/OriginalImages/"
fileNameIndex = 0
imageList = []


print("Opening file")
inputFile = open(rootDirectory + sourceFile, 'r')
fileLines = inputFile.readlines()

for line in fileLines:

    fileName = line.split(',')[0]
    imageList.append(fileName)


for image in imageList:
    shutil.copy(inputrootDirectory+image, targetDirectory)
    print("Copy file",image)

