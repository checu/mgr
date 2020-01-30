import os
import shutil

sourceFile = "HandInfo.txt"
sourceDirectory = "Hands"
rootDirectory = "/Users/chekumis/Desktop/"
fileNameIndex = 7

print("Opening file")
inputFile = open(rootDirectory + sourceFile, 'r')
fileLines = inputFile.readlines()
fileHeader = fileLines[0]

print("Filtering data")
outputData = []

for lineNumber in range(1, len(fileLines)):
    if "palmar" in fileLines[lineNumber]:
        outputData.append(fileLines[lineNumber])

outputDirectory = rootDirectory + "Palmar"

print("Creating directory")
if not os.path.exists(outputDirectory):
    os.mkdir(outputDirectory)
    print("Directory ", outputDirectory, " Created ")
else:
    print("Directory ", outputDirectory, " already exists")

print("Saving file + copy file")
outputFile = open(outputDirectory + "/" + sourceFile, "w+")
outputFile.write(fileHeader)

source = rootDirectory + sourceDirectory + "/"
destination = outputDirectory + "/"

for line in outputData:
    image = line.split(',')[fileNameIndex]
    if os.path.isfile(source + image):
        shutil.copyfile(source + image, destination + image)
        outputFile.write(line)

outputFile.close()
print("End")
