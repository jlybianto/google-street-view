# ----------------
# IMPORT PACKAGES
# ----------------

Pkg.add("Images")
Pkg.add("DataFrames")

using Images
using DataFrames

# ----------------
# PROFILE DATA
# ----------------

# Resized images are 20 x 20 pixels
imageSize = 400

# Set path for location of files
path = "C:\\Users\\Jesse Lybianto\\Documents\\Kaggle\\First Steps With Julia"

# Obtain datasets
labelsInfoTrain = readtable("$(path)/trainLabels.csv")
labelsInfoTest = readtable("$(path)/sampleSubmission.csv")

function read_data(typeData, labelsInfo, imageSize, path)
	x = zeros(size(labelsInfo, 1), imageSize)
	for (index, idImage) in enumerate(labelsInfo[:ID])
		nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
		img = load(nameFile)

		temp = convert(Image{Images.Gray}, img)

		if ndims(temp) == 3
			temp = mean(temp.data, 1)
		end

		x[index, :] = reshape(temp, 1, imageSize)
	end

xTrain = read_data("train", labelsInfoTrain, imageSize, path)
xTest = read_data("test", labelsInfoTest, imageSize, path)

# Obtain the first character of a string
yTrain = map(x -> x[1], labelsInfoTrain[:Class])
yTrain = int(yTrain)