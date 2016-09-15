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

# ----------------
# MODEL DATA - RANDOM FOREST
# ----------------

Pkg.add("DecisionTree")

using DecisionTree

# Train (Build) Random Forest Classifier
# Method: build_forest{T<:Float64,U<:Real}(::Array{T<:Float64,1}, ::Array{U<:Real,2}, ::Integer, ::Integer, ::Any)
yTrain = float(yTrain)
xTrain = float(xTrain)

model = build_forest(yTrain, xTrain, 20, 50, 1.0)
# Number of Chosen Features: 20
# Number of Trees: 50
# Ratio of Sub-Sampling: 1.0

# Ensemble of Decision Trees
# Trees: 50
# Avg Leaves: 3164.84
# Avg Depth: 37.98

# Predict the test dataset
predict = apply_forest(model, xTest)
labelsInfoTest[:Class] = map(char, predict)

# Cross Validation
accuracy = nfoldCV_forest(yTrain, xTrain, 20, 50, 4, 1.0); # n = 4 fold

# ----------------
# MODEL DATA - K-NEAREST NEIGHBOR
# ----------------

# Transpose dataset so columns represent data points and rows represent features.
xTrain = xTrain'
xTest = xTest'

function euclidean_distance(a, b)
	return dot(a-b, a-b)
end

# ----------------
# OUTPUT DATA
# ----------------

# Random Forest Model Output
writetable("$(path)/submission_rfc.csv", labelsInfoTest, separator=',', header=True)

# K-Nearest Neighbor Output