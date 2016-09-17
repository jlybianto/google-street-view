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
# Iteration from one image to the next is an iteration from column to column.
# Iteration over columns is faster in Julia than iteration over rows.
xTrain = xTrain'
xTest = xTest'

# Loops are much slower in languages such as Python, R, and MATLAB.
# The opposite is true for Julia: FOR loops can be faster than vectorized operations.
function euclidean_distance(a, b)
	distance = 0.0
	for index in 1:size(a, 1)
		distance += (a[index]-b[index]) * (a[index]-b[index])
	end
	return distance
end

# Function to find the k-nearest neighbor of the i-th data point.
function get_k_nearest_neighbors(x, i, k)
	nRows, nCols = size(x)

	# Initialize a vector imageI so that it is accessed only once from the X matrix.
	# Filling an empty vector with each element is faster than copying entire vector.
	# Create empty array of nRows elements of type Float32 (decimal).
	imageI = Array(Float32, nRows)
	imageJ = Array(Float32, nRows)

	# Initialize an empty vecotr that will contain the distances between the i-th point.
	distances = Array(Float32, nCols)

	for index in 1:nRows
		imageI[index] = x[index, i]
	end

	for j in 1:nCols
		# Loop to fill the vector imageJ with the j-th data point.
		for index in 1:nRows
			imageJ[index] = x[index, j]
		end
		distances[j] = euclidean_distance(imageI, imageJ)
	end

	sortedNeighbors = sortperm(distances)

	# Select the second closest neighbor since the calculated closest is i-th to itself.
	kNearestNeighbors = sortedNeighbors[2:k+1]
	return kNearestNeighbors
end

# ----------------
# OUTPUT DATA
# ----------------

# Random Forest Model Output
writetable("$(path)/submission_rfc.csv", labelsInfoTest, separator=',', header=True)

# K-Nearest Neighbor Output