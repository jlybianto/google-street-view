# ----------------
# IMPORT PACKAGES
# ----------------

Pkg.add("Images")
Pkg.add("DataFrames")

using Images
using DataFrames

# Add two parallel processes to the program to increase speed by approximately 2.
# '@everywhere' is a macro to perform the parallelization added before each function.
# '@parallel' is a macro added before each FOR loop.
addprocs(2)

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

# Function to store each image in a matrix
function read_data(typeData, labelsInfo, imageSize, path)
	# Initialize matrix
	x = zeros(size(labelsInfo, 1), imageSize)

	for (index, idImage) in enumerate(labelsInfo[:ID])

		# Read image file through location specification
		nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
		img = load(nameFile)

		# Convert image to gray colors
		temp = convert(Image{Images.Gray}, img)
		if ndims(temp) == 3
			temp = mean(temp.data, 1)
		end

		# Transform image matrix to a vector to be stored
		x[index, :] = reshape(temp, 1, imageSize)
	end
	return x
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
@everywhere function euclidean_distance(a, b)
	distance = 0.0
	for index in 1:size(a, 1)
		distance += (a[index]-b[index]) * (a[index]-b[index])
	end
	return distance
end

# Function to find the k-nearest neighbor of the i-th data point.
@everywhere function get_k_nearest_neighbors(xTrain, imageI, k)
	nRows, nCols = size(xTrain)

	# Initialize a vector imageI so that it is accessed only once from the X matrix.
	# Filling an empty vector with each element is faster than copying entire vector.
	# Create empty array of nRows elements of type Float32 (decimal).
	imageJ = Array(Float32, nRows)

	# Initialize an empty vector that will contain the distances between the i-th point.
	distances = Array(Float32, nCols)

	for j in 1:nCols
		# Loop to fill the vector imageJ with the j-th data point.
		for index in 1:nRows
			imageJ[index] = xTrain[index, j]
		end
		distances[j] = euclidean_distance(imageI, imageJ)
	end

	sortedNeighbors = sortperm(distances)

	# Select the second closest neighbor since the calculated closest is i-th to itself.
	kNearestNeighbors = sortedNeighbors[1:k]
	return kNearestNeighbors
end

# Function to assign a label to the i-th point according to the labels.
# Training data is stored in the X matrix while the labels are stored in y.
@everywhere function assign_label(xTrain, yTrain, k, imageI)
	kNearestNeighbors = get_k_nearest_neighbors(xTrain, imageI, k)

	# Dictionary to save the counts of the labels
	counts = Dict{Int, Int}()

	highestCount = 0
	mostPopularLabel = 0

	# Iterate over the labels of the k-nearest neighbor
	for n in kNearestNeighbors
		labelOfN = yTrain[n]
		# Add the current label to the dictionary if it is not already there
		if !haskey(counts, labelOfN)
			counts[labelOfN] = 0
		end
		# Add one to the count
		counts[labelOfN] += 1

		if counts[labelOfN] > highestCount
			highestCount = counts[labelOfN]
			mostPopularLabel = labelOfN
		end
	end
	return mostPopularLabel
end

# Define k
k = 3

# Run predictive modeling
yPredictions = @parallel (vcat) for i in 1:size(xTest, 2)
	nRows = size(xTrain, 1)
	imageI = Array(Float32, nRows)
	for index in 1:nRows
		imageI[index] = xTeset[index, i]
	end
	assign_label(xTrain, yTrain, k, imageI)
end

# Convert integer predictions into character type
labelsInfoTest[:Class] = map(Char, yPredictions)

# ----------------
# OUTPUT DATA
# ----------------

# Random Forest Model Output
writetable("$(path)/submission_rfc.csv", labelsInfoTest, separator=',', header=true)

# K-Nearest Neighbor Output
writetable("$(path)/submission_knn.csv", labelsInfoTest, separator=',', header=true)