import cv2
import numpy as np
import joblib
import sys
import sudokusolver

# Import the trained classifier.
clf = joblib.load('classifier.pkl')
font = cv2.FONT_HERSHEY_SIMPLEX
test = 1
# If we get error while extraction then it will try a different image ratio to do that.
while test != 4:
	# Reading image
	img = cv2.imread(sys.argv[1])
	# It will resize the image to a speific dimensions.
	if test ==1:
		img = cv2.resize(img, (346,336))
	# It will zoom out the image if its size is too big.
	elif test == 2:
		print "inelse"
		img = cv2.pyrDown(img)	
	# It will zoom out and resize the image.
	elif test == 3:
		print "in here"
		img = cv2.pyrDown(img)	
		img = cv2.resize(img, (346,336))
	# Convert to grayscale
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Apply Canny edge detection, return will be a binary image
	edges = cv2.Canny(img,50,100,apertureSize = 3)
	# Apply Hough Line Trmatrix_columnform, minimum lenght of line is 200 pixels
	lines = cv2.HoughLines(edges,1,np.pi/180,200)	
	# Print and draw line on the original image

	l = []
	l1=[]
	# Just a check so that image is not None.
	if lines is not None:
		# Pushing all the lines coordinates in an array.	
		for a in lines:
			l.append([a[0][0],a[0][1]])
		# Pushing horizontal and vertical lines coordinates in l1.
		for i in l:
			# print i
			rho = i[0]
			theta = i[1]
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			# line_detection = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
			# print x1,y1,x2,y2

			# 1 in the array represents its horizontal
			if abs(x1) != 1000:
					l1.append([x1,0,1])
			# 0 the last item in [0,y1,0] represents that it is a vertical line.
			if abs(y1) != 1000:
					l1.append([0,y1,0])
		points = []
		l=l1
		for a in l:
			if a[2] == 1:
				for i in l:
					if i[2] == 0:
						points.append([a[0],i[1]])
		points= sorted(points)
		i=0
		# Merging points which are very close to each other horizontally.
		while i < len(points)-1:
			# If difference between to consecutive points less than 10.
			# Take the average of both and make it 1 point.
			if points[i+1][1] - points[i][1] <= 10 and points[i+1][1] - points[i][1]>0:
				points[i][1] = (points[i][1]+points[i+1][1])/2
				points.pop(i+1)
			i=i+1
		i=0
		final_points=[]
		flag=0
		# If points is less than 99 mematrix_column that we did not found all the corners of sudoku.
		if len(points) > 99:
			# Merging points which are very close to each other horizontally.
			while i < len(points)-10:
				# If difference between to consecutive points less than 10.
				# Take the average of both and make it 1 point.
				if points[i+10][0] - points[i][0] <= 10:
					points[i][0] = (points[i][0]+points[i+10][0])/2
					final_points.append(points[i])
					flag=0
				else:
					final_points.append(points[i])
					flag = 1
				i=i+1
				if flag ==0:
					if i%10 == 0:
						i+=10
			if len(final_points) != 100:
				for i in range(len(points)-10, len(points)):
					final_points.append(points[i])
			prev = final_points[11][0]
			# matrix_column will be used to append 9 digits which are extracted.
			# matrix will have sudoku as a matrix.
			j=0
			k=0
			matrix_column=[]
			matrix=[]
			p=[]
			p1=[]

			'''
			Now we have total of 100 points.
			So we can extract 81 cells of sudoku.
			So we will take two points of each cell that is top left corner and bottom right corner.
			Then with the help of our trained classifier we will extract data from all 81 cells.
			Once extracted we will send it to sudoku solver, which will return a matrix with solution.
			'''
			for i in range(len(final_points)-11):
				if final_points[i+11][0] == prev:
					# (x1,y1) ==> Top left corner and (x2,y2) ====> Right Bottom Corner
					x1 = final_points[i][0]
					y1 = final_points[i][1]
					x2 = final_points[i+11][0]
					y2 = final_points[i+11][1]
					diff_x =int((x2-x1)/5)
					diff_y = int((y2-y1)/5)
					# Creating a window.
					x1 = x1 + diff_x
					x2 = x2 - diff_x
					y1 = y1 + diff_y
					y2 = y2 - diff_y
					# We will use this image for digit extraction
					X = img[y1:y2,x1:x2]
					if X.size !=0:
						X = cv2.bitwise_not(X)
						X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
						X = cv2.resize(X, (36,36))
						img1 = np.reshape(X,(1,-1))
						# num is the predicted number for the cell with coordinates (x1,y1) and (x2,y2)
 						num = clf.predict(img1)
						matrix_column.append(num[0])
						p.append([x1,y1])
						k+=1
						if k == 9:
							k=0
							matrix.append(matrix_column)
							matrix_column=[]
							p1.append(p)
							p=[]
							
				else:
					prev = final_points[i+11][0]
			flag = 0
			for i in matrix:
				for j in i:
					if list(i).count(j) > 1 and j != 0:
						flag =1
						break;
			if len(matrix)!=9 and len(matrix[0])!=9:
				flag = 1
			# Transposing matrix to get sudoku as it is in image.
			sudoku = np.array(matrix).transpose()
			if flag==0:
				for i in sudoku:
					for j in i:
						if list(i).count(j) > 1 and j != 0:
							flag = 1
							break;
			print "Input Sudoku\n", sudoku,"\n"
			# Checking if data extracted is correct or not. This is done by checking if extracted sudoku has same number twice in same column or row.
			if(flag==0):
				# Solving sudoku using a function from sudokusolver.py
				sudoku = sudokusolver.run(sudoku)
				if len(sudoku) == 9:
					print "Solved Sudoku\n",np.array(sudoku)
					sudoku= np.array(sudoku).transpose()
					# Putting text on input image to show the final output.
					for i in range(0,9):
						for j in range(0,9):
								if(matrix[i][j]==0):
									cv2.putText(img,str(sudoku[i][j]),(p1[i][j][0]+diff_x,p1[i][j][1]+3*diff_y),font,0.75,(0,255,0),2)
					# Display image if everything is successful.
					cv2.imshow("Solved Sudoku", img)
					cv2.waitKey(0)
					break;
				else:
					print "Wrong data extracted!!!"
					test+=1
			else:
				print "Digits extracted are wrong!! ",
				test+=1
		else:
			print "Rectangles not found"
			test+=1
	else:
		print "Error in image"
		test+=1

cv2.destroyAllWindows()
