# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt 

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   # Write histogram equalization here
   b,g,r = cv2.split(img_in)
   
   #calculate histogram of 3 channels
   bhist,bbins = np.histogram(b.ravel(),256,[0,256])
   ghist,gbins = np.histogram(g.ravel(),256,[0,256])
   rhist,rbins = np.histogram(r.ravel(),256,[0,256])

   #compute cumulative distribution of blue channel and apply masking
   cdfb = bhist.cumsum()
   cdf_m = np.ma.masked_equal(cdfb,0)  #masks where equal to the given value
   cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())  #here 255 is the no. of gray values
   cdfb = np.ma.filled(cdf_m,0).astype('uint8')  #this will assign the pixel value from 0 - 255 to the image

   #compute cumulative distribution of green channel and apply masking
   cdfg = ghist.cumsum()
   cdf_m = np.ma.masked_equal(cdfg,0)
   cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
   cdfg = np.ma.filled(cdf_m,0).astype('uint8')

   #compute cumulative distribution of red channel and apply masking
   cdfr = rhist.cumsum()
   cdf_m = np.ma.masked_equal(cdfr,0)
   cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
   cdfr = np.ma.filled(cdf_m,0).astype('uint8')

   #converting the cdf to each respective image
   img_b = cdfb[b]
   img_g = cdfg[g]
   img_r = cdfr[r]

   #merge channels to single image
   img_in = cv2.merge((img_b, img_g, img_r))

   img_out = img_in   #Histogram Equalization result
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
	
   # Write low pass filter here
   
   img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   img_float32 = np.float32(img_in)
   dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT) #perform DFT and produce a full size complex output array
   dft_shift = np.fft.fftshift(dft)

   #mag_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) #this produces the magnitude spectrum
   rows, cols = img_in.shape
   crow,ccol = rows/2 , cols/2     #retrive the centre of the image in terms of row,column coordinate

  #for low pass filter, mask such that centre pixel is 1 and rest all pixels are zero
   mask = np.zeros((rows, cols, 2), np.uint8) 
   mask[crow-10:crow+10, ccol-10:ccol+10] = 1  # 20*20 mask

   # apply mask and inverse DFT
   fshift = dft_shift*mask
   f_inverse_shift = np.fft.ifftshift(fshift)
   img_in = cv2.idft(f_inverse_shift, flags =cv2.DFT_SCALE )
   img_in = cv2.magnitude(img_in[:,:,0],img_in[:,:,1])
   
   img_out = img_in  # Low pass filter result
   return True, img_out

def high_pass_filter(img_in):

   # Write high pass filter here

   img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

   #apply fast fourier transform
   f = np.fft.fft2(img_in)
   fshift = np.fft.fftshift(f)
   magnitude_spectrum = 20*np.log(np.abs(fshift))
   
   #for highpass filter, align the centre pixel to zero
   rows, cols = img_in.shape
   crow,ccol = rows/2 , cols/2
   fshift[crow-10:crow+10, ccol-10:ccol+10] = 0   #20 * 20 mask

   #apply inverse fourier transform
   f_ishift = np.fft.ifftshift(fshift)
   img_back = np.fft.ifft2(f_ishift)
   img_back = np.abs(img_back)
   
   img_out = img_back # High pass filter result
   
   return True, img_out
   
def deconvolution(img_in):
   
   # Write deconvolution codes here
   
   #img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   gk = cv2.getGaussianKernel(21,5)   #kernel size = 21 and sigma = 5
   gk = gk * gk.T

   #apply fourier transform
   def ft(img_in, newsize=None):
   	dft = np.fft.fft2(np.float32(img_in),newsize)
    	return np.fft.fftshift(dft)
  
   #apply inverse fourier transform
   def ift(shift):
    	f_ishift = np.fft.ifftshift(shift)
    	img_back = np.fft.ifft2(f_ishift)
    	return np.abs(img_back)

   imf = ft(img_in, (img_in.shape[0],img_in.shape[1])) # make sure sizes match
   gkf = ft(gk, (img_in.shape[0],img_in.shape[1])) # so we can multiple easily
   imconvf = imf / gkf

   # now for example we can reconstruct the blurred image from its FT
   blurred = ift(imconvf)
   img_out = 255* blurred   # Deconvolution result
   
   return True, img_out
   
def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   #input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   input_image2  = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);
   
   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   #img_in1 = cv2.cvtColor(img_in1, cv2.COLOR_BGR2RGB)
   #img_in2 = cv2.cvtColor(img_in2, cv2.COLOR_BGR2RGB)
   
   img_in1 = img_in1[:,:img_in1.shape[0]]
   img_in2 = img_in2[:img_in1.shape[0],:img_in1.shape[0]]
   
   # generate Gaussian pyramid for img_in1
   G = img_in1.copy()
   gpA = [G]
   for i in xrange(6):
   	 G = cv2.pyrDown(G)   #blurs the image and downsamples it
    	 gpA.append(G)

   # generate Gaussian pyramid for img_in2
   G = img_in2.copy()
   gpB = [G]
   for i in xrange(6):
    	G = cv2.pyrDown(G)    
    	gpB.append(G)

   # generate Laplacian Pyramid for A
   lpA = [gpA[5]]
   for i in xrange(5,0,-1):
   	GE = cv2.pyrUp(gpA[i])  #upsamples the image and blurs it
    	L = cv2.subtract(gpA[i-1],GE)
    	lpA.append(L)

   # generate Laplacian Pyramid for B
   lpB = [gpB[5]]
   for i in xrange(5,0,-1):
   	GE = cv2.pyrUp(gpB[i])
    	L = cv2.subtract(gpB[i-1],GE)
    	lpB.append(L)
  
   # Now add left and right halves of images in each level
   LS = []
   for la,lb in zip(lpA,lpB):
    	rows,cols,dpt = la.shape
    	ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    	LS.append(ls)
   
   # now reconstruct
   ls_ = LS[0]
   for i in xrange(1,6):
   	ls_ = cv2.pyrUp(ls_)
    	ls_ = cv2.add(ls_, LS[i])
   
   img_out = ls_     # Blending result
   
   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
