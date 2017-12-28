# ================================================

# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region
    aligned_img = cv2.imread("images/pattern001.jpg")
    h,w = ref_white.shape
    
    #create a blank image
    img = np.zeros((h,w,3), np.float32)
    norm_img = np.zeros((h,w,3), np.float32)
    
    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)
    
    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2)), (0,0), fx=scale_factor,fy=scale_factor)
	
	#===convert the image into grayscale image===
	patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2),cv2.IMREAD_GRAYSCALE)/255.0, (0,0), fx=scale_factor,fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
	temp = on_mask
	
        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        
        # TODO: populate scan_bits by putting the bit_code according to on_mask
	for a in range(len(on_mask)):
                for b in range(len(on_mask[a])):
                        k = temp[a,b]
                        if k == True:
				scan_bits[a,b]=scan_bits[a,b] + bit_code
    
    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)
    
    camera_points = []
    projector_points = []
    rgb_array = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code
	    a,b = binary_codes_ids_codebook[scan_bits[y,x]]
            c_x,c_y = x/2.0,y/2.0  
            p_x,p_y = a,b
	    if p_x >= 1279 or p_y >= 799:
		continue
	    camera_points.append((c_x,c_y))
	    projector_points.append((p_x,p_y))	
            blue,green,red = aligned_img[y,x]
	    img[y,x] = [0,p_y,p_x]
    	    rgb_array.append((blue,green,red))

    #obtain the correspondence image    
    cv2.normalize(img,norm_img,  0,255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite("correspondence.jpg",norm_img)

    camera_points = np.reshape(camera_points,(len(camera_points),1,2))
    projector_points = np.array(np.reshape(projector_points,(len(projector_points),1,2)), dtype=np.float32)
    rgb_array = np.reshape(rgb_array,(len(rgb_array),1,3))
            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
     
            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
    # now that we have 2D-2D correspondances, we can triangulate 3D points!
    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']   #camera_matrix
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']
 	
    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    norm_camera = cv2.undistortPoints(camera_points,camera_K,camera_d)
    
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    norm_proj = cv2.undistortPoints(projector_points,projector_K,projector_d)
    

    # Projection matrix
    rt_matrix = np.hstack((projector_R,projector_t))

    cam_proj_matrix = np.array([ [1,0,0,0],
                		[0,1,0,0],
                		[0,0,1,0]], dtype = np.float32)

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    triangulation = cv2.triangulatePoints(cam_proj_matrix,rt_matrix,norm_camera,norm_proj)	

    triangulation_points = triangulation.transpose()

    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    points_3d_prev = cv2.convertPointsFromHomogeneous(triangulation_points)
    
    # TODO: name the resulted 3D points as "points_3d"	
    
    points_3d = []
    points_3d_rgb = []
    mask = (points_3d_prev[:,:,2] > 200) & (points_3d_prev[:,:,2] < 1400)
    
    for i in range(len(points_3d_prev)):
        for j in range(len(points_3d_prev[i])):
            if mask[i] == True:
		points_3d.append(points_3d_prev[i,:,:])
 		points_3d_rgb.append(rgb_array[i,:,:])
    
    points_3d = np.array(points_3d)
    points_3d_rgb = np.array(points_3d_rgb)
    
    points_3d_rgb = np.concatenate((points_3d,points_3d_rgb),axis=1)
    
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name,"w") as f:
         for p in points_3d_rgb:
             f.write("%d %d %d %d %d %d\n"%(p[0,0],p[0,1],p[0,2],p[1,2],p[1,1],p[1,0]))

    return points_3d
	
def write_3d_points(points_3d):
	
	# ===== DO NOT CHANGE THIS FUNCTION =====
	
	print("write output point cloud")
    	print(points_3d.shape)
	output_name = sys.argv[1] + "output.xyz"
    	with open(output_name,"w") as f:
        	for p in points_3d:
            		f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))

    	#return points_3d, camera_points, projector_points
    
if __name__ == '__main__':

	# ===== DO NOT CHANGE THIS FUNCTION =====
	
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)

	
