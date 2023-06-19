######################################################
# Creating the images for task3
######################################################
#
# Rings with digits
#

import numpy as np
import cv2

# Get a dictionairy of markers
dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Get the markers
marker_side = 500
marker1 = cv2.aruco.drawMarker(dict,1,marker_side)
marker2 = cv2.aruco.drawMarker(dict,2,marker_side)
marker3 = cv2.aruco.drawMarker(dict,3,marker_side)
marker4 = cv2.aruco.drawMarker(dict,4,marker_side)

# Transform them into RGB images
marker1_rgb = np.stack([marker1, marker1, marker1], axis=2)
marker2_rgb = np.stack([marker2, marker2, marker2], axis=2)
marker3_rgb = np.stack([marker3, marker3, marker3], axis=2)
marker4_rgb = np.stack([marker4, marker4, marker4], axis=2)
    
# i and j are the numbers that will be included in the ring
for i in range(10):
    for j in range(10):
        
        # The en empty image for filling with content
        ring_image = np.zeros((3508,2480,3), np.uint8)
        ring_image[:,:]=(255,255,255)

        # Generate a random color for the ring (blue, green, red, yellow or black)
        rgb = np.random.randint(5)
        color = [0,0,0]
        if rgb == 4:
            color = (0,0,0)
        elif rgb == 3:
            color = (0,215,255)
        else:
            color[rgb] = 255
            color = tuple(color)

        # The font we will use for the numbers
        font = cv2.FONT_HERSHEY_SIMPLEX 

        # Draw the ring on the final image
        cv2.circle(ring_image,(ring_image.shape[1]/2,ring_image.shape[0]/2),1050,color,270)

        # Include the marker in the final image
        ring_image[:marker_side,:marker_side,:]=marker1_rgb
        ring_image[:marker_side,-marker_side:,:]=marker2_rgb
        ring_image[-marker_side:,:marker_side,:]=marker3_rgb
        ring_image[-marker_side:,-marker_side:,:]=marker4_rgb

        # Include some numbers in the ring
        string = str(i)+' '+str(j)
        cv2.putText(ring_image, string, (650,ring_image.shape[0]/2+220), font, 22, (0,0,0), 80, cv2.LINE_AA)
        
        # Create the image name (which includes the folder it will be saved into)
        image_name = 'ring_dig_' + str(i) + str(j) + '.png' 
        cv2.imwrite(image_name,ring_image)
        
        print('Generated: ', image_name)
