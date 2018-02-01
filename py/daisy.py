import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
import time
import code



def _lowe_ratio_test( kp1, kp2, matches_org ):
    """ Input keypoints and matches. Compute keypoints like : kp1, des1 = orb.detectAndCompute(im1, None)
    Compute matches like : matches_org = flann.knnMatch(des1.astype('float32'),des2.astype('float32'),k=2) #find 2 nearest neighbours

    Returns 2 Nx2 arrays. These arrays contains the correspondences in images. """
    __pt1 = []
    __pt2 = []
    for m in matches_org:
        if (m[0].distance) < 0.8 * m[1].distance: #good match
            # print 'G', m[0].trainIdx, 'th keypoint from im1 <---->', m[0].queryIdx, 'th keypoint from im2'

            _pt1 = np.array( kp1[ m[0].queryIdx ].pt ) #co-ordinate from 1st img
            _pt2 = np.array( kp2[ m[0].trainIdx ].pt )#co-ordinate from 2nd img corresponding
            #now _pt1 and _pt2 are corresponding co-oridnates

            __pt1.append( np.array(_pt1) )
            __pt2.append( np.array(_pt2) )

            # cv2.circle( im1, (int(_pt1[0]), int(_pt1[1])), 3, (0,255,0), -1 )
            # cv2.circle( im2, (int(_pt2[0]), int(_pt2[1])), 3, (0,255,0), -1 )

            # cv2.circle( im1, _pt1, 3, (0,255,0), -1 )
            # cv2.circle( im2, _pt2, 3, (0,255,0), -1 )
            # cv2.imshow( 'im1', im1 )
            # cv2.imshow( 'im2', im2 )
            # cv2.waitKey(10)
    __pt1 = np.array( __pt1)
    __pt2 = np.array( __pt2)
    return __pt1, __pt2



def debug_draw_matches( im1, pt1, im2, pt2, mask ):
    canvas = np.concatenate( (im1, im2), axis=1 )
    cv2.putText( canvas, 'nMatches: %03d' %(pt1.shape[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) )
    cv2.putText( canvas, 'nInliers: %03d' %(mask.sum()), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) )
    for p in range( pt1.shape[0] ): #loop over pt1 (and pt2)
        if mask[p,0] == 0:
            continue
        p1 = tuple(np.int0(pt1[p,:]))
        p2 = tuple(np.int0(pt2[p,:]) + [ 320,0 ] )

        cv2.circle( canvas, p1, 2, (255,0,0) )
        cv2.circle( canvas, p2, 2, (0,0,255) )
        cv2.line( canvas, p1, p2, (0,0,255) )

    return canvas

#
# Load Images
# im1 = cv2.imread( '../image/church1.jpg', 0)
# im2 = cv2.imread( '../image/church2.jpg', 0)

im1 = cv2.imread( '../image/a.png')
im2 = cv2.imread( '../image/b.png')
# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.xfeatures2d.DAISY_create()


step_size = 5
kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, im1.shape[0], step_size)
                                    for x in range(0, im1.shape[1], step_size)]

im1_keypts = cv2.drawKeypoints(im1,kp, None)
im2_keypts = cv2.drawKeypoints(im2,kp, None)
cv2.imshow( 'im1_keypts', im1_keypts)
cv2.imshow( 'im2_keypts', im2_keypts)
cv2.waitKey(0)

startTime = time.time()
dense_feat1 = sift.compute(im1, kp)
dense_feat2 = sift.compute(im2, kp)
print 'time taken for DAISY in sec : ', time.time() - startTime


#
# FLANN Matcher
startFLANN = time.time()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(dense_feat1[1].astype('float32'),dense_feat2[1].astype('float32'),k=2)
print 'flann matching took in sec : ', time.time() - startFLANN

#
# Lowe's ratio test
print 'Lowe Ratio Test'
__pt1, __pt2 = _lowe_ratio_test( kp, kp, matches )


#
# Essential Matrix Test
print 'F-test'
E, mask = cv2.findEssentialMat( __pt1, __pt2 )

print __pt1.shape
cv2.imshow( 'canvas', debug_draw_matches(im1, __pt1, im2, __pt2, mask ) )
# code.interact( local=locals() )
cv2.waitKey(0)
