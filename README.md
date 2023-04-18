# star-tracker
The codebase consists of two main functions. The first function detects stars in a single image using OpenCV library. The function applies image thresholding to convert the image to binary format and then identifies star shapes by finding contours. The circularity of each contour is calculated to filter out non-star shapes. The function then sorts the stars by brightness and returns a list of tuples that represent the star's center, radius, and brightness. <br>

The second function takes two images and finds corresponding stars in both images. The function first randomly selects a subset of points from each image and then finds the triangles that can be formed from each set of three points. For each triangle, the function applies an affine transformation to one of the images and scales, rotates, and translates it to match the triangle in the other image. The function then checks if the stars in the transformed image match the stars in the other image. This process is repeated for all possible triangles formed from the selected points, and the best matching set of stars is returned. <br>


