# star-tracker
![fr1_image](https://user-images.githubusercontent.com/90526270/232881799-08b399cc-0aeb-4e24-8d91-042cbf83ea71.jpg)

The codebase consists of two main functions: <br>
`detect_stars()`
<br>The first function detects stars in a single image using OpenCV library. The function applies image thresholding to convert the image to binary format and then identifies star shapes by finding contours. <br>
The circularity of each contour is calculated to filter out non-star shapes. The function then sorts the stars by brightness and returns a list of tuples that represent the star's center, radius, and brightness. <br>
<br>

`find_common_stars()` <br>
The second function takes two images and finds corresponding stars in both images.<br>
The function first randomly selects a subset of points from each image and then finds the triangles that can be formed from each set of three points. For each triangle, the function applies an affine transformation to one of the images and scales, rotates, and translates it to match the triangle in the other image.<br>
The function then checks if the stars in the transformed image match the stars in the other image. This process is repeated for all possible triangles formed from the selected points, and the best matching set of stars is returned. <br>

<img width="610" alt="image" src="https://user-images.githubusercontent.com/90526270/232882031-6367590e-4b91-421a-9d7d-a59c17c2155c.png">


## HOW TO USE
If you want to run the code by yourself, please import star_labeling.py and specifically the `find_common_stars(image1_path, image2_path)` function. <br> 
The function also takes 3 additional arguments: `create_image=True, create_json=True, visualize=True `. <br>
Use these 3 arguments to customize the output according to your likings. <br>

Feel free to explore the output, if you can not manage to run it yourself, and output example is in the output_example directory. <br> 

<img width="1573" alt="image" src="https://user-images.githubusercontent.com/90526270/232882876-3bd45e08-2c5f-4f87-8610-e40e1cda2e86.png">


