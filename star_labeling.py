import cv2
import json
import matplotlib.pyplot as plt

def detect_stars(img_name, create_json: bool = False ,create_image: bool = False) -> list:
    # Load the image
    img = cv2.imread(img_name)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to isolate the brightest objects in the image
    _, thresh = cv2.threshold(gray, 160, 200, cv2.THRESH_BINARY)
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store the coordinates of the brightest stars
    stars = []
    # Iterate through the contours and filter out the small ones (which are likely noise)
    for i, c in enumerate(contours):
        # Get the bounding box coordinates of the contour
        x, y, w, h = cv2.boundingRect(c)
        
        # Calculate the radius of the bounding box
        r = int((w + h) / 4)
        
        # Calculate the brightness of the star (average pixel value in the bounding box)
        b = int(gray[y:y+h, x:x+w].mean())
        
        # Store the coordinates of the star as x,y,r,b
        stars.append((x + w/2, y + h/2, r, b))



    # Sort stars by brightness
    stars.sort(key=lambda x: x[3], reverse=True)
    if create_image:
        for i, (x, y, r, b) in enumerate(stars):
            # Draw a circle and label text around the center of the contour
            cv2.circle(img=img, center=(int(x+5), int(y+5)), radius=2, color=(0, 0, 255), thickness=2)
            # Construct the label text and draw it on the image
            label_text = f"({i})"
            cv2.putText(img, label_text, (int(x) + r + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save the modified image to a file
        cv2.imwrite(img_name.split('.')[0]+'_image.jpg', img)

    if create_json:
        # Write the star coordinates to a json file
        with open(img_name.split('.')[0]+'_data.json', 'w') as f:
            json.dump(stars, f)

    return stars       

def match_stars(stars1, stars2):
    matched_pairs = []
    found = False
    for star1_1 in range(len(stars1)):
        found = False
        for star1_2 in range(len(stars1)):
            for star1_3 in range(len(stars1)):
                for star2_1 in range(len(stars2)):
                    for star2_2 in range(len(stars2)):
                        for star2_3 in range(len(stars2)):
                            if star1_1 < star1_2 and star1_1 < star1_3 and star1_2 < star1_3 and star2_1 < star2_2 and star2_1 < star2_3 and star2_2 < star2_3:
                                triangle1 = (distance(stars1[star1_1], stars1[star1_2]), distance(stars1[star1_2], stars1[star1_3]), distance(stars1[star1_3], stars1[star1_1]))
                                triangle2 = (distance(stars2[star2_1], stars2[star2_2]), distance(stars2[star2_2], stars2[star2_3]), distance(stars2[star2_3], stars2[star2_1]))
                                if are_same_triangles(triangle1, triangle2, 0.1):
                                    # determine which star in triangle1 is which star in triangle2, by the distance to the center of the triangle
                                    # the closest star to the center of the triangles is the same star in both triangles
                                    # when the stars are matched, add them to the matched_pairs list as a pair
                                    center1 = (sum([star[0] for star in [stars1[star1_1], stars1[star1_2], stars1[star1_3]]])/3, sum([star[1] for star in [stars1[star1_1], stars1[star1_2], stars1[star1_3]]])/3)
                                    center2 = (sum([star[0] for star in [stars2[star2_1], stars2[star2_2], stars2[star2_3]]])/3, sum([star[1] for star in [stars2[star2_1], stars2[star2_2], stars2[star2_3]]])/3)

                                    star1 = min([stars1[star1_1], stars1[star1_2], stars1[star1_3]], key=lambda star: distance(star, center1))
                                    star2 = min([stars2[star2_1], stars2[star2_2], stars2[star2_3]], key=lambda star: distance(star, center2))

                                    star3 = max([stars1[star1_1], stars1[star1_2], stars1[star1_3]], key=lambda star: distance(star, center1))
                                    star4 = max([stars2[star2_1], stars2[star2_2], stars2[star2_3]], key=lambda star: distance(star, center2))

                                    star5 = [stars1[star1_1], stars1[star1_2], stars1[star1_3]]
                                    star5.remove(star1)
                                    star5.remove(star3)
                                    star6 = [stars2[star2_1], stars2[star2_2], stars2[star2_3]]
                                    star6.remove(star2)
                                    star6.remove(star4)

                                    matched_pairs.append((star1, star2))
                                    matched_pairs.append((star3, star4))
                                    matched_pairs.append((star5[0], star6[0]))
                                    found = True
                                    break
                            
                        if found: break
                    if found: break
                if found: break
            if found: break

                                                                        
                                    

                        
    
    for pair in matched_pairs:
        print(stars1.index(pair[0]), stars2.index(pair[1]))
    return matched_pairs

def visualize_matched_pairs(matched_pairs):
    """
    Function to visualize matched star pairs.

    Args:
        matched_pairs (list): List of pairs of matched star tuples.

    Returns:
        None
    """
    # Extract x, y coordinates of stars from matched pairs
    x1 = [star1[0] for star1, star2 in matched_pairs]
    y1 = [star1[1] for star1, star2 in matched_pairs]
    x2 = [star2[0] for star1, star2 in matched_pairs if star2 is not None]
    y2 = [star2[1] for star1, star2 in matched_pairs if star2 is not None]
    print(len(x2))
    # Create scatter plot of matched stars
    plt.scatter(x1, y1, color='blue', label='Stars 1', s=10)
    plt.scatter(x2, y2, color='red', label='Stars 2', s=30)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Matched Star Pairs')
    plt.legend()
    plt.show()


def find_common_stars(image1_path, image2_path):
    # Detect stars in both images
    image1_stars = detect_stars(image1_path, create_image=True)
    image2_stars = detect_stars(image2_path, create_image=True)

    print(f"Stars in {image1_path}: {len(image1_stars)}, \n {image1_stars} \n")
    print(f"Stars in {image2_path}: {len(image2_stars)} \n {image2_stars} \n")
    
    visualize_matched_pairs(match_stars(image1_stars, image2_stars))

def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def are_same_triangles(triangle1, triangle2, tolerance=0.1):
    # Normalize side lengths
    min_side_length1 = min(triangle1)
    normalized_triangle1 = [side_length / min_side_length1 for side_length in triangle1]
    min_side_length2 = min(triangle2)
    normalized_triangle2 = [side_length / min_side_length2 for side_length in triangle2]

    # Sort normalized side lengths
    sorted_triangle1 = sorted(normalized_triangle1)
    sorted_triangle2 = sorted(normalized_triangle2)

    # Compare sorted normalized side lengths with tolerance
    for i in range(3):
        if abs(sorted_triangle1[i] - sorted_triangle2[i]) > tolerance:
            return False

    return True



image1_stars = detect_stars('stars.png', create_image=True)
image2_stars = detect_stars('stars2.png', create_image=True)

# in image1_stars 0, 1, 2 are
# in image2_stars 4, 9, 3
# calculate the distances between them to compare the triangle:
# print(are_same_triangles((distance(image1_stars[0], image1_stars[1]), distance(image1_stars[1], image1_stars[2]), distance(image1_stars[2], image1_stars[0])) ,(distance(image2_stars[4], image2_stars[9]), distance(image2_stars[9], image2_stars[3]), distance(image2_stars[3], image2_stars[4])) ))


# Path: star_matching.py
print(find_common_stars('stars.png', 'stars2.png'))
