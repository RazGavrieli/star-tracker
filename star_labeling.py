import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import random as rand

def detect_stars(img_name, create_json: bool = False ,create_image: bool = False) -> list:
    """
    Function to detect stars in an image.

    Args:
        img_name (str): Name of the image file.
        create_json (bool): Flag to create a json file with the star coordinates (default: False).
        create_image (bool): Flag to create an image with the star coordinates (default: False).

    Returns:
        list: List of star tuples, each represented as (x, y, r, b).
    """
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 160, 200, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stars = []
    for i, c in enumerate(contours):
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter out contours that are too large
        if w*h > 500:
            continue
        
        # Calculate the circularity of the contour
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter != 0:
            circularity = 4*math.pi*(area/(perimeter*perimeter))
            # Filter out contours that are not circular enough
            if circularity < 0:
                continue
        
        r = int((w + h) / 4)
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

def transform_point_set(point_set_to_transform, three_source_points, three_target_points):
    # for each points take only x and y values
    three_source_points = [(x, y) for x, y, _, _ in three_source_points]
    three_target_points = [(x, y) for x, y, _, _ in three_target_points]
    point_set_to_transform = [(x, y) for x, y, _, _ in point_set_to_transform]
    # Convert the point set to a NumPy array
    point_set = np.array(point_set_to_transform)
    
    # Add a column of ones to the point set
    point_set_ones = np.hstack([point_set, np.ones((len(point_set), 1))])
    
    # Compute the affine transformation matrix
    A = np.vstack([
        [three_source_points[0][0], three_source_points[0][1], 1, 0, 0, 0],
        [0, 0, 0, three_source_points[0][0], three_source_points[0][1], 1],
        [three_source_points[1][0], three_source_points[1][1], 1, 0, 0, 0],
        [0, 0, 0, three_source_points[1][0], three_source_points[1][1], 1],
        [three_source_points[2][0], three_source_points[2][1], 1, 0, 0, 0],
        [0, 0, 0, three_source_points[2][0], three_source_points[2][1], 1]
    ])
    b = np.array([three_target_points[0][0], three_target_points[0][1], three_target_points[1][0], three_target_points[1][1], three_target_points[2][0], three_target_points[2][1]])
    T = np.linalg.solve(A, b)
    T = np.reshape(np.append(T, [0, 0, 1]), (3, 3))
    
    # Apply the affine transformation to the point set
    transformed_point_set = np.dot(point_set_ones, T.T)[:, :2]
    
    return transformed_point_set.tolist()

def match_stars_random(stars1, stars2):
    # make sure stars1 is the smaller image
    if len(stars1) > len(stars2):
        print("warning: stars1 is larger than stars2, swapping..")
        stars1, stars2 = stars2, stars1
    
    iterations = 0
    treshold = 15
    bestTransformation = []
    bestAmountOfMatches = 0
    while iterations < 3000:
        # pick 3 random stars from stars1 and stars2
        random_stars1 = rand.sample(stars1, 3)
        random_stars2 = rand.sample(stars2, 3)
        if not (random_stars1[0][3] >= random_stars1[1][3] >= random_stars1[2][3] or random_stars2[0][3] >= random_stars2[1][3] >= random_stars2[2][3]):
            continue
        # check if it is a valid posibility
        triangle1 = (distance(random_stars1[0], random_stars1[1]), distance(random_stars1[1], random_stars1[2]), distance(random_stars1[2], random_stars1[0]))
        triangle2 = (distance(random_stars2[0], random_stars2[1]), distance(random_stars2[1], random_stars2[2]), distance(random_stars2[2], random_stars2[0]))
        if not same_triangles(triangle1, triangle2, 0.05):
            continue
        # transform stars1 with the transformation that is needed to match stars1 and stars2
        new_stars1 = transform_point_set(stars1, random_stars1, random_stars2)
        # count the amount of stars in stars2 that are within a certain distance from the transformed stars1
        currAmountOfMatches = 0
        for i in new_stars1:
            for j in stars2:
                if distance(i, j) < treshold:
                    currAmountOfMatches += 1
        if currAmountOfMatches > bestAmountOfMatches:
            bestAmountOfMatches = currAmountOfMatches
            bestTransformation = new_stars1
        iterations += 1
        if iterations % 100000 == 0:
            print(iterations)
        if bestAmountOfMatches >= len(stars1):
            print('breaking', iterations)
            return validate_transformation(stars1, stars2, treshold, bestTransformation)
    print('finishing', iterations)
    return validate_transformation(stars1, stars2, treshold, bestTransformation)

def match_stars(stars1, stars2):
    # make sure stars1 is the smaller image
    if len(stars1) > len(stars2):
        print("warning: stars1 is larger than stars2, swapping..")
        stars1, stars2 = stars2, stars1
    
    iterations = 0
    treshold = 15
    # iterate over each pair of stars in stars1 and stars2, and find the transformation, rotation and scale that is needed to be performed 
    # on stars1 to match stars2. Then, check if all the stars in stars1 are within a certain distance from the stars in stars2 after the transformation.
    bestTransformation = []
    bestAmountOfMatches = 0
    for star1_1 in stars1:
        for star2_1 in stars1:
            for star3_1 in stars1:
                if star1_1 == star2_1 or star1_1 == star3_1 or star2_1 == star3_1:
                    continue
                for star1_2 in stars2:
                    for star2_2 in stars2:
                        for star3_2 in stars2:
                            if star1_2 == star2_2 or star1_2 == star3_2 or star2_2 == star3_2:
                                continue
                            # check if brightness level is similar between each pair of 3 stars
                            if not (star1_1[3] >= star2_1[3] >= star3_1[3] or star1_2[3] >= star2_2[3] >= star3_2[3]):
                                continue
                            triangle1 = (distance(star1_1, star2_1), distance(star2_1, star3_1), distance(star3_1, star1_1))
                            triangle2 = (distance(star1_2, star2_2), distance(star2_2, star3_2), distance(star3_2, star1_2))
                            if not same_triangles(triangle1, triangle2, 0.05):
                                continue
                            new_stars1 = transform_point_set(stars1, [star1_1, star2_1, star3_1], [star1_2, star2_2, star3_2])
                            currAmountOfMatches = 0
                            for i in range(len(new_stars1)):
                                for j in range(len(stars2)):
                                    if distance(new_stars1[i], stars2[j]) < treshold:
                                        currAmountOfMatches += 1
                            if currAmountOfMatches > bestAmountOfMatches:
                                bestAmountOfMatches = currAmountOfMatches
                                bestTransformation = new_stars1
                            iterations += 1
                            if iterations % 100000 == 0:
                                print(iterations)
                            if bestAmountOfMatches >= len(stars1):
                                print('breaking', iterations)
                                return validate_transformation(stars1, stars2, treshold, bestTransformation)
    print('finishing', iterations)
    return validate_transformation(stars1, stars2, treshold, bestTransformation)

def validate_transformation(stars1, stars2, treshold, bestTransformation):
    # print("best amount of matches: ", len(bestTransformation))
    # print("best transformation: ", bestTransformation)
    
    matched_stars = []
    used_stars = set()
    for i in range(len(bestTransformation)):
        for j in range(len(stars2)):
            if distance(bestTransformation[i], stars2[j]) < treshold and stars2[j] not in used_stars:
                used_stars.add(stars2[j])
                matched_stars.append((stars1[i], stars2[j]))

    return matched_stars
            
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
    # Create scatter plot of matched stars
    plt.scatter(x1, y1, color='blue', label='Stars 1', s=30)
    plt.scatter(x2, y2, color='red', label='Stars 2', s=10)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Matched Star Pairs Visualization - Originally: Blue: Stars1, Red: Stars2')
    plt.legend()
    plt.show()

def print_and_save_results(res):
    for pair in res:
        print(pair)
    # save to json file
    with open('results.json', 'w') as f:
        json.dump(res, f)

def find_common_stars(image1_path, image2_path):
    # Detect stars in both images
    image1_stars = detect_stars(image1_path, create_image=True)
    image2_stars = detect_stars(image2_path, create_image=True)

    if len(image1_stars) == 0 or len(image2_stars) == 0:
        print("No stars detected in one or both images.")
        return
    elif len(image1_stars) + len(image2_stars) > 30:
        print("big input, optimazing")
        res = match_stars_random(image1_stars, image2_stars)
        print_and_save_results(res)
        visualize_matched_pairs(res)
    else:
        res = match_stars(image1_stars, image2_stars)
        print_and_save_results(res)
        visualize_matched_pairs(res)

def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def same_triangles(triangle1, triangle2, tolerance=0.1):
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


print(find_common_stars('data/ST_db1.png', 'data/fr1.jpg'))
# print(find_common_stars('data/stars.png', 'data/stars2.png'))
# print(find_common_stars('data/fr2.jpg', 'data/fr1.jpg'))


