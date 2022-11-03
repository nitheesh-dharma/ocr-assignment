"""Completed classifiaction system.
Author : Nitheesh Dharmapalan
username : aca18nd 
"""
import numpy as np
import difflib
import utils.utils as utils
import scipy.linalg
from collections import OrderedDict
import operator
import math
from collections import Counter


def reduce_dimensions(feature_vectors_full, model):
    """Method that uses PCA 10 best features

    Param:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    # get vector v from model for testing stage which 
    # represents A 
    v = model.get("v")
    # get the best features 
    train_labels = model["labels_train"]
    # if v is none then, system is in traning stage
    if v is None:
        # Get the covariance
        covx = np.cov(feature_vectors_full, rowvar=0)

        # Calculate and get the best principal components
        N = covx.shape[0]
        w, v = scipy.linalg.eigh(covx, eigvals=(N - 30, N - 1))
        v = np.fliplr(v)

        # calculate the reduced training data 
        pca_data = np.dot((feature_vectors_full), v)
    else: 
        # calculate the reduced training data 
        pca_data = np.dot(feature_vectors_full, np.array(v))
    return v,pca_data[:,1:11]

def read_wordlist():
    """
    Get the dictionary and store it in a list
    """
    # open the dictionary file
    word_list_file = open("wordlist.txt","r")
    # Create an array to store the list of words
    word_list = []
    # Iterate throught each line
    for word in word_list_file: 
        # store word in file
        word_list.append(word.strip()) 
    word_list_file.close()
    return word_list




def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        # get padding
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        # get minimum height and weight
        h = min(h, bbox_h)
        w = min(w, bbox_w)

        # create the padded image 
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors



def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print("Reading data")
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print("Extracting features from training data")
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data["labels_train"] = labels_train.tolist()
    model_data["bbox_size"] = bbox_size
    

    print("Reducing to 10 dimensions")
    v,fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    # Store the vector v
    model_data["v"] = v.tolist()
    # Store the reduced feature vectors from training
    model_data["fvectors_train"] = fvectors_train.tolist()
    # Store the word list(deleted becauuse it was done in training stage)

    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    # Get the bounding box size
    bbox_size = model["bbox_size"]
    # load the test images
    images_test = utils.load_char_images(page_name)
    # convert test images to feature vectors
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    _,fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced

def find_most_common(feats,labels_train):
    """ 
    Finds the most common ocuuring label in 
    the features 

    Params: 
    feats - numpy array, containing features of an image 
    labels_train - numoy array, containg the training labels 

    """
    # Create a dictionary to count the occurence of a label
    labels_count = dict()
    # iterate through features
    for f in feats:
        currLabel = labels_train[f]
        # Check if current label is in the dictionary
        if currLabel in labels_count: 
            # Increment count of 
            labels_count[currLabel] += 1 
        else: 
            # Add label to dictionary
            labels_count[currLabel] = 1
    # Find most occuring label 
    label = max(labels_count.items(), key=operator.itemgetter(1))[0]
    return label

def classify_page(page, model):
    """
    Classifies a page using a the k nearest neighbour clasifier

    Params: 
    page - 2d array, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    # Get reduced trained feature vectors
    fvectors_train = np.array(model["fvectors_train"])
    # Get train labels 
    labels_train = np.array(model["labels_train"])

    # Calculate cosine distance
    x= np.dot(page, fvectors_train.transpose())
    modtest=np.sqrt(np.sum(page * page, axis=1))
    modtrain=np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x/np.outer(modtest, modtrain.transpose()) 

    # Sort the cosine distance 
    sorted_dist = np.argsort(dist,axis = 1)

    # Create an array to store labels 
    labels = []
    for i in sorted_dist: 
        # Find the most common label
       label = find_most_common(i[-9:],labels_train)
       # Add it to list of labels 
       labels.append(label)

    return np.array(labels)




def correct_errors(page, labels, bboxes, model):
    """ Returns labels unchanged.

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    
    return labels

def find(dictionary, word):
    """
    Finds the word in the dictionary using binary search. If it can't returns None 

    Params: 
    word - string, word to be found
    dictionary- list, dictionary of words 

    Code adapted from : https://stackoverflow.com/questions/34327244/binary-search-through-strings
    """
    # Get start and end points
    start = 0
    end = len(dictionary) - 1

    # Check if start is less or equal to end 
    while start <= end:
        # Get middle index 
        middle = (start + end)// 2
        # Get middle word 
        midpoint = dictionary[middle]
        # Check if midpoint is greater than word
        if midpoint > word:
            # Check the right subarray
            end = middle - 1
        elif midpoint < word:
            # check the left subarray 
            start = middle + 1
        else:
            return midpoint


def word2vec(word):
    """
    converts word to a vector 

    Params: 
    word: String to be converted to a vector 

    Code adapted from: https://towardsdatascience.com/finding-similar-names-using-cosine-similarity-a99eb943e1ab
    """

    # Count the number of characters in each word.
    count_characters = Counter(word)
    # Gets the set of characters.
    set_characters = set(count_characters)
    # Get the number of occurences of characters
    char_values = np.array(list(count_characters.values())) 
    # Calculate length of the vector 
    length = np.sqrt(np.sum(char_values * char_values))
    return count_characters, set_characters, length, word


def cosine_similarity(vector1, vector2):
    """
    Calculates the cosine distance between two vectors formed from words 

    Params: 
    vector1- list containing first vector 
    vector2 - lisr containing second vector 

    Code adapted from: https://towardsdatascience.com/finding-similar-names-using-cosine-similarity-a99eb943e1ab
    """
    # Get the common characters between the two character sets
    common_characters = vector1[1].intersection(vector2[1]) 
    # Get the common characters from vector 1
    vect1_arr = np.array([vector1[0][character]for character in common_characters])   
    # Get the common characters from vector 2 
    vect2_arr = np.array([vector2[0][character]for character in common_characters]) 
    # Sum of the product of each intersection character.
    product_summation = np.dot(vect1_arr,vect2_arr.transpose())
    # Gets the length of each vector from the word2vec output.
    length = vector1[2] * vector2[2]    
    # Calculates cosine similarity and rounds the value to ndigits decimal places.
    if length == 0:
        # Set value to 0 if word is empty.
        similarity = 0
    else:
        similarity = product_summation/length   
    
    return similarity

def correct_errors_2(page,labels,bboxes,model): 

    """
    Tries to correct spelling errors from clasifier labels 

    Params: 
    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """

    #list to store space between each word
    spaces = []
    for i in range (bboxes.shape[0]-1): 
        # calculate spance between current character and next character 
        curr_diff = (bboxes[i+1,0]-bboxes[i,2])
        if(curr_diff>=0): 
            spaces.append(curr_diff)

    # Convert spaces to numpy array 
    spaces = np.array(spaces)
    # Find the mean of spaces 
    mean_space = np.mean(spaces)
    # Find the standard deviation of spaces
    std_space = np.std(spaces)
    # Calculate the threshold for the space  
    ovr_space = round(mean_space + (0.5*std_space))

    # List to store the words from the classifier
    word_list_classifier = []
    # Dictionary from traininf stage
    word_list_dict = np.array(model["word_list"])

    word = ""
    for i in range(labels.size):
        curr_char = labels[i]
        # check if i is at the laste character 
        if i == labels.size-1: 
            # add character to word
            word += curr_char
            # add it to the list 
            word_list_classifier.append(word)
        else: 
            # Find difference between current character and nect charecter 
            curr_diff = abs(bboxes[i+1,0]-bboxes[i,2])
            # Add character to word
            word += curr_char
            # Check there is a space following the character 
            if curr_diff >= ovr_space: 
                # Add word to the list 
                word_list_classifier.append(word)
                word = ""


    word_list_classifier = np.array(word_list_classifier)

    # Counter to keep track of the current number of characters 
    curr_char_count = 0
    for word in word_list_classifier: 
        # set the word length and the new word length
        word_length = new_word_length= len(word)
        # update character count
        curr_char_count += word_length
        # Check if word does not start with upper case 
        if not word[0].isupper():
            # check if last character is not a lettter
            if  not word[-1].isalpha(): 
                # remove it from word
                word = word[:-1]
                # calculate new word length
                new_word_length = len(word)

            # Find the word in the dictionary 
            word_found = find(word_list_dict,word)

            # If no word is found 
            if word_found == None: 
                # Set the max similarity to zero
                max_sim = 0 
                # Store the correct word 
                correct_word = ""
                # Convert wrong word to vector
                vector1 = word2vec(word)
                for corr_word in word_list_dict:
                    # convert potetntial correct word to vector
                    vector2 = word2vec(word)
                    # Find the cosine similarity between vectors
                    curr_sim = cosine_similarity(vector1,vector2)
                    # Check if current similarity is more than max similarity 
                    if curr_sim > max_sim: 
                        # Set max sim to current sim
                        max_sim = curr_sim 
                        # Set the current word to the correct word
                        correct_word = corr_word 
                    # Check if the new word length is equal to the correct word length and
                    # max similiarity is more than 0.8
                    if new_word_length == len(correct_word) and max_sim >= 0.8: 
                        for i in range(len(correct_word)): 
                            # Find the wrong character 
                            if correct_word[i] != word[i]: 
                                # Get the index of the character 
                                index_to_change = curr_char_count -(word_length-i)
                                # Change the value of the character 
                                labels[index_to_change] = correct_word[i]
