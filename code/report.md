# OCR assignment report

## Feature Extraction (Max 200 Words)
The feature extraction technique was PCA. The reason no other technique was used was
because it's not as robust and will not give as accurate of results.
The number of principal components used was 30 because after experimenting with higher numbers, 
it gave results as good as 40 and 50 and gave better results than 60 and higher. 
The function returns PCA components in the range 2 to 11. 
The first feature is ignored because the first component is an average of all the other components. 
The features were also not centred about the mean as this also lowered the score. 
There was an attempt to use feature selection with PCA by computing the 1D multivariance divergence between all the features, 
combining the divergence and finding the best feature. 
There were several methods used to combine the divergence of each feature.
This included finding the mean of each feature and taking the features with the 10 highest scores, finding the highest divergence scores overall, 
taking the worst divergence scores from each feature and finding the best among them,
taking the most occurring divergences out of the 10 highest divergences for each feature pair. 
None worked as well as basic PCA. 


## Classifier (Max 200 Words)
The classifier chosen was k nearest neighbour. A gaussian classifier was not chosen because words in
a book are unlikely to fall in a distribution. So a non parametric classifier is chosen. The KNN
algorithm is preferred over the NN algorithm because it is more representative of the closest labels 
as it takes into consideration the k nearest labels instead of just assigning the closest label to 
the test label. The KNN algorithm will take into consideration the 9 closest labels and then pick the
most occurring label out of all. 
This seemed to work the best when tested against k of size 10,9,8,7,4,3. 

## Error Correction (Max 200 Words)
The error correction first combines characters from classifier to list of words
The space between the words is found by taking the average between all the spaces plus half a
standard deviation above the average. This seemed to work best for the set provided. 
If a space value above the value calculated was used, then some words were merged together. 
For every word, binary search was used to see if the word belongs in the dictionary. If the word was not found, then 
a replacement word would be tried to be found. The replacement word would only be found if the incorrect word did not begin
with a capital so that names and places are ignore. This word would only be used if it is the same
length as the original word. If it isn't, it means that some type of punctuation might have been mistaken for a character. 
It is hard to see what punctuation should be used as this would require the context of surrounding words,
which would be hard to implement(because of time). The cosine distance was used as a similarity measure between words as 
it would be faster for a big set than the Levenstein distance.  

## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
- Page 1: 96.3% 
- Page 2: 96.2% 
- Page 3: 83.5% 
- Page 4: 59.7%
- Page 5: 41.4% 
- Page 6: 32.4% 

## Other information (Optional, Max 100 words)
From the above results, it can be seen that the system works better for all the pages except the 
first 2 which are lower by about 2 percent.So,it performs better than basic PCA and nearest neighbour. 
The error correction part has been moved to a different function called correct_errors_2 as it took too
long to run. 
