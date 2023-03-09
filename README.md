# Brain-tumor-detection-using-fuzzy-c-means-FCM-algorithm


the program loads an image of a brain, preprocesses the image using Otsu's thresholding method to create a binary image, and flattens the image into a 1D array. It then initializes the FCM algorithm parameters, including the number of clusters, the fuzziness parameter, the maximum number of iterations, and a tolerance threshold for convergence. The program then runs the FCM algorithm to cluster the pixels in the image into two clusters (tumor and non-tumor). After convergence, the program applies K-means clustering to the membership matrix to extract the tumor pixels, and then highlights these pixels in the original image. The resulting image with highlighted tumor pixels is then saved as 'brain_tumor.jpg'.
