
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        if(descriptorType.compare("DES_HOG") == 0)
        {
            normType = cv::NORM_L1;
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // make sure to handle binary vs hog descriptors
        if(descSource.type() != CV_32F || descRef.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        int k = 2;
        double ratio = 0.8;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        for(int i =0; i<knn_matches.size(); i++)
        {
            if(knn_matches[i][0].distance < ratio*(knn_matches[i][1].distance))
            {
                matches.push_back(knn_matches[i][0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    static std::vector<double> extract_time;

    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    // "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        // extractor = cv::xfeatures2d::SiftDescriptorExtractor::create(); // OLDER VERSIONS OF OPENCV
	extractor = cv::SiftDescriptorExtractor::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    
    extract_time.push_back(t);
    if(extract_time.size()==10)
    {
        double average = std::accumulate(extract_time.begin(), extract_time.end(), 0.0) / 10.0 * 1000.0;
        cout << "AVERAGE EXTRACT TIME: " << average << " ms.\n";
        // return average;
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    static std::vector<double> detect_time;

    // output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    // parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    double threshold = 100.0;

    // apply detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    // Normalizing
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // loop over corners
    for(int r=0; r<dst_norm.rows;r++)
    {
        for(int c=0; c<dst_norm.cols; c++)
        {
            if((int)dst_norm.at<float>(r,c) > threshold)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(c,r);
                newKeyPoint.size = blockSize;
                keypoints.push_back(newKeyPoint);
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    detect_time.push_back(t);
    if(detect_time.size()==10)
    {
        double average = std::accumulate(detect_time.begin(), detect_time.end(), 0.0) / 10.0 * 1000.0;
        cout << "AVERAGE DETECT TIME: " << average << " ms.\n";
        // return average;
    }
    

}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    static std::vector<double> detect_time;

    cv::Ptr<cv::FeatureDetector> detector;

    // Create detector
    if(detectorType.compare("FAST") == 0)
    {
        //cv::Ptr<cv::FastFeatureDetector>
        detector = cv::FastFeatureDetector::create();
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        // cv::Ptr<cv::BRISK> 
        detector = cv::BRISK::create();
    }
    else if(detectorType.compare("ORB") == 0)
    {
        // cv::Ptr<cv::ORB>
        detector = cv::ORB::create();
    } 
    else if(detectorType.compare("AKAZE") == 0)
    {
        // cv::Ptr<cv::AKAZE> 
        detector = cv::AKAZE::create();
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        // cv::Ptr<cv::xfeatures2d::SIFT>
        //detector = cv::xfeatures2d::SIFT::create(); // OLDER VERSIONS OF OPENCV
	    detector = cv::SIFT::create();
    }

    // Run detection
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Modern Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    detect_time.push_back(t);
    if(detect_time.size()==10)
    {
        double average = std::accumulate(detect_time.begin(), detect_time.end(), 0.0) / 10.0 * 1000.0;
        cout << "AVERAGE DETECT TIME: " << average << " ms.\n";
        // return average;
    }

}
