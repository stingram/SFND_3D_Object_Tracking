
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        // std::cout << str1 << "\n";
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_HERSHEY_SIMPLEX, 4, currColor); // cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        std::cout << str2 << "\n";
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_HERSHEY_SIMPLEX, 4, currColor); // cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);//cv::namedWindow(windowName, 1); cv::WINDOW_NORMAL)
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Associate the given bounding box with all keypoint matches whose Curr keypoint is within the ROI
    // Calculate euclidean distance between the curernt and previous keypoint for each match
    std::vector<double> euclidean_dists;
    for(auto it=kptMatches.begin();it!=kptMatches.end();it++)
    {
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        if(!boundingBox.roi.contains(kpCurr.pt))
        {
            cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);
            double dist = cv::norm(kpPrev.pt-kpCurr.pt);
            euclidean_dists.push_back(dist);
        }

    }
    // Compute median euclidean distance over all matches
    std::sort(euclidean_dists.begin(),euclidean_dists.end());
    double median = euclidean_dists.size() % 2 == 0 ? 0.5*(euclidean_dists[euclidean_dists.size()/2]+euclidean_dists[euclidean_dists.size()/2-1]) : euclidean_dists[euclidean_dists.size()/2];
    double max_dist = 2.0;
    // Remove all matches where the distance between the current and previous is too far from the median
    for(auto it=kptMatches.begin();it!=kptMatches.end();it++)
    {
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        if(!boundingBox.roi.contains(kpCurr.pt))
        {
            cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);
            double dist = cv::norm(kpPrev.pt-kpCurr.pt);
            if(dist < median + max_dist && dist > median - max_dist)
            {
                boundingBox.kptMatches.push_back(*it);
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    std::sort(distRatios.begin(), distRatios.end());

    double medianDistRatio;
    if(distRatios.size() % 2 == 0)
    {
        medianDistRatio = 0.5*(distRatios[int(distRatios.size()/2)] + distRatios[int(distRatios.size()/2)-1]);
    }
    else
    {
        medianDistRatio = distRatios[int(distRatios.size()/2)];
    }
    TTC = -(1.0/frameRate) / (1 - medianDistRatio);
}



double min_x_lidar_inlier(std::vector<LidarPoint> &lidarPoints)
{
    std::vector<double> x_values;
    for(int i=0;i<lidarPoints.size();i++)
    {
        x_values.push_back(lidarPoints[i].x);
    }
    std::sort(x_values.begin(),x_values.end());

    // Q1 = median of n smallest entries
    // Q3 = median of n largest entries
    // IQR = Q3 - Q1
    // if x_i < Q1 - 1.5*IQR, then it's an outlier
    double median, Q1, Q3, IQR, k, lower_bound;
    int n;
    k = 1.5;

    if(x_values.size() % 2 == 0)
    {
        median = (x_values[x_values.size()/2] + x_values[x_values.size()/2-1]) / 2.0; 
        n = x_values.size()/2;
    }
    else
    {
        median = x_values[x_values.size()/2];
        n = (x_values.size() - 1) / 2;
    }

    if(n % 2 == 0)
    {
        Q1 = (x_values[n/2] + x_values[n/2-1]) / 2.0;
        Q3 = (x_values[x_values.size()-n+n/2] + x_values[x_values.size()-n+n/2-1]) / 2.0; 
    }
    else
    {
        Q1 = x_values[n/2];
        Q3 = x_values[x_values.size()-n+n/2];
    }

    IQR = Q3 - Q1;
    lower_bound = Q1 - k*IQR;

    for(auto it=x_values.begin();it!=x_values.end();it++)
    {
        if(*it < lower_bound){
            x_values.erase(it);
            it--;
        }
    }
    // Returns smallest inlier x value
    return x_values[0];
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // At this point, we've already got Lidar points associated with a single bounding box in previous and current frame

    // USE IQR to remove outliers for each LidarPoint vector
    // Use smallest X values from previous and current 3D bounding box 
    double prev_X = min_x_lidar_inlier(lidarPointsPrev);
    double curr_X = min_x_lidar_inlier(lidarPointsCurr);

    // Assume constant velocity model
    TTC = prev_X * (1.0/frameRate) / (prev_X-curr_X);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::multimap<int, int> bbox_matches;
    cv::KeyPoint prevKP;
    cv::KeyPoint currKP;
        
    // loop over all matches
    for(auto it=matches.begin();it!=matches.end();it++)
    {
        // get position of matches from each image
        // train is prevFrame and query is currFrame
        // everything is in the Dataframe
        prevKP = prevFrame.keypoints[it->queryIdx];
        currKP = currFrame.keypoints[it->trainIdx];


        // find out which bboxes in previous frame and current frame contain the match
        for(auto prev_it=prevFrame.boundingBoxes.begin();prev_it!=prevFrame.boundingBoxes.end();prev_it++)
        {
            // add these bbox ids to a multimap where matchIdx is the key and the values are bbox ids
            if(prev_it->roi.contains(prevKP.pt))
            {
                // now see which current frame bounding box this point is in
                for(auto curr_it=currFrame.boundingBoxes.begin();curr_it!=currFrame.boundingBoxes.end();curr_it++)
                {
                    if(curr_it->roi.contains(currKP.pt))
                    {
                        bbox_matches.insert(std::pair<int,int>(prev_it->boxID, curr_it->boxID));
                    }
                }
            }
        }
    }

    // loop over multimap and count all matches that have the same bbox id in previous frame, then
    // the current bbox that has the most matches is associated with that previous bbox
    std::unordered_map<int,int> counts;
    int max_count, res;
    typedef std::multimap<int,int>::iterator MMAPIterator;
    std::pair<MMAPIterator, MMAPIterator> result;
    
    for(auto it=bbox_matches.begin(); it!=bbox_matches.end(); it++)
    {
        // find mode for each key
        result = bbox_matches.equal_range(it->first);
        for(MMAPIterator mit = result.first; mit!= result.second; mit++)
        {
            counts[mit->second]++;
        }
        // now that I have counts, I pick one with most elements
        max_count = 0; res = -1;
        for(auto i: counts)
        {
            if(max_count < i.second)
            {
                res = i.first;
                max_count = i.second;
            }
        }
        // res is the current bbox id for given prev bbox id
        if(res != -1)
        {
            bbBestMatches[it->first]=res;
        }
        counts.clear();
    }
}