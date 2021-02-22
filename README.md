# SFND 3D Object Tracking

![Final TTC estimation](results/ttc_estimation.png)

This projects implements a Time-to-Collision (TTC) processing pipeline that would be part of a collision avoidance system on a self-driving vehicle. It makes TTCs estimates based on measurements from Lidar and Radar sensors. This pipeline implements the following:
- OpenCV keypoint detectors descriptors
- Object detection using a pre-trained YOLO DNN on COCO dataset (training included images of vehicles) 
- Methods to track objects of interest by matching keypoints and corresponding bounding boxes across successive images
- Associating regions in a camera image with lidar points in 3D space

The flowchart illustrates the TTC processing pipeline. Blocks in the orange rectangle use read images from an XYZ camera, detect, extract, and match relevant keypoints. In the blue box and downstream, Lidar points are cropped nd clustered cropped, while objects detected with the YOLO deep neural network from the camera are matched the corrsponding lidar point cloud. These clusters of lidar points and keypoints are tracked across frames by considering the strength of keypoint correspondences within their bounding boxes. Finally, a robust estimation of time-to-collision (TTC) is performed with data from both the lidar and camera sensors.

<img src="images/course_code_structure.png" width="779" height="414" />

## Dependencies
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
  * `brew install git-lfs  # Install for macOS, see link above for others`
  * `git remote add https://github.com/udacity/SFND_3D_Object_Tracking.git`
  * `git lfs install`
  * `git lfs pull upstream`
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Building and running the project
```
mkdir build && cd build
cmake ..
make
./3D_object_tracking
```

## Implementation Write-up

### Matching 3D objects
_Lines 301-367 in camFusion.cpp_  
The "matchBoundingBoxes" method , which takes as input both the previous and the current data frames and provides as the output the ids of the matched regions of interest (i.e. the boxID property), is implemented where each bounding box is assigned the match candidate with the highest number of occurences. See function definition below for implementation details. 
```cpp
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
```

### Computing lidar-based TTC
_Lines 286-298 in camFusion.cpp_  
The time-to-collision (TTC) in seconds for all matched 3D objects using only Lidar measurements from the matched bounding boxes between the current and previous frame is computed. Outliers are removed using the InterQuartile Range (IQR) algorithm. Points that lie too far from the median of the lower quartile are not considered when computing TTC.
```cpp
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
```

_Lines 233-284 in camFusion.cpp_  
To calculate minimum inlier points I implemented the function below.
```cpp
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
```

### Associating Keypoint Correspondences with Bounding Boxes
_Lines 137-171 in camFusion.cpp_  
The function "clusterKptMatchesWithROI" prepares the TTC computation based on camera measurements by associating keypoint correspondences to bounding boxes which enclose them. All the matches that sastisfy this condiation are to added to vector in the respective bounding boxes only after outliers have been removed based on the euclidean distance between them in relation to all the matches in the bounding box.
```cpp
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
```

### Computing Camera-based TTC
_Lines 145-189 in camFusion.cpp_  
The time-to-collision in seconds for all matched 3D objects using only keypoint correspondences from the matching bounding boxes between the current and previous frame is computed in function `computeTTCCamera`. Instead of using the mean to compute TTC, the median is used and is therefore less affected by outliers.
```cpp
TTC = (-1.0 / frameRate) / (1 - medianDistRatio);
```

### Performance evaluation, lidar outliers
A couple of examples of where the TTC of the Lidar sensor does not seem plausible are determined based on estimating the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points. As the images below show, there are some points that were not picked up in the previous frame that are closer to the ego car than the preceding vehicle actually is. I believe these points are why TTC from the Lidar sensor in these cases are smaller than expected. Lidar TTC estimates are recorded in FP5.csv.

![Lidar Outlier Example 1](lidar_outlier_01.png)
*Lidar Outlier Example 1*

![Lidar Outlier Example 2](lidar_outlier_02.png)
*Lidar Outlier Example 2*

### Performance evaluation, detector/descriptor combinations
All detector/descriptor combinations were implemented and the camera-based TTC estimates are recorded in FP6.csv. The data and plot of cameras-based TTC are given in the first sheet. 





Based on the data, it seems like any combinations with ORB as the detector perform the worst. Those have a number of instances where the TTC is -inf and many cases where TTC is very large in the negative direction. With respect to estimating minimum TTC, the best combinations include the following detector/descriptor pairs: Harris/SIFT, Harris/ORB, HARRIS/FREAK. It seems when Harris detector is used the best estimates for TTC are possible. Some examples where the camera-based TTC estimation is inaccurate are shown below.

![Camera Outlier Example 1](camera_outlier_01.png)
*Camera Outlier Example 1*

![Camera Outlier Example 2](camera_outlier_02.png)
*Camera Outlier Example 2*