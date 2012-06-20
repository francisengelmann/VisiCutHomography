//
//  main.cpp
//  VisiCutHomography
//
//  Created by Francis Engelmann on 6/8/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp> 

using namespace cv;
using namespace std;

void display(Mat input);
void detectMarkers(Mat input, Point origin);

std::vector<Vec3f> markers;

int main(int argc, const char * argv[]){
  //std::cout << getBuildInformation();
  VideoCapture cap(0);
  Mat inputImage;
  Mat gray;
  Mat gray2;
  Mat hsvImage;
  Mat edges;
  std::vector<Mat> channels;
  std::vector<Vec3f> circles;
  
  std::vector<Point2f> srcPoints;
  std::vector<Point2f> dstPoints;
  
  float x = 150;
  float y = 180;
  float width = 63*20;
  float height = 30*20;
  Point2f p1 = Point2f(x,y);
  Point2f p2 = Point2f(x+width,y);
  Point2f p3 = Point2f(x,y+height);
  Point2f p4 = Point2f(x+width,y+height);
  
  dstPoints.push_back(p2);
  dstPoints.push_back(p1);
  dstPoints.push_back(p4);
  dstPoints.push_back(p3);  
  
  for(;;){
    srcPoints.clear();
    markers.clear();
    cap >> inputImage;
    //inputImage = imread("photo3.jpg");
    flip(inputImage, inputImage, 1);
    if(!inputImage.data){
      std::cout << "No image read from camera! Is it connected/available? Exiting." << std::endl;
      exit(0);
    }else{
      std::cout << inputImage.size().width << "x" << inputImage.size().height << std::endl;
    }
    
    //resize the camera image to the expected size
    //if smaller image is provided by lets say an unexpected webcam,
    //it would crash when trying to access pixels from out of range
    if(inputImage.size().width!=1620 || inputImage.size().height!=1080){
      std::cout << "Resizing because input image does not match the expected size!" <<endl<< "Maybe wrong camera attached?" << endl;
      resize(inputImage, inputImage, Size(1620,1080), 0, 0, INTER_LINEAR);
    }
    
    Point roiOrigin1 = Point(100,160);
    Mat roi1 = inputImage(Rect(100,160,200,200));
    rectangle(inputImage, Point(100,160), Point(300,360), Scalar(0,255,0), 2);
    
    Point roiOrigin2 = Point(1320,160);
    Mat roi2 = inputImage(Rect(1320,160,200,200));
    rectangle(inputImage, Point(1320,160), Point(1520,360), Scalar(0,255,0), 2);
    
    Point roiOrigin3 = Point(130,850);
    Mat roi3 = inputImage(Rect(130,850,200,200));
    rectangle(inputImage, Point(130,850), Point(330,1050), Scalar(0,255,0), 2);
    
    Point roiOrigin4 = Point(1320,850);
    Mat roi4 = inputImage(Rect(1320,850,200,200));
    rectangle(inputImage, Point(1320,850), Point(1520,1050), Scalar(0,255,0), 2);

    display(inputImage);
    detectMarkers(roi1, roiOrigin1);
    detectMarkers(roi2, roiOrigin2);
    detectMarkers(roi3, roiOrigin3);
    detectMarkers(roi4, roiOrigin4);

    for(int i=0;i<markers.size();i++){
      //circle(inputImage, Point(markers[i][0],markers[i][1]),markers[i][2],Scalar(100,0,255),3);
      circle(inputImage, Point(markers[i][0],markers[i][1]),5,Scalar(100,0,255),3);
      cout << "("<<markers[i][0]<<"," << markers[i][1] << ") r=" <<markers[i][2] << endl;
    }

    //threshold(gray,gray,50,255,THRESH_BINARY);
    //std::vector< std::vector<Point> > contours;
    //findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //cv::Mat result(gray.size(),CV_8U,cv::Scalar(255));
    /*drawContours(result,contours, -1, cv::Scalar(0), 2); 
    for(int i=0;i<contours.size();i++){
      float radius;
      Point2f center;
      minEnclosingCircle(Mat(contours[i]), center, radius);
      srcPoints.push_back(center);
      circle(inputImage, center, radius,Scalar(0),2);
    }*/
    //std::cout << contours.size();
    
    //setImageROI(inputImage, Rect(10, 15, 150, 250));
  
    
    
    display(inputImage);
    //it total 4 markers should be recognized, one for each corner
    
    
    if(markers.size()==4){
      srcPoints.push_back(Point(markers[0][0],markers[0][1]));
      srcPoints.push_back(Point(markers[1][0],markers[1][1]));
      srcPoints.push_back(Point(markers[2][0],markers[2][1]));
      srcPoints.push_back(Point(markers[3][0],markers[3][1]));
      Mat H = findHomography(Mat(srcPoints), Mat(dstPoints), CV_RANSAC);
      warpPerspective(inputImage, inputImage, H, inputImage.size() );
    }

    rectangle(inputImage, p1, p4, Scalar(255,100,0), 2);

    display(inputImage);

    resize(inputImage, inputImage, Size(810,540), 0, 0, INTER_LINEAR);
    //resize(input, blub, Size(800,600), 0, 0, INTER_LINEAR);
    imshow("My Image", inputImage);
    //waitKey();

    //Mat result(gray.size(), CV_8U, Scalar(255));
    //drawContours(result, contours, -1, Scalar(0),2);
    
    
    
    //Mat centers;
    //kmeans(gray, 4, , 4, 1, KMEANS_PP_CENTERS, centers);
    
    //GaussianBlur(channels[2], channels[2], Size(9,9), 2,2);
    //HoughCircles(channels[2], circles, CV_HOUGH_GRADIENT, 2, channels[2].rows/4);
    //cvtColor(inputImage, gray, CV_BGR2GRAY);
    //HoughCircles(channels[2], circles, CV_HOUGH_GRADIENT, 2, channels[2].rows/4);
    //for(int i=0; i<circles.size(); i++){
    //  circle(inputImage, Point2f(circles[i][0] ,circles[i][1]), 3, CV_RGB(255,0,0), 2, 8, 0);
    //}
    //imshow("My Image", channels[2]);

  //}
}

void detectMarkers(Mat roi, Point origin){
    Mat gray;
    cvtColor(roi, gray, CV_BGR2GRAY);
    GaussianBlur(gray, gray, Size(3,3), 1.5);
    adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 55, 2);
    
    dilate(gray, gray, Mat());

    erode(gray,gray,Mat());    
    erode(gray,gray,Mat());    
    dilate(gray, gray, Mat());
    dilate(gray, gray, Mat());
    erode(gray,gray,Mat());
    dilate(gray, gray, Mat());
    
    //Canny(gray, gray, 100, 200, 3);
    display(gray);
    std::vector<Vec3f> circles;
    double minDist = 10;
    double cannyThreshold = 50;
    double minVotes = 15;
    double minRad = 5;
    double maxRad = 25;
    HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, minDist, cannyThreshold, minVotes, minRad, maxRad);

    for(int i=0;i<circles.size();i++){
      circle(gray, Point(circles[i][0],circles[i][1]),circles[i][2],Scalar(100),3);
      //cout << "("<<circles[i][0]<<"," << circles[i][1] << ") r=" <<circles[i][2] << endl;
    }
    //here we add the best detected circle (index 0) to the main list of markers (which should contain 4 in the end)
    
    if(circles.size()<1){
      std::cout << "No marker detected for this corner!" << endl;
      return;
    }
    
    Vec3f marker;
    marker[0] = circles[0][0]+origin.x;
    marker[1] = circles[0][1]+origin.y;
    marker[2] = circles[0][2];
    markers.push_back(marker);
    display(gray);
}

void display(Mat input){
  //return;
  Mat src;
  input.copyTo(src);
  Mat blub;
  resize(input, blub, Size(810,540), 0, 0, INTER_LINEAR);
  //resize(input, blub, Size(800,600), 0, 0, INTER_LINEAR);
  imshow("My Image", blub);
  waitKey(0);
}

int main2(int argc, const char * argv[]){
  std::cout << getBuildInformation();
  VideoCapture cap(0);
  Mat inputImage;
  Mat gray;
  Mat gray2;
  Mat hsvImage;
  Mat corners;
  std::vector<Mat> channels;
  std::vector<Vec3f> circles;
  
  std::vector<Point2f> srcPoints;
  std::vector<Point2f> dstPoints;
  
  float x = 50;
  float y = 50;
  float width = 63*29;
  float height = 30*29;
  Point2f p1 = Point2f(x,y);
  Point2f p2 = Point2f(x+width,y);
  Point2f p3 = Point2f(x,y+height);
  Point2f p4 = Point2f(x+width,y+height);
  
  dstPoints.push_back(p3);
  dstPoints.push_back(p4);
  dstPoints.push_back(p1);  
  dstPoints.push_back(p2);
  
  for(;;){
    srcPoints.clear();
    //cap >> inputImage;
    inputImage = imread("photo3.jpg");
    //resize(inputImage, inputImage, Size(640,480), 0, 0, INTER_LINEAR);
    
    bool patternFound = findChessboardCorners(inputImage, cvSize(9,14) , corners);
    drawChessboardCorners(inputImage, cvSize(9,14), corners, patternFound);

    /*cvtColor(inputImage, hsvImage, CV_BGR2HSV,3);
    //for(int i=0;i<50;i++){
      //inRange(hsvImage, Scalar(i,25,25), Scalar(1+i,255,255), gray);
      //std::cout << "i=" << i << std::endl;
      inRange(hsvImage, Scalar(9,25,25), Scalar(11,255,255), gray);
      erode(gray,gray,Mat());
      erode(gray,gray,Mat());
      erode(gray,gray,Mat());
      erode(gray,gray,Mat());
      dilate(gray, gray, Mat());
      dilate(gray, gray, Mat());
      dilate(gray, gray, Mat());
      dilate(gray, gray, Mat());
      dilate(gray, gray, Mat());
      dilate(gray, gray, Mat());
      dilate(gray, gray, Mat());
      resize(gray, gray, Size(1024,768), 0, 0, INTER_LINEAR);
      imshow("My Image", gray);
      waitKey(0);
    //}*/
        
    split(inputImage,channels);
    subtract(channels[2], channels[0], channels[2]);
    subtract(channels[2], channels[1], channels[2]);
    threshold(channels[2], channels[2], 30, 255, THRESH_BINARY);
    gray=channels[2];
    
    std::vector< std::vector<Point> > contours;
    findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    for(int i=0;i<contours.size();i++){
      float radius;
      Point2f center;
      minEnclosingCircle(Mat(contours[i]), center, radius);
      srcPoints.push_back(center);
      circle(inputImage, center, radius,Scalar(0),2);
    }
    
    if(srcPoints.size()==4){
      Mat H = findHomography(Mat(srcPoints), Mat(dstPoints), CV_RANSAC);
      warpPerspective(inputImage, inputImage, H, inputImage.size() );
    }
    
    //Mat result(gray.size(), CV_8U, Scalar(255));
    //drawContours(result, contours, -1, Scalar(0),2);
    
    
    
    //Mat centers;
    //kmeans(gray, 4, , 4, 1, KMEANS_PP_CENTERS, centers);
    
    //GaussianBlur(channels[2], channels[2], Size(9,9), 2,2);
    //HoughCircles(channels[2], circles, CV_HOUGH_GRADIENT, 2, channels[2].rows/4);
    //cvtColor(inputImage, gray, CV_BGR2GRAY);
    //HoughCircles(channels[2], circles, CV_HOUGH_GRADIENT, 2, channels[2].rows/4);
    //for(int i=0; i<circles.size(); i++){
    //  circle(inputImage, Point2f(circles[i][0] ,circles[i][1]), 3, CV_RGB(255,0,0), 2, 8, 0);
    //}
    //imshow("My Image", channels[2]);
    line(inputImage, p4, p3, Scalar(255,255,0));
    line(inputImage, p1, p2, Scalar(255,255,0));
    line(inputImage, p1, p3, Scalar(255,255,0));
    line(inputImage, p4, p2, Scalar(255,255,0));

    display(inputImage);
  }
  
}

int working(int argc, const char * argv[]){
  //there seems to be a problem with OpenCV 2.4 when reading from an image
  Mat inputImage = imread("img.jpg");
  namedWindow("My Image");
  imshow("My Image", inputImage);
  waitKey(0);
  
  std::vector<Point2f> srcPoints;
  std::vector<Point2f> dstPoints;
  
  srcPoints.push_back( Point2f(200,230) );
  srcPoints.push_back( Point2f(450,220) );
  srcPoints.push_back( Point2f(160,380) );
  srcPoints.push_back( Point2f(460,360) );

  dstPoints.push_back(Point2f(120,120));
  dstPoints.push_back(Point2f(220,120));
  dstPoints.push_back(Point2f(120,220));
  dstPoints.push_back(Point2f(220,220));
  
  for(int i=0; i<srcPoints.size(); i++){
    circle(inputImage, srcPoints[i], 3, CV_RGB(255,0,0), 2, 8, 0);
  }
  imshow("My Image", inputImage);
  waitKey(0);
  
  Mat H = findHomography(Mat(srcPoints), Mat(dstPoints), CV_RANSAC);
  warpPerspective(inputImage, inputImage, H, inputImage.size() );
  imshow("My Image", inputImage);
  waitKey(0);
}


int mainBla(int argc, const char * argv[])
{
  VideoCapture cap(0);

  Mat image = imread("eva.jpg");
  Mat result;
  
  image.copyTo(result);
  
  //(1024,768,CV_8UC3,cv::Scalar(100));
  
  std::vector<Point2f> srcPoints;
  std::vector<Point2f> dstPoints;

  srcPoints.push_back( Point2f(100,100) );
  srcPoints.push_back( Point2f(200,100) );
  srcPoints.push_back( Point2f(70,200) );
  srcPoints.push_back( Point2f(130,200) );
  
  dstPoints.push_back(Point2f(200,200));
  dstPoints.push_back(Point2f(400,200));
  dstPoints.push_back(Point2f(200,400));
  dstPoints.push_back(Point2f(400,400));
  
  namedWindow("My Image");
  //for(;;){
    cap >> image;
    //resize(image, image, Size(640,480), 0, 0, INTER_LINEAR);
    //Mat H(3,3,CV_32F,Scalar(2));
    
    //SiftFeatureDetector detector;
    //std::vector<cv::KeyPoint> keypoints;
    //detector.detect(image, keypoints);
    
    for(int i=0; i<srcPoints.size(); i++){
      circle(image, srcPoints[i], 3, CV_RGB(255,0,0), 2, 8, 0);
    }
    
    Mat H = findHomography(Mat(srcPoints), Mat(dstPoints), CV_RANSAC);
    
    std::cout << "H.size() = (" << H.rows << " " << H.cols <<")";

    //drawKeypoints(image, keypoints, result);
    
    warpPerspective(image, result, H, image.size() );
    
    //GaussianBlur(image, image, Size(7,7), 3.0, 3.0);
    //Canny(image,image,0,30,3);
    imshow("My Image", result);
  //}
  waitKey(0);
  imwrite("image.jpg", image);
  // insert code here...
  std::cout << "Hello, World!\n";
    return 0;
}

