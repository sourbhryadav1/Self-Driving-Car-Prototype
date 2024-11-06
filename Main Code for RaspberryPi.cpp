#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <wiringPi.h>

using namespace std;
using namespace cv;

// Image Processing variables
Mat frame, Matrix, framePers, frameGray, frameThresh, frameEdge, frameFinal, frameFinalDuplicate, frameFinalDuplicate1;
Mat ROILane, ROILaneEnd;
int LeftLanePos, RightLanePos, frameCenter, laneCenter, Result, laneEnd;

VideoCapture Camera;   // USe raspiCam_Cv if you are using picamera 

stringstream ss;

vector<int> histogramLane;
vector<int> histogramLaneEnd;

Point2f Source[] = {Point2f(40,135), Point2f(360,135), Point2f(0,185), Point2f(400,185)};         // change these variables acc to your frame, this is to select your region of interest ROI
Point2f Destination[] = {Point2f(100,0), Point2f(280,0), Point2f(100,240), Point2f(280,240)};     // same goes here , this is for slescting a particualr area of lanes from your ROI

CascadeClassifier Stop_Cascade, Object_Cascade;
Mat frame_Stop, RoI_Stop, gray_Stop, frame_Object, RoI_Object, gray_Object;
vector<Rect> Stop, Object;
int dist_Stop, dist_Object;


 
void Setup()
{
    Camera.open(0); // Open the default camera (webcam)
    if (!Camera.isOpened())
    {
        cerr << "Error: Could not open the camera." << endl;
        exit(1);
    }
    Camera.set(CAP_PROP_FRAME_WIDTH, 400);
    Camera.set(CAP_PROP_FRAME_HEIGHT, 240);
    Camera.set(CAP_PROP_BRIGHTNESS, 50);
    Camera.set(CAP_PROP_CONTRAST, 50);
    Camera.set(CAP_PROP_SATURATION, 50);
    Camera.set(CAP_PROP_GAIN, 50);
    Camera.set(CAP_PROP_FPS, 30);
}

void Capture()
{
    Camera.grab();
    Camera.retrieve( frame);
    cvtColor(frame, frame_Stop, COLOR_BGR2RGB);
    cvtColor(frame, frame_Object, COLOR_BGR2RGB);
    cvtColor(frame, frame, COLOR_BGR2RGB);
    
}

void Perspective()
{
	line(frame,Source[0], Source[1], Scalar(0,0,255), 2);
	line(frame,Source[1], Source[3], Scalar(0,0,255), 2);
	line(frame,Source[3], Source[2], Scalar(0,0,255), 2);
	line(frame,Source[2], Source[0], Scalar(0,0,255), 2);
	
	
	Matrix = getPerspectiveTransform(Source, Destination);
	warpPerspective(frame, framePers, Matrix, Size(400,240));
}

void Threshold()
{
	cvtColor(framePers, frameGray, COLOR_RGB2GRAY);
	inRange(frameGray, 230, 255, frameThresh);
	Canny(frameGray,frameEdge, 900, 900, 3, false);
	add(frameThresh, frameEdge, frameFinal);
	cvtColor(frameFinal, frameFinal, COLOR_GRAY2RGB);
	cvtColor(frameFinal, frameFinalDuplicate, COLOR_RGB2BGR);   //used in histogram function only
	cvtColor(frameFinal, frameFinalDuplicate1, COLOR_RGB2BGR);   //used in histogram function only
	
}

void Histogram()
{
    histogramLane.resize(400);
    histogramLane.clear();
    
    for(int i=0; i<400; i++)       //frame.size().width = 400
    {
	ROILane = frameFinalDuplicate(Rect(i,140,1,100));
	divide(255, ROILane, ROILane);
	histogramLane.push_back((int)(sum(ROILane)[0])); 
    }
	
	histogramLaneEnd.resize(400);
        histogramLaneEnd.clear();
	for (int i = 0; i < 400; i++)       
	{
		ROILaneEnd = frameFinalDuplicate1(Rect(i, 0, 1, 240));   
		divide(255, ROILaneEnd, ROILaneEnd);       
		histogramLaneEnd.push_back((int)(sum(ROILaneEnd)[0]));  
		
	
	}
	   laneEnd = sum(histogramLaneEnd)[0];
	   cout<<"Lane END = "<<laneEnd<<endl;
}

void LaneFinder()
{
    vector<int>:: iterator LeftPtr;
    LeftPtr = max_element(histogramLane.begin(), histogramLane.begin() + 150);
    LeftLanePos = distance(histogramLane.begin(), LeftPtr); 
    
    vector<int>:: iterator RightPtr;
    RightPtr = max_element(histogramLane.begin() +250, histogramLane.end());
    RightLanePos = distance(histogramLane.begin(), RightPtr);
    
    line(frameFinal, Point2f(LeftLanePos, 0), Point2f(LeftLanePos, 240), Scalar(0, 255,0), 2);
    line(frameFinal, Point2f(RightLanePos, 0), Point2f(RightLanePos, 240), Scalar(0,255,0), 2); 
}

void LaneCenter()
{
    laneCenter = (RightLanePos-LeftLanePos)/2 +LeftLanePos;
    frameCenter = 188;
    
    line(frameFinal, Point2f(laneCenter,0), Point2f(laneCenter,240), Scalar(0,255,0), 3);
    line(frameFinal, Point2f(frameCenter,0), Point2f(frameCenter,240), Scalar(255,0,0), 3);

    Result = laneCenter-frameCenter;
}


void Stop_detection()
{
    if(!Stop_Cascade.load("//home//pi//Desktop//MACHINE_LEARNING//Stop_cascade.xml"))  // this stop cascade file consist of trained data for styop detection , you can use more faster methods as it was processsing slower or you can go for yoputube tutorials
    {
	printf("Unable to open stop cascade file");
    }
    
    RoI_Stop = frame_Stop(Rect(200,0,200,140));
    cvtColor(RoI_Stop, gray_Stop, COLOR_RGB2GRAY);
    equalizeHist(gray_Stop, gray_Stop);
    Stop_Cascade.detectMultiScale(gray_Stop, Stop);
    
    for(int i=0; i<Stop.size(); i++)
    {
	Point P1(Stop[i].x, Stop[i].y);                                                       // p1 , p2 arre the end points of the stop sign   
	Point P2(Stop[i].x + Stop[i].width, Stop[i].y + Stop[i].height);                   
	
	rectangle(RoI_Stop, P1, P2, Scalar(0, 0, 255), 2);
	putText(RoI_Stop, "Stop Sign", P1, FONT_HERSHEY_PLAIN, 1,  Scalar(0, 0, 255, 255), 2);
	dist_Stop = (-1.07)*(P2.x-P1.x) + 102.597;
	
       ss.str(" ");
       ss.clear();
       ss<<"D = "<<dist_Stop<<"cm";
       putText(RoI_Stop, ss.str(), Point2f(1,130), 0,1, Scalar(0,0,255), 2);
	
    }
    
}


void Object_detection()
{
    if(!Object_Cascade.load("//home//pi//Desktop//MACHINE_LEARNING//Object_cascade.xml"))
    {
	printf("Unable to open Object cascade file");
    }
    
    RoI_Object = frame_Object(Rect(100,50,200,190));
    cvtColor(RoI_Object, gray_Object, COLOR_RGB2GRAY);
    equalizeHist(gray_Object, gray_Object);
    Object_Cascade.detectMultiScale(gray_Object, Object);
    
    for(int i=0; i<Object.size(); i++)
    {
	Point P1(Object[i].x, Object[i].y);
	Point P2(Object[i].x + Object[i].width, Object[i].y + Object[i].height);
	
	rectangle(RoI_Object, P1, P2, Scalar(0, 0, 255), 2);
	putText(RoI_Object, "Object", P1, FONT_HERSHEY_PLAIN, 1,  Scalar(0, 0, 255, 255), 2);
	dist_Object = (-0.48)*(P2.x-P1.x) + 56.6;
	
       ss.str(" ");
       ss.clear();
       ss<<"D = "<<dist_Object<<"cm";
       putText(RoI_Object, ss.str(), Point2f(1,130), 0,1, Scalar(0,0,255), 2);
	
    }
    
}


int main(int argc,char **argv)
{
	
    wiringPiSetup();
    pinMode(21, OUTPUT);       // gpio pins of raspberryPi 3 model B  -- connected to 1,2,3,4 pins of arduino
    pinMode(22, OUTPUT);
    pinMode(23, OUTPUT);
    pinMode(24, OUTPUT);
    
    Setup(argc, argv, Camera);
    cout<<"Connecting to camera"<<endl;
    if (!Camera.open())
    {
		
	cout<<"Failed to Connect"<<endl;
    }
     
	cout<<"Camera Id = "<<Camera.getId()<<endl;
     
 
    while(1)
    {
	
    auto start = std::chrono::system_clock::now();

    Capture();
    Perspective();
    Threshold();
    Histogram();
    LaneFinder();
    LaneCenter();
    Stop_detection();
    Object_detection();
    
    if (dist_Stop > 5 && dist_Stop < 20)
    {
	digitalWrite(21, 0);
	digitalWrite(22, 0);    //decimal = 8
	digitalWrite(23, 0);
	digitalWrite(24, 1);
	cout<<"Stop Sign"<<endl;
	dist_Stop = 0;
	
	goto Stop_Sign;
    }
    
        if (dist_Object > 5 && dist_Object < 20)
    {
	digitalWrite(21, 1);
	digitalWrite(22, 0);    //decimal = 9
	digitalWrite(23, 0);
	digitalWrite(24, 1);
	cout<<"Object"<<endl;
	dist_Object = 0;
	
	goto Object;
    }
    
 
    
    if (laneEnd > 3000)
    {
       	digitalWrite(21, 1);
	digitalWrite(22, 1);    //decimal = 7
	digitalWrite(23, 1);
	digitalWrite(24, 0);
	cout<<"Lane End"<<endl;
    }
    
    
    if (Result == 0)
    {
	digitalWrite(21, 0);
	digitalWrite(22, 0);    //decimal = 0
	digitalWrite(23, 0);
	digitalWrite(24, 0);
	cout<<"Forward"<<endl;
    }
    
        
    else if (Result >0 && Result <10)
    {
	digitalWrite(21, 1);
	digitalWrite(22, 0);    //decimal = 1
	digitalWrite(23, 0);
	digitalWrite(24, 0);
	cout<<"Right1"<<endl;
    }
    
        else if (Result >=10 && Result <20)
    {
	digitalWrite(21, 0);
	digitalWrite(22, 1);    //decimal = 2
	digitalWrite(23, 0);
	digitalWrite(24, 0);
	cout<<"Right2"<<endl;
    }
    
        else if (Result >20)
    {
	digitalWrite(21, 1);
	digitalWrite(22, 1);    //decimal = 3
	digitalWrite(23, 0);
	digitalWrite(24, 0);
	cout<<"Right3"<<endl;
    }
    
        else if (Result <0 && Result >-10)
    {
	digitalWrite(21, 0);
	digitalWrite(22, 0);    //decimal = 4
	digitalWrite(23, 1);
	digitalWrite(24, 0);
	cout<<"Left1"<<endl;
    }
    
        else if (Result <=-10 && Result >-20)
    {
	digitalWrite(21, 1);
	digitalWrite(22, 0);    //decimal = 5
	digitalWrite(23, 1);
	digitalWrite(24, 0);
	cout<<"Left2"<<endl;
    }
    
        else if (Result <-20)
    {
	digitalWrite(21, 0);
	digitalWrite(22, 1);    //decimal = 6
	digitalWrite(23, 1);
	digitalWrite(24, 0);
	cout<<"Left3"<<endl;
    }
    
    Stop_Sign:
    Object:
    
    
   if (laneEnd > 3000)
    {
       ss.str(" ");
       ss.clear();
       ss<<" Lane End";
       putText(frame, ss.str(), Point2f(1,50), 0,1, Scalar(255,0,0), 2);
    
     }
    
    else if (Result == 0)
    {
       ss.str(" ");
       ss.clear();
       ss<<"Result = "<<Result<<" (Move Forward)";
       putText(frame, ss.str(), Point2f(1,50), 0,1, Scalar(0,0,255), 2);
    
     }
    
    else if (Result > 0)
    {
       ss.str(" ");
       ss.clear();
       ss<<"Result = "<<Result<<" (Move Right)";
       putText(frame, ss.str(), Point2f(1,50), 0,1, Scalar(0,0,255), 2);
    
     }
     
     else if (Result < 0)
    {
       ss.str(" ");
       ss.clear();
       ss<<"Result = "<<Result<<" (Move Left)";
       putText(frame, ss.str(), Point2f(1,50), 0,1, Scalar(0,0,255), 2);
    
     }
    
    
    namedWindow("orignal", WINDOW_KEEPRATIO);
    moveWindow("orignal", 0, 100);
    resizeWindow("orignal", 640, 480);
    imshow("orignal", frame);
    
    namedWindow("Perspective", WINDOW_KEEPRATIO);
    moveWindow("Perspective", 640, 100);
    resizeWindow("Perspective", 640, 480);
    imshow("Perspective", framePers);
    
    namedWindow("Final", WINDOW_KEEPRATIO);
    moveWindow("Final", 1280, 100);
    resizeWindow("Final", 640, 480);
    imshow("Final", frameFinal);
    
    namedWindow("Stop Sign", WINDOW_KEEPRATIO);
    moveWindow("Stop Sign", 1280, 580);
    resizeWindow("Stop Sign", 640, 480);
    imshow("Stop Sign", RoI_Stop);
    
    namedWindow("Object", WINDOW_KEEPRATIO);
    moveWindow("Object", 640, 580);
    resizeWindow("Object", 640, 480);
    imshow("Object", RoI_Object);
    
    
    waitKey(1);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    
    float t = elapsed_seconds.count();
    int FPS = 1/t;
    cout<<"FPS = "<<FPS<<endl;
    
    }

    
    return 0;
     
}

// still you can google or go for youtube tutorials