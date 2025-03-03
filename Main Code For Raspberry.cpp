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

vector<int> histogramLane;
vector<int> histogramLaneEnd;

VideoCapture Camera(0);

stringstream ss;


              //    upper left
Point2f Source[] = {Point2f(35,170), Point2f(275,170), Point2f(05,205), Point2f(310,205)};
Point2f Destination[] = {Point2f(100,0), Point2f(300,0), Point2f( 100,280), Point2f(300,280)};


//Machine Learning variables
CascadeClassifier Stop_Cascade;
Mat frame_Stop, RoI_Stop, gray_Stop;
vector<Rect> Stop;

  
void Setup() {
    Camera.set(CAP_PROP_FRAME_WIDTH, 400);
    Camera.set(CAP_PROP_FRAME_HEIGHT, 280);
    Camera.set(CAP_PROP_BRIGHTNESS, 30);
    Camera.set(CAP_PROP_CONTRAST, 20);
    Camera.set(CAP_PROP_SATURATION, 100);
    Camera.set(CAP_PROP_FPS, 0);
    
    if(!Stop_Cascade.load("/home/pi/Desktop/MACHINE_LEARNING/Stop_cascade.xml")){
        cout<<"error loading stop sign casacade"<<endl;
    }
}

void Perspective() {
    line(frame, Source[0], Source[1], Scalar(0,255,0),2);
    line(frame, Source[1], Source[3], Scalar(0,255,0),2);
    line(frame, Source[3], Source[2], Scalar(0,255,0),2);
    line(frame, Source[2], Source[0], Scalar(0,255,0),2);

    Matrix = getPerspectiveTransform(Source, Destination);
    warpPerspective(frame, framePers, Matrix, Size(400,280));
    
    Mat mask = Mat::zeros(framePers.size(),CV_8UC1);
    
    vector<Point> regionOfInterest = {
        Point(100,0),
        Point(300,0),
        Point(300,280),
        Point(100,280)
    };
    
    fillConvexPoly(mask, regionOfInterest, Scalar(255));
    
    framePers.setTo(Scalar(255,255,255), ~mask);
    
}
void Capture() {
    Camera >> frame;
    frame.copyTo(frame_Stop);
}



void Threshold() {
    cvtColor(framePers, frameGray, COLOR_BGR2GRAY);
    inRange(frameGray, 0,120, frameThresh);
    Canny(frameGray, frameEdge, 500, 500, 3, false);
    add(frameThresh, frameEdge, frameFinal); 
    cvtColor(frameFinal, frameFinal, COLOR_GRAY2RGB);
    cvtColor(frameFinal, frameFinalDuplicate, COLOR_RGB2BGR);
    cvtColor(frameFinal, frameFinalDuplicate1, COLOR_RGB2BGR);
}

void Histogram(){
	histogramLane.clear();
	histogramLane.resize(400,0);
	
	
	for(int i=0; i<400; i++){
		ROILane = frameFinalDuplicate(Rect(i,160,1,80));               // ROI- region of interest for final frame of which we have choose rectangular part
		histogramLane[i] = (sum(ROILane)[0] / 3);             // store intesnsity of each column
		}
        
    histogramLaneEnd.clear();
	histogramLaneEnd.resize(400,0);
	
	
	for(int i=0; i<400; i++){
		ROILaneEnd = frameFinalDuplicate1(Rect(i,0,1,240));               // ROI- region of interest for final frame of which we have choose rectangular part
		histogramLaneEnd[i] = (sum(ROILaneEnd)[0] / 3);             // store intesnsity of each column
		}
        
    laneEnd = sum(histogramLaneEnd)[0];
    cout<<"Lane End ="<<laneEnd<<endl;
    
}



void LaneFinder() {
    vector<int>::iterator LeftPtr = max_element(histogramLane.begin(), histogramLane.begin() + 150);
    LeftLanePos = distance(histogramLane.begin(), LeftPtr); 

    vector<int>::iterator RightPtr = max_element(histogramLane.begin() + 250, histogramLane.end());
    RightLanePos = distance(histogramLane.begin(), RightPtr);

    line(frameFinal, Point2f(LeftLanePos, 0), Point2f(LeftLanePos, frame.rows), Scalar(0, 255, 0), 2);
    line(frameFinal, Point2f(RightLanePos, 0), Point2f(RightLanePos, frame.rows), Scalar(0, 255, 0), 2); 
}

void LaneCenter() {
    laneCenter = (RightLanePos + LeftLanePos) / 2 ;
    frameCenter = 192;

    line(frameFinal, Point2f(laneCenter,0), Point2f(laneCenter,280), Scalar(0,255,0), 3);
    line(frameFinal, Point2f(frameCenter, 0), Point2f(frameCenter, 280), Scalar(255,0,0), 3); 

    Result = laneCenter - frameCenter;
}



void Stop_detection() {
    if(!Stop_Cascade.load("/home/pi/Desktop/MACHINE_LEARNING/Stop_cascade.xml")) {
        printf("Unable to open stop cascade file");
    }
    
    if (frame_Stop.cols >= 200 && frame_Stop.rows >= 280) {  // Ensure frame is large enough
        RoI_Stop = frame_Stop(Rect(200, 0, 200, 280));
    } else {
        RoI_Stop = frame_Stop;  // Use entire frame if dimensions are smaller
    }

    cvtColor(RoI_Stop, gray_Stop, COLOR_RGB2GRAY);
    equalizeHist(gray_Stop, gray_Stop);
    Stop_Cascade.detectMultiScale(gray_Stop, Stop);

    for(int i = 0; i < Stop.size(); i++) {
        Point P1(Stop[i].x, Stop[i].y);
        Point P2(Stop[i].x + Stop[i].width, Stop[i].y + Stop[i].height);

        rectangle(RoI_Stop, P1, P2, Scalar(0, 0, 255), 2);
        putText(RoI_Stop, "Stop Sign", P1, FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255, 255), 2);
        
        ss.str(" ");
            ss.clear();
            ss<< "Distance ="<<P2.x - P1.x<<"Pixels";
            putText(RoI_Stop, ss.str(), Point2f(1,130),0,1,Scalar(0,0,255),2);
    }
}






int main() {
    wiringPiSetup();
    pinMode(21, OUTPUT);
    pinMode(22, OUTPUT);
    pinMode(23, OUTPUT);
    pinMode(24, OUTPUT);
	
    Setup();
    cout << "Connecting to camera" << endl;
    if (!Camera.isOpened()) {
        cout << "Failed to Connect" << endl;
        return -1;
    }

    cout << "Camera successfully connected." << endl;

    while (1) {
        auto start = chrono::system_clock::now();

		
	
	Capture();
	Perspective();
	Threshold();
	Histogram();
	LaneFinder();
	LaneCenter();
	
    
    
    
           
        if (laneEnd == 0) {
            digitalWrite(21, 1);
            digitalWrite(22, 1);                         // decimal 7
            digitalWrite(23, 1);
            digitalWrite(24, 0);
            cout << "Lane End" << endl; 
            
        } else if (Result == 0) {
            digitalWrite(21, 0);
            digitalWrite(22, 0);                           // decimal = 0
            digitalWrite(23, 0);
            digitalWrite(24, 0);
            cout << "Forward" << endl;
        } else if (Result > 0 && Result < 10) {
            digitalWrite(21, 1);
            digitalWrite(22, 0);
            digitalWrite(23, 0);                         // decimal = 1
            digitalWrite(24, 0);
            cout << "Right1" << endl;
        } else if (Result >= 10 && Result < 20) {
            digitalWrite(21, 0);
            digitalWrite(22, 1);                         // decimal = 2
            digitalWrite(23, 0);
            digitalWrite(24, 0);
            cout << "Right2" << endl;
        } else if (Result > 20) {
            digitalWrite(21, 1);
            digitalWrite(22, 1);                         // decimal = 3
            digitalWrite(23, 0);                           
            digitalWrite(24, 0);
            cout << "Right3" << endl;
        } else if (Result < 0 && Result > -10) {
            digitalWrite(21, 0);
            digitalWrite(22, 0);
            digitalWrite(23, 1);                        // decimal = 4
            digitalWrite(24, 0);
            cout << "Left1" << endl;
        } else if (Result <= -10 && Result > -70) {
            digitalWrite(21, 1);
            digitalWrite(22, 0);
            digitalWrite(23, 1);                       // decimal = 5
            digitalWrite(24, 0);
            cout << "Left2" << endl;
        } else if (Result < -70) {
            digitalWrite(21, 0);
            digitalWrite(22, 1);                       // decimal = 6
            digitalWrite(23, 1);
            digitalWrite(24, 0);
            cout << "Left3" << endl;
        }
        
        if(laneEnd ==0){
            ss.str(" ");
            ss.clear();
            ss<< " Lane End";
            putText(frame, ss.str(), Point2f(10,50),0,1,Scalar(0,0,255),2);
        }else if(Result==0){
            ss.str(" ");
            ss.clear();
            ss<< "Result ="<<Result<<"Move Forward";
            putText(frame, ss.str(), Point2f(1,50),0,1,Scalar(0,0,255),2);
        }
        else if(Result > 0){
            ss.str(" ");
            ss.clear();
            ss<< "Result ="<<Result<<"Move Right";
            putText(frame, ss.str(), Point2f(1,50),0,1,Scalar(0,0,255),2);
        }
        if(Result < 0){
            ss.str(" ");
            ss.clear();
            ss<< "Result =" << Result<<"Move Left";
            putText(frame, ss.str(), Point2f(1,50),0,1,Scalar(0,0,255),2);
        }
            
	
	
	namedWindow("original", WINDOW_KEEPRATIO);
    moveWindow("original", 0, 100);
    resizeWindow("original", 400, 240);
    imshow("original", frame);

    namedWindow("Perspective", WINDOW_KEEPRATIO);
    moveWindow("Perspective", 400, 100);
    resizeWindow("Perspective", 400, 240);
    imshow("Perspective", framePers);
    
    namedWindow("Final", WINDOW_KEEPRATIO);
    moveWindow("Final", 800, 100);
    resizeWindow("Final", 400, 240);
    imshow("Final", frameFinal);
    
    namedWindow("Stop Sign",WINDOW_KEEPRATIO);
    moveWindow("Stop Sign",400,580);
    resizeWindow("Stop Sign",400,240);
    imshow("Stop Sign",RoI_Stop);
    
 
    
        
	 waitKey(1);
        auto end = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = end - start;

        float t = elapsed_seconds.count();
        int FPS = 1 / t;
    }


    return 0;
}
