/************************************/
/*       Testing Program            */
/************************************/

#include <iostream>
#include "Towel_detection.hpp"
#include <ctime>
#include <string>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include "Eserial.h"

using namespace std;
using namespace boost;

// Serial to Arduino global declarations
int arduino_command;
Eserial * arduino_com;
unsigned char Signal;

int main(int , char **)
{
	string db_string;
	cout<<"Enter the image path : "<<endl;
	cin>>db_string;
	
	Mat db_image = imread(db_string,1);
	if(!db_image.data ) // Check for invalid input
    	{
	  	cout<<"Could not open or find the image"<<endl ;
	  	return -1;
	}

	Mat q_image;

	VideoCapture cam(1);  //VideoCpature object with external camera

	if(!cam.isOpened())
	{
		cout<<"Camera not ready"<<endl;
		return -1;
	}

	// serial to Arduino setup 
  	arduino_com = new Eserial ();
  	
	if (arduino_com != 0)
    	{
      	 	arduino_com->connect("/dev/ttyACM0", 9600, spNONE);
    	}
	else
	{
		cout<<"Ardunino not connected"<<endl;
		return -1;
	}
	
	clock_t start,end;
	double elapsed;

	start = clock();

	// create Towel_detection object
	Towel_detection towel;
	
	for(;;)
	{
		Signal = 'H';
		arduino_com->sendChar(Signal);
		end = clock();
		Mat frame;
		cam >> frame;
		
		
		if((float)(end-start)/CLOCKS_PER_SEC >= 1.0) // get new frame after every 1 second
		{
			Signal = 'L';
			arduino_com->sendChar(Signal);
			start=end;
			q_image = frame.clone();	// get image frame from camera

			double damage = towel.towel_damage(q_image, db_image);

			int count = 0;

			if(damage < 0.8)
			{
				count++;
				imwrite(lexical_cast<string>(count)+"_damaged.jpg",q_image);
			}
			else
			{
				count++;
				imwrite(lexical_cast<string>(count)+"_nondamaged.jpg",q_image);
			}				
		}		
	}

	return 0;
}

