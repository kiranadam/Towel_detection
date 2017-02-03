/***********************************************************/
/*      Content Based Image Retrival System  C++           */
/*            Author : Kirankumar V. Adam                  */
/***********************************************************/

#include "Towel_detection.hpp"
#include <iostream>
#include <cmath>
#include <boost/range/numeric.hpp>
#include <vector>
#include <omp.h>
#include <algorithm>
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace boost;
using std::vector;
//using boost::assign;
using namespace cv;

// Parametrized Constructor for Towel_detection class
Towel_detection :: Towel_detection()
{
	this->k_size = 21;

	for(int i=0; i<4; i++)
	{
		this->sigma.push_back(0.05*(i+1));
	}

	for(int i=0; i<6; i++)
	{
		this->theta.push_back((30.0/180.0)*CV_PI*i);
	}
	
	this->lambda = 50;
	this->psi = CV_PI/2;
}

// Function for the Gabor Kernel
Mat Towel_detection :: gabor_kernel(int ks, double sig, double th, double lm, double ps)
{
	int hks = (ks-1)/2;
    	double theta = th*CV_PI/180;
    	double psi = ps*CV_PI/180;
    	double del = 2.0/(ks-1);
    	double lmbd = lm;
    	double sigma = sig/ks;
    	double x_theta;
    	double y_theta;

    	Mat kernel(ks,ks, CV_32F);
    
	for (int y=-hks; y<=hks; y++)
    	{
		#pragma omp parallel for 
        		for (int x=-hks; x<=hks; x++)
        		{
            		x_theta = x*del*cos(theta)+y*del*sin(theta);
            		y_theta = -x*del*sin(theta)+y*del*cos(theta);
            		kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        		}
    	}
	
    	return kernel;
}

// Color features calculating function
vector<int> Towel_detection :: color_feature(Mat& src)
{
	Mat hsv_image;
	cvtColor(src,hsv_image,CV_BGR2HSV);
	
	// Image for the quantization
	Mat quant_image = Mat::zeros(hsv_image.rows,hsv_image.cols,CV_8UC1);
	
	// vector for color features with 32x1 bins
	vector<int> col_feat(32,0);
	
	// Color Quantization based on "IMAGE RETRIEVAL USING BOTH COLOR AND TEXTURE FEATURES" FAN-HUI KONG
	for(int i=0;i<hsv_image.rows; i++)
	{
		#pragma omp parallel for
		for(int j=0;j<hsv_image.cols; j++)
		{
			int h = (int) hsv_image.at<Vec3b>(i,j)[0];
			int s = (int) hsv_image.at<Vec3b>(i,j)[1];
			int v = (int) hsv_image.at<Vec3b>(i,j)[2];
			
			int n;
			if(v<=0.1*255)
			{
				n = 0;
			}
			else if((s<=0.1*255)&&(v>0.1*255)&&(v<=0.4*255))
			{
				n = 1;
			}
			else if((s<=0.1*255)&&(v>0.4*255)&&(v<=0.7*255))
			{
				n = 2;
			}
			else if((s<=0.1*255)&&(v>0.7*255)&&(v<=1*255))
			{
				n = 3;
			}
			else if((h>=0.0/360.0*180 && h<=20.0/360.0*180) || (h>330.0/360.0*180 && h<360.0/360.0*180))
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 4;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 5;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 6;
				}
				else
				{
					n = 7;
				}
			}
			else if(h>20.0/360.0*180&&h<=45.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 8;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 9;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 10;
				}
				else
				{
					n = 11;
				}
			}
			else if(h>45.0/360.0*180&&h<=75.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 12;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 13;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 14;
				}
				else
				{
					n = 15;
				}
			}
			else if(h>75.0/360.0*180&&h<=155.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 16;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 17;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 18;
				}
				else
				{
					n = 19;
				}
			}
			else if(h>155.0/360.0*180&&h<=210.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 20;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 21;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 22;
				}
				else
				{
					n = 23;
				}
			}
			else if(h>210.0/360.0*180&&h<=270.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 24;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 25;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 26;
				}
				else
				{
					n =27;
				}
			}
			else
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 28;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 29;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 30;
				}
				else
				{
					n = 31;
				}
			}
			col_feat.at(n) = col_feat.at(n) + 1; 
			quant_image.at<uchar>(i,j) = (uchar)n;
		} 
		
	}
	return col_feat;
}

// Texture feature calculating function
vector<double> Towel_detection :: texture_feature(Mat& src)
{
	// Converting src image to gray.
	Mat gray, gray_image;
	cvtColor(src,gray,CV_BGR2GRAY);
	gray.convertTo(gray_image,CV_32F, 1.0/255, 0);

	// Vector for the texture feature definition
	vector<double> tex_feat;
	
	for(int i = 0; i < theta.size(); i++)
	{
		#pragma omp parallel for
		for(int j = 0; j < sigma.size(); j++)
		{
			Mat kernel = gabor_kernel(k_size, sigma.at(j), theta.at(i), lambda, psi);
			//cout<<"sigma = "<<sigma.at(j)<<"theta = "<<theta.at(i)<<endl;
			
			// Convolution of gray scale image with gabor kernel
			filter2D(gray_image,gray,CV_32F,kernel);
			
			// Calculating mean and standard deviation 
			Scalar t_mu, t_sigma;
			meanStdDev(gray,t_mu,t_sigma);
			
			//cout<<" Iteration value "<<i*theta.size()+j<<endl;
			tex_feat.push_back(t_mu[0]);
			tex_feat.push_back(t_sigma[0]);
		}
	}
	
	return tex_feat;
}

// Image similarity 
double Towel_detection :: image_similarity(Mat& query_image, Mat& db_image)
{
	// Image Similarity based on "A Clustering Based Approach to Efficient Image Retrieval" -R. Zhang, and Z. Zhang
	vector<int> q_col_feat,db_col_feat;
	vector<double> q_tex_feat,db_tex_feat;
	
	// get texture feature for query image and DB image
	q_tex_feat = texture_feature(query_image);
	db_tex_feat = texture_feature(db_image);
	
	double d_t = L2dist_texture(db_tex_feat, q_tex_feat); // L2 distance calculation

	// get color feature for query image and DB image
	q_col_feat = color_feature(query_image);
	db_col_feat = color_feature(db_image);
	
	double d_c = HITdist_color(db_col_feat, q_col_feat);  //HIT distance calculation
	
	double img_sim = 0.35*d_t + 0.65*d_c;
	
	return (1-img_sim);
}

// Euclidean distance calculation for texture feature vectors
double Towel_detection :: L2dist_texture(vector<double>& db_tex_feat, vector<double>& q_tex_feat)
{	
	double tex_dist = 0;
	
	for(unsigned i=0; i <q_tex_feat.size(); i++)
	{
		tex_dist += pow((db_tex_feat.at(i) - q_tex_feat.at(i)), 2.0);
	}
	
	tex_dist = sqrt(tex_dist);
	
	return tex_dist;
}

// Histogram Intersection Technique for color feature
double Towel_detection :: HITdist_color(vector<int>& db_col_feat, vector<int>& q_col_feat)
{
	double col_dist = 0;
	
	for(unsigned i=0; i <q_col_feat.size(); i++)
	{
		col_dist += (double) min(q_col_feat.at(i),db_col_feat.at(i));
	}
	
	col_dist = col_dist/(double)accumulate(q_col_feat,0); 
	
	return (1 - col_dist);
}

double Towel_detection :: SIFT_Registeration(Mat& query_image, Mat& db_image)
{
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(800);
	
	Mat gray_query, gray_db;
	cvtColor(query_image, gray_query, CV_RGB2GRAY);
	cvtColor(db_image, gray_db, CV_RGB2GRAY);

	//-- Step 1: Detect the keypoints:
  	vector<KeyPoint> query_keypoint, db_keypoint;    
  	f2d->detect(gray_query, query_keypoint);
  	f2d->detect(gray_db, db_keypoint);

	//-- Step 2: Calculate descriptors (feature vectors)    
  	Mat query_descriptor, db_descriptor;    
  	f2d->compute(gray_query, query_keypoint, query_descriptor);
  	f2d->compute(gray_db, db_keypoint, db_descriptor);
	
	//-- Step 3: Matching descriptor vectors using BFMatcher or FlannMatcher
  	//BFMatcher matcher;
	FlannBasedMatcher matcher;
  	vector<DMatch> matches;
  	matcher.match(db_descriptor, query_descriptor, matches);
	//cout<<"number of matches (FLANN): "<<matches.size()<<endl;

	//-- Step 4: Draw matches
     	Mat image_matches;
     	drawMatches(gray_db, db_keypoint, gray_query, query_keypoint, matches, image_matches);

	//-- Show detected matches
     	imshow("Matches", image_matches);
	
	//cout<<"Matches Size = "<<matches.size()<<endl;
	waitKey(-1);

	return ((double)matches.size()/(double) 800);

}

double Towel_detection :: towel_damage(Mat& query_image, Mat& db_image)
{
	double sr = SIFT_Registeration(query_image, db_image);
	//cout<<"FLANN matches = "<<sr<<endl;

	double is = image_similarity(query_image, db_image); 
	//cout<<"Image similarity = "<<is<<endl;

	double damage = 1-(0.5 * sr + 0.5 * is);

	return damage;	
}
