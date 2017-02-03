/***********************************************************/
/*      Content Based Image Retrival System  Header        */
/*            Author : Kirankumar V. Adam                  */
/***********************************************************/

#ifndef TOWEL_DETECTION_HPP
#define TOWEL_DETECTION_HPP

#include "opencv2/opencv.hpp"
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

class Towel_detection
{
	private :
	    	// variables for the Gabor function for 6 orientations and 4 scales
		int k_size; // kernel size 
		std::vector<double> sigma; //Standard deviation for 4 different scales
		std::vector<double> theta; // Orientations
		double lambda; // Wavelength of sinusoidal factor
		double psi; // Phase offset.		
		
		// function definition for gabor kernel
		Mat gabor_kernel(int ks, double sig, double th, double lm, double ps); 
		
		// Euclidean distance measurement for texture feature
		double L2dist_texture(vector<double>& db_tex_feat, vector<double>& q_tex_feat);
		
		// Histogram Intersection Technique for color feature
		double HITdist_color(vector<int>& db_col_feat, vector<int>& q_col_feat);
		
		// color feature 
		vector <int> color_feature(Mat& src);
		// texture feature
		vector <double> texture_feature(Mat& src);

		// Image similarity calculation 
		double image_similarity(Mat& query_image, Mat& db_image);

		// Image Registeration using SIFT Registeration
		double SIFT_Registeration(Mat& query_image, Mat& db_image);
		
	
	public :
		// Constructor 
		Towel_detection();

		double towel_damage(Mat& query_image, Mat& db_image);		
		
};

#endif 
