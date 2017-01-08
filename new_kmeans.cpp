#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char const *argv[])
{
	const char* keys = {
		"{help h usage | | Help text}"
		"{@image | | Image for filtering}"
	};

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Vivek's Binarizer");

	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	std::string file = parser.get<std::string>(0);

	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	if (file == "") {
		std::cout << "Please supply a file name\n";
		return 0;
	}

	std::cout << "Loading...\n";
	
	cv::Mat img = cv::imread(file);
	cv::imshow("Original image", img);

	cv::cvtColor(img, img, CV_BGR2HSV);

	cv::Mat points(img.rows * img.cols, 1, CV_32FC1);

	for(std::size_t i = 0; i < img.rows; ++i) {
		for(std::size_t j = 0; j < img.cols; ++j) {
			cv::Vec3b color = img.at<cv::Vec3b>(cv::Point(j, i));
			points.at<float>(i*img.cols + j) = color[2];
		}
	}

	const int nclusters = 2;
	const int attempts = 5;

	cv::Mat bestLabels, centers;
	
	cv::kmeans(points, nclusters, bestLabels, cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 4, 1.0), attempts, cv::KMEANS_PP_CENTERS, centers);

	bool zeroIsBackground = true;
	cv::Vec3b center0 = img.at<cv::Vec3b>(cv::Point((int)(centers.at<float>(0, 0)) % img.cols, (int)(centers.at<float>(0, 0)) / img.cols));
	cv::Vec3b center1 = img.at<cv::Vec3b>(cv::Point((int)(centers.at<float>(1, 0)) % img.cols, (int)(centers.at<float>(1, 0)) / img.cols));

	if(center0[2] < center1[2])
		zeroIsBackground = false;
	else if(center0[2] == center1[2]) {
		std::cout << "equal\n";
		if(center0[1] > center1[1])
			zeroIsBackground = false;
	}

	cv::cvtColor(img, img, CV_HSV2BGR);
	
	for(std::size_t i = 0; i < img.rows; ++i) {
		for(std::size_t j = 0; j < img.cols; ++j) {
			if(zeroIsBackground) {
				if(bestLabels.at<int>(cv::Point(0, i*img.cols + j)) == 0)
					img.at<cv::Vec3b>(cv::Point(j, i)) = cv::Vec3b(255, 255, 255);
			}
			else {
				if(bestLabels.at<int>(cv::Point(0, i*img.cols + j)) == 1)
					img.at<cv::Vec3b>(cv::Point(j, i)) = cv::Vec3b(255, 255, 255);				
			}
		}
	}

	cv::imshow("New Image", img);

	std::vector<cv::Mat> channels;
	cv::split(img, channels);
	std::vector<cv::Mat> channels2(3);
	channels2[0] = channels[0];
	channels2[1] = cv::Mat::ones(img.rows, img.cols, CV_8UC1)*255;
	channels2[2] = cv::Mat::ones(img.rows, img.cols, CV_8UC1)*255;
	cv::merge(channels2, img);
	cv::imshow("Hue Image", img);

	cv::waitKey(0);
	return 0;
}