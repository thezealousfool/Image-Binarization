#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv)
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
	
	cv::Mat img = cv::imread(file, 0);
	cv::imshow("Original image", img);

	// cv::GaussianBlur(img, img, cv::Size(3,3), 0, 0);
	// cv::imshow("Blurred image", img);

	cv::normalize(img, img, 150, 200, cv::NORM_MINMAX);
	// cv::GaussianBlur(img, img, cv::Size(3,3), 0, 0);
	// cv::imshow("Normalized image", img);
	
	cv::Mat bestLabels, centers, clustered;
	cv::Mat p = img.reshape(1, 1);
	p.convertTo(p, CV_32F);


	cv::kmeans(p, 2, bestLabels, cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 4, 1.0), 2, cv::KMEANS_PP_CENTERS, centers);
	
	bool zeroIsBackground = true;
	uchar center0 = img.at<uchar>(cv::Point((int)(centers.at<float>(0, 0)) % img.cols, (int)(centers.at<float>(0, 0)) / img.cols));
	uchar center1 = img.at<uchar>(cv::Point((int)(centers.at<float>(1, 0)) % img.cols, (int)(centers.at<float>(1, 0)) / img.cols));

	if(center0 < center1)
		zeroIsBackground = false;

	for(std::size_t i = 0; i < img.rows; ++i) {
		for(std::size_t j = 0; j < img.cols; ++j) {
			if(zeroIsBackground) {
				if(bestLabels.at<int>(cv::Point(0, i*img.cols + j)) == 0)
					img.at<uchar>(cv::Point(j, i)) = 255;
				else
					img.at<uchar>(cv::Point(j, i)) = 0;
			}
			else {
				if(bestLabels.at<int>(cv::Point(0, i*img.cols + j)) == 1)
					img.at<uchar>(cv::Point(j, i)) = 255;				
				else
					img.at<uchar>(cv::Point(j, i)) = 0;
			}
		}
	}

	cv::imshow("New image", img);
	std::cout << "Done!\n";

	cv::waitKey(0);
	return 0;
}