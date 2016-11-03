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
	
	for(long long int i = 0; i < img.rows * img.cols; ++i) {
		img.at<char>(i/img.cols, i%img.cols) = char((bestLabels.at<int>(0, i) - 1) * -255);
	}

	cv::imshow("New image", img);
	std::cout << "Done!\n";

	cv::waitKey(0);
	return 0;
}