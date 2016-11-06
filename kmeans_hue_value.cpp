#include <iostream>
#include <vector>
#include <cmath>
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
	
	cv::Mat img = cv::imread(file);
	cv::imshow("Original image", img);

	cv::cvtColor(img, img, CV_BGR2HSV);

	std::vector<cv::Mat> chans;
	cv::split(img, chans);
	cv::normalize(chans[2], chans[2], 0, 255, cv::NORM_MINMAX);
	cv::merge(chans, img);
	
	cv::Mat bestLabels, centers, clustered;
	cv::Mat p(img.rows * img.cols, 1, CV_32FC2);

	for(long long int i = 0; i < img.rows * img.cols; ++i) {
		cv::Vec3b color = img.at<cv::Vec3b>(cv::Point(i % img.cols, i / img.cols));
		p.at<cv::Vec2f>(i)[0] = color[0];
		p.at<cv::Vec2f>(i)[1] = color[2];
	}

	int nclusters;
	std::cout << "Clusters: ";
	std::cin >> nclusters;

	std::vector<bool> cluster_fb(nclusters, false);

	cv::kmeans(p, nclusters, bestLabels, cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 4, 1.0), 2, cv::KMEANS_PP_CENTERS, centers);

	cv::Mat new_img(img.rows, img.cols, CV_8U);

	int forground_cno = 0, background_cno = 0;
	for(int i = 1; i < nclusters; ++i) {
		if(centers.at<cv::Vec2f>(i)[1] < centers.at<cv::Vec2f>(forground_cno)[1]) {
			forground_cno = i;
		}
		if(centers.at<cv::Vec2f>(i)[1] > centers.at<cv::Vec2f>(background_cno)[1]) {
			background_cno = i;
		}
	}
	cluster_fb[forground_cno] = true;
	
	{
		double f_dist, b_dist;
		cv::Vec2f fcluster, bcluster;
		fcluster = centers.at<cv::Vec2f>(forground_cno);
		bcluster = centers.at<cv::Vec2f>(background_cno);
		for(int i = 0; i < nclusters; ++i) {
			if (i == forground_cno || i == background_cno)
				continue;
			f_dist = abs(fcluster[0] - centers.at<cv::Vec2f>(i)[0]);
			b_dist = abs(bcluster[0] - centers.at<cv::Vec2f>(i)[0]);
			if (f_dist < b_dist)
				cluster_fb[i] = true;
		}
	}

	for(long long int i = 0; i < img.rows * img.cols; ++i) {
		new_img.at<char>(i/img.cols, i%img.cols) = char(cluster_fb[bestLabels.at<int>(0, i)] ? 0 : 255);
	}
	
	cv::imshow("New image", new_img);
	std::cout << "Done!\n";

	cv::waitKey(0);
	return 0;
}