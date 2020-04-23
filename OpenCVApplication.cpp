// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "math.h"
#include <map>
#include <random>
using namespace std;
/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/

void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


// LAB 2
void grayChannels() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat R = Mat(height, width, CV_8UC1);
		Mat G = Mat(height, width, CV_8UC1);
		Mat B = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b pixel = src.at<Vec3b>(i, j);
				B.at<uchar>(i, j) = pixel[0];
				G.at<uchar>(i, j) = pixel[1];
				R.at<uchar>(i, j) = pixel[2];
			}
		}

		imshow("input image", src);
		imshow("R", R);
		imshow("G", G);
		imshow("B", B);
		waitKey();
	}

}

void grayscale() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b pixel = src.at<Vec3b>(i, j);
				uchar b = pixel[0];
				uchar g = pixel[1];
				uchar r = pixel[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;

			}
		}

		imshow("input image", src);
		imshow("grayscale image", dst);
		waitKey();
	}

}

void toBinary(int thresh) {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				dst.at<uchar>(i, j) = (pixel < thresh) ? 0 : 255;
			}
		}

		imshow("input image", src);
		imshow("binary image", dst);
		waitKey();
	}

}

void toHSV() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b pixel = src.at<Vec3b>(i, j);
				float b = (float)pixel[0] / 255;
				float g = (float)pixel[1] / 255;
				float r = (float)pixel[2] / 255;
				float h, s, v;
				float M = max(r, g);
				M = max(M, b);
				float m = min(r, g);
				m = min(m, b);
				float C = M - m;

				// Value
				v = M;

				// Saturation
				s = (M != 0) ? (C / M) : 0;

				// Hue
				if (C != 0) {
					if (M == r) h = 60 * (g - b) / C;
					if (M == g) h = 120 + 60 * (b - r) / C;
					if (M == b) h = 240 + 60 * (r - g) / C;
				}
				else { // grayscale
					h = 0;
				}

				if (h < 0) {
					h += 360;
				}

				H.at<uchar>(i, j) = h * 255 / 360;
				S.at<uchar>(i, j) = s * 255;
				V.at<uchar>(i, j) = v * 255;

			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);
		waitKey();
	}

}

bool isInside(Mat src, int i, int j) {
	int height = src.rows;
	int width = src.cols;
	printf("Image width: %d, height: %d\n", height, width);
	if (i >= 0 && i < height && j >= 0 && j < width) {
		return true;
	}

	return false;
}

void testIsInside(int i, int j) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);
		bool isIn = isInside(src, i, j);
		if (isIn) {
			printf("Position (%d, %d) is inside image\n", i, j);
		}
		else {
			printf("Position (%d, %d) is not inside image\n", i, j);
		}
		//imshow("input image", src);
		//waitKey();
	}
}

void fromHSV() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		Mat finalm = Mat(height, width, CV_8UC3);

		Mat hsv = Mat(height, width, CV_8UC3);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b pixel = src.at<Vec3b>(i, j);
				float b = (float)pixel[0] / 255;
				float g = (float)pixel[1] / 255;
				float r = (float)pixel[2] / 255;

				// FROM RGB TO HSV
				float h, s, v;
				float M = max(r, g);
				M = max(M, b);
				float m = min(r, g);
				m = min(m, b);
				float C = M - m;

				// Value
				v = M;

				// Saturation
				s = (M != 0) ? (C / M) : 0;

				// Hue
				if (C != 0) {
					if (M == r) h = 60 * (g - b) / C;
					if (M == g) h = 120 + 60 * (b - r) / C;
					if (M == b) h = 240 + 60 * (r - g) / C;
				}
				else { // grayscale
					h = 0;
				}

				if (h < 0) {
					h += 360;
				}

				// FROM HSV TO RGB

				hsv.at<Vec3b>(i, j)[0] = h * 180 / 360;
				hsv.at<Vec3b>(i, j)[1] = s * 255;
				hsv.at<Vec3b>(i, j)[2] = v * 255;


			}
		}

		imshow("input image", src);
		cvtColor(hsv, finalm, CV_HSV2BGR);
		//imshow("H", H);
		//imshow("S", S);
		//imshow("V", V);
		imshow("from HSV to RGB", finalm);

		waitKey();
	}

}

// LAB 3
void histogram() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		int histo[256] = { 0 };

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				histo[pixel] ++;

			}
		}

		imshow("input image", src);
		showHistogram("Histogram", histo, 256, 256);


		waitKey();
	}
}

void pdf() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		int histo[256] = { 0 };

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				histo[pixel] ++;

			}
		}

		float pdf[256];
		int M = height * width;
		for (int i = 0; i < 256; i++) {
			pdf[i] = (float)histo[i] / M;
		}

		imshow("input image", src);
		waitKey();
	}
}

Mat multilevelThreshold(Mat src) {

	int height = src.rows;
	int width = src.cols;

	// histogram
	int histo[256] = { 0 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar pixel = src.at<uchar>(i, j);
			histo[pixel] ++;

		}
	}

	// pdf
	float pdf[256];
	int M = height * width;
	for (int i = 0; i < 256; i++) {
		pdf[i] = (float)histo[i] / M;
	}

	std::vector<int> maxim;
	maxim.push_back(0);
	int WH = 5;
	int windowWidth = 2 * WH + 1;
	float threshold = 0.0003;

	for (int k = WH; k <= 255 - WH; k++) {
		float sum = 0;
		float locmax = -1;
		bool greq = true;
		for (int j = k - WH; j <= k + WH; j++) {
			sum += pdf[j];
			if (pdf[j] > pdf[k]) {
				greq = false;
			}

		}
		float v = sum / (2 * WH + 1);

		if ((pdf[k] > v + threshold) && greq) {
			// histogram maximum
			maxim.push_back(k);
			printf("k=%d\n", k);
			printf("am pus in max\n");
		}
	}

	maxim.push_back(255);

	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			uchar pixel = src.at<uchar>(i, j);
			int k = 0;
			int close;
			int diff_min = 256;
			for (int k = 0; k < maxim.size(); k++) {
				int maxx = maxim.at(k);
				if (diff_min > abs(maxx - pixel)) {
					diff_min = abs(maxx - pixel);
					close = maxx;
				}
			}

			dst.at<uchar>(i, j) = close;
		}
	}

	return dst;
}

void multilevelThresholdTest() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = multilevelThreshold(src);

		imshow("original image", src);
		imshow("multilevel threshold", dst);
		waitKey();
	}
}

Mat ditheringgg(Mat srcin, Mat thr) {
	int height = srcin.rows;
	int width = srcin.cols;
	
	Mat src = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			src.at<uchar>(i, j) = srcin.at<uchar>(i, j);
		}
	}


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar oldpixel = srcin.at<uchar>(i, j);
			uchar newpixel = thr.at<uchar>(i, j);
			src.at<uchar>(i, j) = newpixel;
			float error = oldpixel - newpixel;
			cout << "error " << error << endl;
			if (j + 1 < width) {
				src.at<uchar>(i, j + 1) += float(7 * error / 16);
			}
			if (i + 1 < height && j - 1 >= 0) {
				src.at<uchar>(i + 1, j - 1) += float(3 * error / 16);
			}
			if (i + 1 < height) {
				src.at<uchar>(i + 1, j) += float(5 * error / 16);
			}
			if (i + 1 < height && j + 1 < width) {
				src.at<uchar>(i + 1, j + 1) += float(error / 16);
			}
		}
	}


	return src;
}

Mat dithering(Mat src) {

	int height = src.rows;
	int width = src.cols;

	// histogram
	int histo[256] = { 0 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar pixel = src.at<uchar>(i, j);
			histo[pixel] ++;

		}
	}

	// pdf
	float pdf[256];
	int M = height * width;
	for (int i = 0; i < 256; i++) {
		pdf[i] = (float)histo[i] / M;
	}

	std::vector<int> maxim;
	maxim.push_back(0);
	int WH = 5;
	int windowWidth = 2 * WH + 1;
	float threshold = 0.0003;

	for (int k = WH; k <= 255 - WH; k++) {
		float sum = 0;
		float locmax = -1;
		bool greq = true;
		for (int j = k - WH; j <= k + WH; j++) {
			sum += pdf[j];
			if (pdf[j] > pdf[k]) {
				greq = false;
			}

		}
		float v = sum / (2 * WH + 1);

		if ((pdf[k] > v + threshold) && greq) {
			// histogram maximum
			maxim.push_back(k);
			printf("k=%d\n", k);
			printf("am pus in max\n");
		}
	}

	maxim.push_back(255);

	Mat dith = Mat(height, width, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dith.at<uchar>(i, j) = src.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			uchar oldpixel = dith.at<uchar>(i, j);
			// find closest histogram maximum of oldpixel
			int k = 0;
			uchar close;
			int diff_min = 256;
			for (int k = 0; k < maxim.size(); k++) {
				int maxx = maxim.at(k);
				if (diff_min > abs(maxx - oldpixel)) {
					diff_min = abs(maxx - oldpixel);
					close = maxx;
				}
			}
			uchar newpixel = close;
			dith.at<uchar>(i, j) = newpixel;
			float error = oldpixel - newpixel;
			if (j + 1 < width) {
				dith.at<uchar>(i, j + 1) += float(7 * error / 16);
			}
			if (i + 1 < height && j - 1 >= 0) {
				dith.at<uchar>(i + 1, j - 1) += float(3 * error / 16);
			}
			if (i + 1 < height) {
				dith.at<uchar>(i + 1, j) += float(5 * error / 16);
			}
			if (i + 1 < height && j + 1 < width) {
				dith.at<uchar>(i + 1, j + 1) += float(error / 16);
			}

		
		}
	}
	return dith;
}

void ditheringTest() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		//Mat thr = multilevelThreshold(src);

		Mat dith = dithering(src);

		imshow("original image", src);
		//imshow("multilevel threshold", thr);
		imshow("Floyd-Steinberg dithering", dith);
		waitKey();
	}
}

// LAB 4
void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	int height = src->rows;
	int width = src->cols;

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		// unde am dat click
		int r = (int)(*src).at<Vec3b>(y, x)[2];
		int g = (int)(*src).at<Vec3b>(y, x)[1];
		int b = (int)(*src).at<Vec3b>(y, x)[0];

		Mat imgcol = Mat(height, width, CV_8UC3);
		Mat imgcol2 = Mat(height, width, CV_8UC3);
		// compute the binary image and a clone
		Mat img = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if ((int)(*src).at<Vec3b>(i, j)[2] == r &&
					(int)(*src).at<Vec3b>(i, j)[1] == g &&
					(int)(*src).at<Vec3b>(i, j)[0] == b) {
					// negru
					img.at<uchar>(i, j) = 0;
					// culoarea
					imgcol2.at<Vec3b>(i,j) = src->at<Vec3b>(i, j);
				}
				else {
					// alb
					img.at<uchar>(i, j) = 255;
					imgcol2.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}

				imgcol.at<Vec3b>(i, j) = src->at<Vec3b>(i, j);


			}
		}
		
		// 1. Area
		int area = 0;
		// 2. Center of mass
		int sumr = 0;
		int sumc = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (img.at<uchar>(i, j) == 0) {
					// 1. Area
					area++;
					// 2. Center of mass
					sumr += i;
					sumc += j;
				}
			}
		}

		float center_mass_r = sumr / area;
		float center_mass_c = sumc / area;

		cv::Point center_mass = cv::Point(center_mass_c, center_mass_r);
		circle(imgcol, center_mass, 2, (109, 28, 92),5);


		// 3. Axis of elongation
		int asum1 = 0;
		int asum2 = 0;
		int asum3 = 0;

		// 4. Perimeter
		int perimeter = 0;

	    // 6. Aspect ratio
		int cmax = 0;
		int rmax = 0;
		int cmin = width;
		int rmin = height;

		// 7. Projections
		std::vector<int> projh;
		std::vector<int> projv;


		for (int i = 0; i < height; i++) {
			int sumline = 0;
			for (int j = 0; j < width; j++) {
				uchar pixel = img.at<uchar>(i, j);
				if (pixel == 0) {
					// 3. Axis of elongation
					asum1 += (i - center_mass_r)*(j - center_mass_c);
					asum2 += (j - center_mass_c)*(j - center_mass_c);
					asum3 += (i - center_mass_r)*(i - center_mass_r);
					// 4. Perimeter
					if ((i > 0 && j > 0 && img.at<uchar>(i - 1, j - 1) == 255) ||
						(i > 0 && img.at<uchar>(i - 1, j) == 255) ||
						(i > 0 && j < width - 1 && img.at<uchar>(i - 1, j + 1) == 255) ||
						(j > 0 && img.at<uchar>(i, j - 1) == 255) ||
						(j < width - 1 && img.at<uchar>(i, j + 1) == 255) ||
						(i < height - 1 && j > 0 && img.at<uchar>(i + 1, j - 1) == 255) ||
						(i < height - 1 && img.at<uchar>(i + 1, j) == 255) ||
						(i < height - 1 && j < width - 1 && img.at<uchar>(i + 1, j + 1) == 255)) {
						perimeter += 1;
						// draw contour
						imgcol.at<Vec3b>(i, j) = Vec3b(0,0,0);
					}
					// 6. Aspect ratio
					if (cmax < j) cmax = j;
					if (rmax < i) rmax = i;
					if (cmin > j) cmin = j;
					if (rmin > i) rmin = i;
					// 7. Horizontal projection
					sumline++;
				}
			}
			projh.push_back(sumline);
		}

		// 7. Vertical projection
		for (int j = 0; j < width; j++) {
			int sumcol = 0;
			for (int i = 0; i < height; i++) {
				if (img.at<uchar>(i, j) == 0) {
					sumcol++;
				}
			}
			projv.push_back(sumcol);
		}

		// draw Horizontal projection
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < projh.at(i); j++) {
				imgcol2.at<Vec3b>(i, j) = Vec3b(42, 87, 93);
			}
		}
		// draw vertical projection
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < projv.at(width-i-1); j++) {
				imgcol2.at<Vec3b>(j, i) = Vec3b(82, 93, 23);
			}
		}
		printf("asum2%d asum3%d\n", asum2, asum3);

		float angle = float(atan2(2 * asum1, asum2 - asum3)) * 180 / PI;
		angle /= 2;
		if (angle < 0) angle += 180;

		perimeter = perimeter * PI / 4;

		// 5. Thinnes ratio
		float thinnes_ratio = float(4 * PI * area / (perimeter*perimeter));

		// 6. Aspect ratio
		float aspect_ratio = float(cmax - cmin + 1) / (rmax - rmin + 1);

		// draw axis of elongation
		int length = 150;
		Point P1;
		P1.x = (int)round(center_mass_c + length * cos(angle * PI / 180.0));
		P1.y = (int)round(center_mass_r + length * sin(angle * PI / 180.0));
		line(imgcol, center_mass, P1, (100,200,0), 1);
		Point P2;
		P2.x = (int)round(center_mass_c + length * cos((angle+180) * PI / 180.0));
		P2.y = (int)round(center_mass_r + length * sin((angle+180) * PI / 180.0));
		line(imgcol, center_mass, P2, (100, 200, 0), 1);


		cout << "Area: " << area << endl;
		cout << "Center of mass: row " << center_mass_r <<
			" column " << center_mass_c << endl;
		cout << "Angle of elongation: " << angle << endl;
		cout << "Perimeter: " << perimeter << endl;
		cout << "Thinnes ratio: " << thinnes_ratio << endl;
		cout << "Aspect ratio:" << aspect_ratio << endl;
		imshow("Contour, center and axis of elongation", imgcol);
		imshow("Projections", imgcol2);
	}
}

void computeGeoFeatures()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);

		//Create a window
		namedWindow("My Window", 1);
		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void threshArea(int thArea, int phiLow, int phiHigh) {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		cout << "ihfrhfrhfhr" << phiLow << phiHigh;

		// map the colors to their area
		std::vector<Vec3b> colors;
		std::vector<int> area;
		std::vector<int> sumr;
		std::vector<int> sumc;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b color = src.at<Vec3b>(i, j);
				bool found = false;
					for (int k = 0; k < colors.size(); k++) {
						if (colors.at(k)[0] == color[0] && 
							colors.at(k)[1] == color[1] &&
							colors.at(k)[2] == color[2]) {
							area.at(k) += 1;
							sumr.at(k) += i;
							sumc.at(k) += j;
							found = true;
						}
					}
					if (!found) {
						colors.push_back(color);
						area.push_back(0);
						sumr.push_back(0);
						sumc.push_back(0);
					}
					
			}
		}

		// center mass
		std::vector<float> cen_mass_r;
		std::vector<float> cen_mass_c;
		for (int i = 0; i < sumr.size(); i++) {
			cen_mass_r.push_back(float(sumr.at(i) / area.at(i)));
			cen_mass_c.push_back(float(sumc.at(i) / area.at(i)));
		}

		std::vector<float> asum1, asum2, asum3;
		for (int i = 0; i < colors.size(); i++) {
			asum1.push_back(0);
			asum2.push_back(0);
			asum3.push_back(0);
		}


		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				//cout << "aici";
				Vec3b color = src.at<Vec3b>(i, j);
				for (int k = 0; k < colors.size(); k++) {
					if (colors.at(k)[0] == color[0] &&
						colors.at(k)[1] == color[1] &&
						colors.at(k)[2] == color[2]) {
						asum1.at(k) += (i - cen_mass_r.at(k))*(j - cen_mass_c.at(k));
						asum2.at(k) += (j - cen_mass_c.at(k))*(j - cen_mass_c.at(k));
						asum3.at(k) += (i - cen_mass_r.at(k))*(i - cen_mass_r.at(k));
					}
				}

			}
		}

		std::vector<float> angles;
		for (int i = 0; i < sumr.size(); i++) {
			float angle = float(atan2(2 * asum1.at(i), asum2.at(i) - asum3.at(i))) * 180 / PI;
			angle /= 2;
			if (angle < 0) angle += 180;
			angles.push_back(angle);
		}

		Mat img = Mat(height, width, CV_8UC3);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b color = src.at<Vec3b>(i, j);
				int index = -1;
				// caut indexul culorii
				for (int k = 0; k < colors.size(); k++) {
					if (colors.at(k)[0] == color[0] &&
						colors.at(k)[1] == color[1] &&
						colors.at(k)[2] == color[2]) {
						index = k;
					}
				}
				//cout << "index %d \n" << index << endl;
				if (area.at(index) < thArea && 
					angles.at(index) < phiHigh &&
					angles.at(index) > phiLow){
					img.at<Vec3b>(i, j) = color;
				}

				else {
					img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
				//cout << "area " << area.at(index) << endl;

			}
		}

		//show the image
		imshow("My Window", src);
		// threshold area
		imshow("Threshold area", img);

		// Wait until user press some key
		waitKey(0);
	}
}

// LAB 5

// Algorithm 1 - Breadth first traversal
void BFS() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname,IMREAD_GRAYSCALE);
		int height = img.rows;
		int width = img.cols;

		// result image
		Mat dst = Mat(height, width, CV_8UC3);

		int label = 0;
		// initialize with 0 the label matrix
		Mat labels = cv::Mat::zeros(cv::Size(width, height),CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				// if the pixel is unlabeled
				if (img.at<uchar>(i, j) == 0 && labels.at<uchar>(i,j) == 0) {
					// create a new label
					label++;
					labels.at<uchar>(i, j) = label;
					std::queue < std::tuple<int, int>> Q;
					// put the pixel in Q
					Q.push({ i,j });
					// propagate the label to pixel's 8-neighbors
					while (!Q.empty()) {
						std::tuple<int, int> q = Q.front();
						int qi = std::get<0>(q);
						int qj = std::get<1>(q);
						Q.pop();
						// if pixel's neighbors are unlabeled
						// label with pixel's label
						// and push in Q
						if (qi > 0 && qj > 0 && img.at<uchar>(qi - 1, qj - 1) == 0 && labels.at<uchar>(qi - 1, qj - 1) == 0) {
							labels.at<uchar>(qi - 1, qj - 1) = label;
							Q.push({ qi - 1,qj - 1 });
						}
						if (qi > 0 && img.at<uchar>(qi - 1, qj) == 0 && labels.at<uchar>(qi - 1, qj) == 0) {
							labels.at<uchar>(qi - 1, qj) = label;
							Q.push({ qi - 1,qj });
						}
						if (qi > 0 && qj < width - 1 && img.at<uchar>(qi - 1, qj + 1) == 0 && labels.at<uchar>(qi - 1, qj + 1) == 0) {
							labels.at<uchar>(qi - 1, qj + 1) = label;
							Q.push({ qi - 1,qj + 1 });
						}
						if (qj > 0 && img.at<uchar>(qi, qj - 1) == 0 && labels.at<uchar>(qi, qj - 1) == 0) {
							labels.at<uchar>(qi, qj - 1) = label;
							Q.push({ qi ,qj - 1 });
						}
						if (qj < width - 1 && img.at<uchar>(qi, qj + 1) == 0 && labels.at<uchar>(qi, qj + 1) == 0) {
							labels.at<uchar>(qi, qj + 1) = label;
							Q.push({ qi ,qj + 1 });
						}
						if (qi < height - 1 && qj > 0 && img.at<uchar>(qi + 1, qj - 1) == 0 && labels.at<uchar>(qi+1, qj - 1) == 0) {
							labels.at<uchar>(qi+1, qj-1) = label;
							Q.push({ qi+1 ,qj - 1 });
						}
						if (qi < height - 1 && img.at<uchar>(qi + 1, qj) == 0 && labels.at<uchar>(qi + 1, qj ) == 0) {
							labels.at<uchar>(qi + 1, qj) = label;
							Q.push({ qi + 1 ,qj });
						}
						if (qi < height - 1 && qj < width - 1 && img.at<uchar>(qi + 1, qj + 1)  == 0 && labels.at<uchar>(qi + 1, qj+1) == 0) {
							labels.at<uchar>(qi + 1, qj+1) = label;
							Q.push({ qi + 1 ,qj +1});
						}
					}
				}

			}
		}



		// generate random colors for each label
		std::vector<Vec3b> colors;
		for (int i = 0; i < label; i++) {
			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;
			cv::Vec3b color;
			color = Vec3b(b, g, r);
			colors.push_back(color);
		}

		// color the picture
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (labels.at<uchar>(i, j) == 0) {
					dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
				else {
					int label = labels.at<uchar>(i, j);
					dst.at<Vec3b>(i, j) = colors.at(label-1);
				}
			}
		}
		imshow("input image", img);
		imshow("labeled image", dst);
		waitKey();
	}

}

// Min element from a vector
uchar vecMin(std::vector<uchar> v) {
	uchar m = v.at(0);
	int size = v.size();
	for (int i = 1; i < size; i++) {
		if (m > v.at(i)) {
			m = v.at(i);
		}
	}
	return m;
}

// Algorithm 2 - Two - pass with equivalence classes
// First pass
void twoPassInterm() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		int height = img.rows;
		int width = img.cols;

		// result image
		Mat dst = Mat(height, width, CV_8UC3);

		uchar label = 0;
		Mat labels = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
		vector<vector<uchar>> edges;
		edges.resize(255);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				// if the pixel is unlabeled
				if (img.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0) {
					vector<uchar> L;
					// if the previously visited pixels are labeled
					// put their labels in L 
					if (j > 0 &&  labels.at<uchar>(i, j - 1) > 0){
						L.push_back(labels.at<uchar>(i, j - 1));
					}
					if (i > 0 && j > 0  && labels.at<uchar>(i - 1, j - 1) > 0) {
						L.push_back(labels.at<uchar>(i - 1, j - 1));
					}

					if (i > 0 && labels.at<uchar>(i - 1, j) > 0) {
						L.push_back(labels.at<uchar>(i - 1, j));
					}
					if (i > 0 && j < width - 1 && labels.at<uchar>(i - 1, j + 1) > 0) {
						L.push_back(labels.at<uchar>(i - 1, j + 1));
					}

					// if none of p-neighbors are labeled
					 // assign a new label
					if(L.size()==0){
						label++;
						labels.at<uchar>(i, j) = label;
					}
					else {
						// choose the minimum label from p-neighbors' labels
						int x = vecMin(L);
						labels.at<uchar>(i, j) = x;
						for (int i = 0; i < L.size(); i++) {
							int y = L.at(i);
							if (y != x) {
								// mark neighboring label y as equivalent to x
								edges.at(x).push_back(y);
								edges.at(y).push_back(x);
							}
						}
					}
				}

			}
		}


		// generate random colors for each label
		std::vector<Vec3b> colors;
		for (int i = 0; i < label; i++) {
			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;
			cv::Vec3b color;
			color = Vec3b(b, g, r);
			colors.push_back(color);
		}

		// color the picture
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (labels.at<uchar>(i, j) == 0) {
					dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
				else {
					int label = labels.at<uchar>(i, j);
					dst.at<Vec3b>(i, j) = colors.at(label - 1);
				}
			}
		}
		imshow("input image", img);
		imshow("labeled image", dst);
		waitKey();
	}

}

// Algorithm 2 - Two-pass with equivalence classes
// Final result
void twoPass() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		int height = img.rows;
		int width = img.cols;

		// result image
		Mat dst = Mat(height, width, CV_8UC3);

		uchar label = 0;
		Mat labels = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
		vector<vector<uchar>> edges;
		edges.resize(255);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				// if the pixel is unlabeled
				if (img.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0) {
					vector<uchar> L;
					// if the previously visited pixels are labeled
					// put their labels in L 
					if (j > 0 && labels.at<uchar>(i, j - 1) > 0) {
						L.push_back(labels.at<uchar>(i, j - 1));
					}
					if (i > 0 && j > 0 && labels.at<uchar>(i - 1, j - 1) > 0) {
						L.push_back(labels.at<uchar>(i - 1, j - 1));
					}

					if (i > 0 && labels.at<uchar>(i - 1, j) > 0) {
						L.push_back(labels.at<uchar>(i - 1, j));
					}
					if (i > 0 && j < width - 1 && labels.at<uchar>(i - 1, j + 1) > 0) {
						L.push_back(labels.at<uchar>(i - 1, j + 1));
					}

					// if none of p-neighbors are labeled
					 // assign a new label
					if (L.size() == 0) {
						label++;
						labels.at<uchar>(i, j) = label;
					}
					else {
						// choose the minimum label from p-neighbors' labels
						int x = vecMin(L);
						labels.at<uchar>(i, j) = x;
						for (int i = 0; i < L.size(); i++) {
							int y = L.at(i);
							if (y != x) {
								// mark neighboring label y as equivalent to x
								edges.at(x).push_back(y);
								edges.at(y).push_back(x);
							}
						}
					}
				}

			}
		}

		// assign new label to each equivalence class
		int newlabel = 0;
		int newlabels[256] = { 0 };
		for (int i = 1; i <= label; i++) {
			// if equivalence class isn't labeled
			if (newlabels[i] == 0) {
				newlabel++;
				queue<int> Q;
				// label the equivalence class's first item
				newlabels[i] = newlabel;
				Q.push(i);
				// label all items from equivalence class
				while (!Q.empty()) {
					int x = Q.front();
					Q.pop();
					// iterate through all members of the class
					// label the member and put in Q
					for (int j = 0; j < edges.at(x).size(); j++) {
						int y = edges.at(x).at(j);
						if (newlabels[y] == 0) {
							newlabels[y] = newlabel;
							Q.push(y);
						}
					}
				}
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				labels.at<uchar>(i, j) = newlabels[labels.at<uchar>(i, j)];
			}
		}

		// generate random colors for each label
		std::vector<Vec3b> colors;
		for (int i = 0; i < label; i++) {
			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;
			cv::Vec3b color;
			color = Vec3b(b, g, r);
			colors.push_back(color);
		}

		// color the picture
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (labels.at<uchar>(i, j) == 0) {
					dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
				else {
					int label = labels.at<uchar>(i, j);
					dst.at<Vec3b>(i, j) = colors.at(label - 1);
				}
			}
		}
		
		imshow("input image", img);
		imshow("labeled image", dst);
		waitKey();
	}

}

// LAB 6
void borderTracing() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		// result image
		Mat dst = cv::Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				dst.at<uchar>(i, j) = 255;
			}
		}

		// chain code
		vector<int> chain_code;

		// P0 - starting pixel of the region border
		// P1 - second border element
		// Pn-1 - previous border element
		// Pn - current boundary element
		Point2i P0, P1, Pn, Pn1;

		// search the image until a pixel of a new region is found
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0) {
					// we know that the image contains only one object
					// so we stop when we find the object
					// P0 - starting pixel 
					P0 = { j,i };
					i = height;
					j = width;
				}
			}
		}

		// indexes of 8-connectivity neighbors
		int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

		// 8-connectivity border
		int dir = 7;
		
		Point2i N;

		// P0 - starting pixel of the region border
		int i = P0.y;
		int j = P0.x;

		// iterator
		int n = 0;


		do {
			// dir - even -> (dir + 7) mod 8
			if (dir % 2 == 0) {
				dir = (dir + 7) % 8;
			}
			// dir - odd -> (dir + ) mod 8
			else {
				dir = (dir + 6) % 8;
			}

			// previous border element
			Pn1 = { j,i };

			cout << j << " " << i<< "\n";

			// in the 3x3 neighorhood of the current pixel
			// search for the next pixel with the same value as the current pixel
			// the pixel is a new boundary element Pn
			while (src.at<uchar>(i + di[dir], j + dj[dir]) != 0) {
				// update dir
				dir = (dir + 1) % 8;
			}

			// push in chain code vector
			chain_code.push_back(dir);

			// new boundary element
			Pn = { j + dj[dir], i + di[dir] };

			// if it is the first iteration
			if (n == 0) {
				// assign the second border element
				P1 = Pn;
			}
			
			// update i and j
			i = Pn.y;
			j = Pn.x;

			// draw the border
			dst.at<uchar>(i, j) = 0;

			n++;

		// stop condition: Pn is equal to P1 and Pn-1 is equal to P0
		// we have to perform at least two iterations
		} while (!((Pn == P1) && (Pn1 == P0) && (n >= 2)));

		imshow("input image", src);
		imshow("border", dst);

		chain_code.pop_back();
		//chain_code.pop_back();

		printf("CHAIN CODE:\n");
		for (int i = 0; i < chain_code.size()-1; i++) {
			printf( "%d ", chain_code.at(i));
		}

		printf("\nDERIVATIVE CHAIN CODE: \n");
		for (int i = 0; i < chain_code.size()-1; i++) {
			int der = ((chain_code.at(i + 1) - chain_code.at(i)) + 8) % 8;
			printf("%d ", der);
		}
		int der = ((chain_code.at(0) - chain_code.at(chain_code.size()-1)) + 8) % 8;
		printf("%d ", der);
		waitKey();
	}

}

void borderReconstruct() {

	Mat src = imread("Images/gray_background.bmp", IMREAD_GRAYSCALE);
	FILE *pf = fopen("reconstruct.txt", "r");

	// start coordinates
	int i, j;
	fscanf(pf, "%d%d", &i, &j);
	//cout << i << " " << j;

	// number of pixels
	int n;
	fscanf(pf, "%d", &n);
	//cout << n;

	// indexes of 8-connectivity neighbors
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int dir;

	for (int k = 0; k < n; k++) {
		src.at<uchar>(i, j) = 0;
		fscanf(pf, "%d ", &dir);
		i += di[dir];
		j += dj[dir];
	}
	

	imshow("result", src);

	waitKey(0);

}

// LAB 7
// 1. Image dilation
void dilate(int strElement, int n) {
	
	// choose structuring element 
	int increment = 0;
	if (strElement == 4) {
		increment = 2;
	}
	else if (strElement == 8) {
		increment = 1;
	}
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		imshow("input image", src);

		Mat dst = cv::Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				dst.at<uchar>(i, j) = 255;
			}
		}

		for (int it = 0; it < n; it++) {
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					uchar pixel = src.at<uchar>(i, j);
					if (pixel == 0) {
						//If the origin of the structuring element coincides with an 'object' pixel in the image, make
						//(label) all pixels from the image covered by the structuring element as ‘object’ pixels.
						dst.at<uchar>(i, j) = 0;
						for (int k = 0; k < 8; k += increment) {
							if (i + di[k] < height && i + di[k] >= 0 && j + dj[k] < width && j + dj[k] >= 0) {
								dst.at<uchar>(i + di[k], j + dj[k]) = 0;
							}
						}
					}

				}
			}
			src = dst.clone();
		}

		
		imshow("dilated image", dst);
		waitKey();
	}


}

// 2. Image erosion
void erode(int strElement, int n) {
	int increment = 0;
	if (strElement == 4) {
		increment = 2;
	}
	else if (strElement == 8) {
		increment = 1;
	}
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = cv::Mat(height, width, CV_8UC1);

		imshow("input image", src);

		for (int it = 0; it < n; it++) {
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					uchar pixel = src.at<uchar>(i, j);
					if (pixel == 0) {
						// If the origin of the structuring element coincides with an 'object' pixel in the image
						bool toErode = false;
						for (int k = 0; k < 4 && toErode == false; k += increment) {
							if (i + di[k] < height && i + di[k] >= 0 && j + dj[k] < width && j + dj[k] >= 0) {
								if (src.at<uchar>(i + di[k], j + dj[k]) == 255) {
									toErode = true;
								}
							}
						}

						// if any of the 'object' pixels in the structuring element extend beyond the 'object' pixels in the
						// image, then change the current 'object' pixel in the image to a 'background' pixel.
						if (toErode) dst.at<uchar>(i, j) = 255;
						else dst.at<uchar>(i, j) = 0;
					}
					else
						dst.at<uchar>(i, j) = 255;
				}
			}
			src = dst.clone();
		}
		
		imshow("eroded image", dst);
		waitKey();
	}
}

// equalImg returns true if two images are identical, false otherwise
boolean equalImg(Mat img1, Mat img2) {
	int h1 = img1.rows;
	int w1 = img1.cols;
	int h2 = img2.rows;
	int w2 = img2.cols;
	if (h1 != h2 || w1 != w2) return false;
	for (int i = 0; i < h1; i++) {
		for (int j = 0; j < w1; j++) {
			if (img1.at<uchar>(i, j) != img2.at<uchar>(i, j))
				return false;
		}
	}
	return true;
}

// 3. Image opening
void open(int n) {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat erode = cv::Mat(height, width, CV_8UC1);
		Mat open = cv::Mat(height, width, CV_8UC1);

		int di[8] = { 0, -1, 0, 1 };
		int dj[8] = { 1, 0, -1, 0 };

		imshow("input image", src);

		bool noChange = false;
		for (int it = 0; it < n && noChange == false; it++) {
			// erode
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					uchar pixel = src.at<uchar>(i, j);
					if (pixel == 0) {
						bool toErode = false;
						for (int k = 0; k < 4 && toErode == false; k++) {
							if (i + di[k] < height && i + di[k] >= 0 && j + dj[k] < width && j + dj[k] >= 0) {
								if (src.at<uchar>(i + di[k], j + dj[k]) == 255) {
									toErode = true;
								}
							}
						}

						if (toErode) erode.at<uchar>(i, j) = 255;
						else erode.at<uchar>(i, j) = 0;
					}
					else
						erode.at<uchar>(i, j) = 255;
				}
			}


			// dilate
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					open.at<uchar>(i, j) = 255;
				}
			}

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					uchar pixel = erode.at<uchar>(i, j);
					if (pixel == 0) {
						open.at<uchar>(i, j) = 0;
						for (int k = 0; k < 4; k++) {
							if (i + di[k] < height && i + di[k] >= 0 && j + dj[k] < width && j + dj[k] >= 0) {
								open.at<uchar>(i + di[k], j + dj[k]) = 0;
							}
						}
					}
				}
			}

			if (equalImg(src, open))
				noChange = true;

			src = open.clone();
		}

		
		imshow("opened image", open);
		waitKey();
	}

}

// 4. Image closing
void close(int n) {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dilate = cv::Mat(height, width, CV_8UC1);
		Mat close = cv::Mat(height, width, CV_8UC1);

		imshow("input image", src);

		int di[8] = { 0, -1, 0, 1 };
		int dj[8] = { 1, 0, -1, 0 };

		bool noChange = false;
		for (int it = 0; it < n && noChange == false; it++) {
			// dilate
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					dilate.at<uchar>(i, j) = 255;
				}
			}

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					uchar pixel = src.at<uchar>(i, j);
					if (pixel == 0) {
						dilate.at<uchar>(i, j) = 0;
						for (int k = 0; k < 4; k++) {
							if (i + di[k] < height && i + di[k] >= 0 && j + dj[k] < width && j + dj[k] >= 0) {
								dilate.at<uchar>(i + di[k], j + dj[k]) = 0;
							}
						}
					}
					
				}
			}

			// erode
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					uchar pixel = dilate.at<uchar>(i, j);
					if (pixel == 0) {
						bool toErode = false;
						for (int k = 0; k < 4 && toErode == false; k++) {
							if (i + di[k] < height && i + di[k] >= 0 && j + dj[k] < width && j + dj[k] >= 0) {
								if (dilate.at<uchar>(i + di[k], j + dj[k]) == 255) {
									toErode = true;
								}
							}
						}

						if (toErode) close.at<uchar>(i, j) = 255;
						else close.at<uchar>(i, j) = 0;
					}
					else
						close.at<uchar>(i, j) = 255;
				}
			}

			if (equalImg(src, close))
				noChange = true;

			src = close.clone();

		}


		imshow("closed image", close);
		waitKey();
	}

}

// 5. Boundary extraction
void boundaryExtraction() {

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat erode = Mat(height, width, CV_8UC1);
		Mat boundary = Mat(height, width, CV_8UC1);

		// erode
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				if (pixel == 0) {
					bool toErode = false;
					for (int k = 0; k < 8 && toErode == false; k++) {
						if (i + di[k] < height && i + di[k] >= 0 && j + dj[k] < width && j + dj[k] >= 0) {
							if (src.at<uchar>(i + di[k], j + dj[k]) == 255) {
								toErode = true;
							}
						}
					}

					if (toErode) erode.at<uchar>(i, j) = 255;
					else erode.at<uchar>(i, j) = 0;
				}
				else
					erode.at<uchar>(i, j) = 255;
			}
		}

		// perform difference between image and its erosion
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) != erode.at<uchar>(i, j)) {
					boundary.at<uchar>(i, j) = 0;
				}
				else {
					boundary.at<uchar>(i, j) = 255;
				}
			}
		}

		imshow("input image", src);
		imshow("eroded image", boundary);
		waitKey();
	}
}

// 6. Region filling
void regionFilling() {
	int di[8] = { 0, -1, 0, 1 };
	int dj[8] = { 1, 0, -1, 0 };

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
	
		int height = img.rows;
		int width = img.cols;

		
		// create a white output image (background) = Xk
		Mat img_xk = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				img_xk.at<uchar>(i, j) = 255;
			}
		}
		// start filling from the center pixe - label center pixel in the ouput image as object
		img_xk.at<uchar>(height / 2, width / 2) = 0;

		// compute the complement of the source image
		Mat img_complement = ~Mat(img);

		// dilated image
		Mat dilated = Mat(height, width, CV_8UC1);
		// image Xk-1
		Mat img_xk_1 = Mat(height, width, CV_8UC1);


		do {
			// Xk-1 = Xk
			img_xk_1 = img_xk.clone();

			// dilate Xk-1
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					dilated.at<uchar>(i, j) = 255;
				}
			}

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					uchar pixel = img_xk_1.at<uchar>(i, j);
					if (pixel == 0) {
						dilated.at<uchar>(i, j) = 0;
						for (int k = 0; k < 4; k++) {
							if (i + di[k] < height && i + di[k] >= 0 && j + dj[k] < width && j + dj[k] >= 0) {
								dilated.at<uchar>(i + di[k], j + dj[k]) = 0;
							}
						}
					}
				}
			}

			// intersect dilated image with complement of the source image
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (dilated.at<uchar>(i, j) == 0 && img_complement.at<uchar>(i, j) == 0) {
						img_xk.at<uchar>(i, j) = 0;
					}
				}
			}

		} while (!equalImg(img_xk, img_xk_1));


		// add the border to the filling
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (img.at<uchar>(i, j) == 0) {
					img_xk.at<uchar>(i, j) = 0;
				}
			}
		}

		imshow("In", img);
		imshow("Out", img_xk);
		waitKey();
	}
}

// LAB 8
// The mean value of intensity levels
// The standard deviation of the intensity levels
void meanValue_StdDev() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);
		
		int M = height * width;

		// histogram
		int histo[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				histo[pixel] ++;

			}
		}
		showHistogram("Histogram", histo, 256, 256);

		// pdf
		float pdf[256];
		for (int i = 0; i < 256; i++) {
			pdf[i] = (float)histo[i] / M;
		}

		// mean value
		float mean_value = 0;
		for (int i = 0; i < 256; i++) {
			mean_value += i * histo[i];
		}
		mean_value /= (float)M;
		cout << "Mean value " << mean_value << "\n";

		// standard deviation
		float stdDev = 0;
		for (int i = 0; i < 256; i++) {
			stdDev += pow(i - mean_value, 2) * pdf[i];
		}
		stdDev = sqrt(stdDev);

		cout << "Standard deviation " << stdDev << "\n";

		imshow("input image", src);

		waitKey();
	}

}

// Basic global thresholding algorithm
void globalThreshold()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		int M = height * width;

		// compute image histogram
		int histo[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				histo[pixel] ++;

			}
		}
		showHistogram("Histogram", histo, 256, 256);

		float error = 0.1;

		// find the maximum intensity and minimum intensity
		int imin = 0;
		int imax = 0;
		for (int i = 0; i <= 255; i++)
		{
			if (histo[i] != 0) {
				imin = i;
				break;
			}
		}
		for (int i = 255; i >= 0; i--)
		{
			if (histo[i] != 0) {
				imax = i;
				break;
			}
		}

		// take an initial value for threshold
		float tk1 = (float)(imin + imax) / 2;
		float tk = tk1;
		float N1 = 0;
		float sum1 = 0;
		float N2 = 0;
		float sum2 = 0;
		do {
			// step 2
			N1 = 0; sum1 = 0; N2 = 0; sum2 = 0;
			tk1 = tk;
			for (int i = imin; i <= tk1; i++)
			{
				N1 += histo[i];
				sum1 += i * histo[i];
			}

			float meanG1 = 1 / N1 * sum1;

			for (int i = tk1 + 1; i < imax; i++)
			{
				N2 += histo[i];
				sum2 += i * histo[i];
			}

			float meanG2 = 1 / N2 * sum2;

			// step 3
			tk = (meanG1 + meanG2) / 2;

		} while (abs(tk - tk1) > error);


		// threshold the image using T
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) > tk) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = 0;
				}
			}

		int histo2[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = dst.at<uchar>(i, j);
				histo2[pixel] ++;

			}
		}
		showHistogram("Histogram threshold", histo2, 256, 256);

		imshow("Original image", src);
		imshow("Global Threshold", dst);
		waitKey(0);

	}

}

// Brightness changing
void brightnessChange(int offset) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);


		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++)
			{
				if (offset > 0) {
					// increase the brightness
					dst.at<uchar>(i, j) = min(255, src.at<uchar>(i, j) + offset);
				}
				else {
					// decrease the brightness
					dst.at<uchar>(i, j) = max(0, src.at<uchar>(i, j) + offset);
				}

			}
		}

		// histogram of original image
		int histo1[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				histo1[pixel] ++;

			}
		}
		showHistogram("Histogram original image", histo1, 256, 256);

		// histogram of output image
		int histo2[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = dst.at<uchar>(i, j);
				histo2[pixel] ++;

			}
		}
		showHistogram("Histogram output image", histo2, 256, 256);

		imshow("Original image", src);
		imshow("Output image", dst);
		waitKey(0);
	}
}

// Histogram stretching / shrinking
void histoStretchShrink(int gout_min, int gout_max) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// histogram of original image
		int histo1[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				histo1[pixel] ++;

			}
		}
		showHistogram("Histogram original image", histo1, 256, 256);

		// find GinMIN and GinMAX
		int gin_min = 0;
		int gin_max = 0;
		for (int i = 0; i <= 255; i++)
		{
			if (histo1[i] != 0) {
				gin_min = i;
				break;
			}
		}
		for (int i = 255; i >= 0; i--)
		{
			if (histo1[i] != 0) {
				gin_max = i;
				break;
			}
		}

		// fraction > 1 => stretch
		// fraction < 1 => shrink
		float fraction = (float)(gout_max - gout_min) / (gin_max - gin_min);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++)
			{
				dst.at<uchar>(i, j) = gout_min + 
					((float)(src.at<uchar>(i, j) - gin_min) * fraction);
			}
		}

		// histogram of output image
		int histo2[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = dst.at<uchar>(i, j);
				histo2[pixel] ++;

			}
		}
		showHistogram("Histogram output image", histo2, 256, 256);

		imshow("Original image", src);
		imshow("Output image", dst);
		waitKey(0);
	}
}

// Gamma correction
void gammaCorrection(float gamma) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);


		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++)
			{
				dst.at<uchar>(i, j) = 255 *
					pow(((float)src.at<uchar>(i, j) / 255), gamma);
			}
		}

		// histogram of original image
		int histo1[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				histo1[pixel] ++;

			}
		}
		showHistogram("Histogram original image", histo1, 256, 256);

		// histogram of output image
		int histo2[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = dst.at<uchar>(i, j);
				histo2[pixel] ++;

			}
		}
		showHistogram("Histogram output image", histo2, 256, 256);

		imshow("Original image", src);
		imshow("Output image", dst);
		waitKey(0);
	}
}

// Histogram equalization
void histogramEqualization()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		int M = height * width;

		// histogram of original image
		int histo1[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = src.at<uchar>(i, j);
				histo1[pixel] ++;

			}
		}
		showHistogram("Histogram original image", histo1, 256, 256);

		// pdf
		float pdf[256];
		for (int i = 0; i < 256; i++) {
			pdf[i] = (float)histo1[i] / M;
		}

		// compute cumulative probability density function (CDPF)
		float pc[256] = { 0 };
		float sum = 0;
		for (int i = 0; i < 256; i++) {
			sum = 0;
			for (int k = 0; k < i; k++) {
				sum = sum + pdf[k];

			}
			pc[i] = sum;
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				dst.at<uchar>(i, j) = 255 * pc[src.at<uchar>(i, j)];
			}
		}


		int histo2[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = dst.at<uchar>(i, j);
				histo2[pixel] ++;

			}
		}
		showHistogram("Histogram output image", histo2, 256, 256);

		imshow("Original image", src);
		imshow("Output image", dst);
		waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Mean value of intensity levels\n");
		printf(" 2 - Global threshold\n");
		printf(" 3 - Brightness change\n");
		printf(" 4 - Histogram stretching / shrinking\n");
		printf(" 5 - Gamma correction\n");
		printf(" 6 - Histogram equalization\n");
		printf("Option: "); 
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			meanValue_StdDev();
			break;
		case 2:
			globalThreshold();
			break;
		case 3:
			printf("Introduce offset: ");
			int offset;
			scanf("%d", &offset);
			brightnessChange(offset);
			break;
		case 4:
			printf("Introduce gout_min:\n");
			int gout_min;
			scanf("%d", &gout_min);
			printf("Introduce gout_max:\n");
			int gout_max;
			scanf("%d", &gout_max);
			histoStretchShrink(gout_min, gout_max);
			break;
		case 5:
			printf("Introduce gamma coefficient: \n");
			float gamma;
			scanf("%f", &gamma);
			gammaCorrection(gamma);
			break;
		case 6:
			histogramEqualization();

		}
	} while (op != 0);
	return 0;
}