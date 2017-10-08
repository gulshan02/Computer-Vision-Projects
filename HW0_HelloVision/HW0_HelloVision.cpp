#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  int a, n;
  Mat new_image;

  Mat image = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);

  if (!image.data) {
    cout << "No image data \n";
    return -1;
  }

  namedWindow("My window", WINDOW_AUTOSIZE);  // Create a window for display.
  imshow("My window", image);                 // Display the image

  waitKey(0);

  cout << "Select one of the options from the following:\n";
  cout << "1: Enter the intensity to add to the pixel\n";
  cout << "2: Enter the intensity to subtract from the pixel\n";
  cout << "3: Enter the intensity to multiply to the pixel\n";
  cout << "4: Enter the intensity to divide from the pixel\n";
  cout << "5: Resize the image by factor of 0.5\n";
  cin >> a;

  switch (a) {
    case 1:
      cout << "Enter the intensity to add to the image pixels:";
      cin >> n;
      add(image, n, new_image);
      break;
    case 2:
      cout << "Enter the intensity to subtract from the image pixels:";
      cin >> n;
      subtract(image, n, new_image);

      break;
    case 3:
      cout << "Enter the intensity to multipy to the image pixels:";
      cin >> n;
      multiply(image, n, new_image);

      break;
    case 4:
      cout << "Enter the intensity to divide to the image pixels:";
      cin >> n;
      divide(image, n, new_image);
      break;
    case 5:
      cout << "Resizing the image";
      resize(image, new_image, cv::Size(), 0.50, 0.50);

      break;
  }

  namedWindow("My window_updated",
              WINDOW_AUTOSIZE);  // Create a window for display.
  imshow("My window_updated", new_image);
  waitKey(0);

  return 0;
}
