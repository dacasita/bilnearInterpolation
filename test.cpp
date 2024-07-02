
#include <vector>

#include "myfunctions.h"
#include <opencv2/opencv.hpp>

using namespace cv;

Mat bilinearInterpolation(const Mat& src, double scale_x, double scale_y) {
  // Get source image size
  int src_rows = src.rows;
  int src_cols = src.cols;

  // Calculate target image size
  int dst_rows = int(src_rows * scale_y + 0.5);
  int dst_cols = int(src_cols * scale_x + 0.5);

  // Create target image with the same data type as the source
  Mat dst(dst_rows, dst_cols, src.type());

  // Loop through each pixel in the target image
  for (int y = 0; y < dst_rows; y++) {
    for (int x = 0; x < dst_cols; x++) {
      // Calculate normalized coordinates in the source image
      double norm_x = (x + 0.5) / scale_x;
      double norm_y = (y + 0.5) / scale_y;

      // Find integer coordinates of the surrounding pixels in the source image
      int y1 = floor(norm_y);
      int y2 = y1 + 1;
      int x1 = floor(norm_x);
      int x2 = x1 + 1;

      // Check for boundary conditions (clamp to avoid accessing outside the image)
      y1 = std::max(0, std::min(y1, src_rows - 2));
      y2 = std::max(0, std::min(y2, src_rows - 1));
      x1 = std::max(0, std::min(x1, src_cols - 2));
      x2 = std::max(0, std::min(x2, src_cols - 1));

      // Calculate distance weights for interpolation
      double dx = norm_x - x1;
      double dy = norm_y - y1;

      // Access pixel values from the source image (handle channel access for color images)
      Vec3b p1 = src.at<Vec3b>(y1, x1);
      Vec3b p2 = src.at<Vec3b>(y1, x2);
      Vec3b p3 = src.at<Vec3b>(y2, x1);
      Vec3b p4 = src.at<Vec3b>(y2, x2);

      // Perform bilinear interpolation for each color channel (if applicable)
      Vec3b interpolated;
      for (int c = 0; c < src.channels(); ++c) {
        interpolated[c] = (1 - dx) * ((1 - dy) * p1[c] + dy * p3[c]) + dx * ((1 - dy) * p2[c] + dy * p4[c]);
      }

      // Set the interpolated value in the target image
      dst.at<Vec3b>(y, x) = interpolated;
    }
  }

  return dst;
}

int main() {
  // Load your source image
  Mat src = imread("face.jpg");

  // Define scaling factors (change as needed)
  double scale_x = 1.5;
  double scale_y = 1.6;

  // Perform bilinear interpolation
  Mat dst = bilinearInterpolation(src, scale_x, scale_y);

  // Display the resized image (optional)
  imshow("Original Image", src);
  imshow("Resized Image", dst);
  waitKey(0);


return 0;
}