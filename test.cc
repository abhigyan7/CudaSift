#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include "cudaImage.h"
#include "cudaSift.h"


void convert_kpts_descriptors_to_opencv(SiftData* cuda_sift_data, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {

  keypoints.resize(cuda_sift_data->numPts);
  for (int i = 0; i < cuda_sift_data->numPts; ++i) {
    SiftPoint sp = cuda_sift_data->h_data[i];
    keypoints[i] = cv::KeyPoint(
      cv::Point2f(sp.xpos, sp.ypos),
      sp.scale,
      sp.orientation,
      sp.score,
      sp.subsampling,
      sp.match
    );
  }

  // Convert SiftData to Mat Descriptor
  std::vector<float> data;
  for (int i = 0; i < cuda_sift_data->numPts; i++)
  {
    data.insert(data.end(), cuda_sift_data->h_data[i].data, cuda_sift_data->h_data[i].data + 128);
  }

  cv::Mat tempDescriptor(cuda_sift_data->numPts, 128, CV_32FC1, &data[0]);
  tempDescriptor.copyTo(descriptors); // Inefficient!

}

void convert_matches_to_opencv(SiftData* sift_data_1, std::vector<cv::DMatch> &matches) {
  for (int i = 0; i < sift_data_1->numPts; i++) {
    SiftPoint sp = sift_data_1->h_data[i];
    cv::DMatch match(i, sp.match, sp.score);
    matches.push_back(match);
  }
}

/* Reserve memory space for a whole bunch of SIFT features. */
SiftData siftData1, siftData2;

int main(int argc, char **argv) {
  InitSiftData(siftData1, 25000, true, true);
  InitSiftData(siftData2, 25000, true, true);

  /* Read image using OpenCV and convert to floating point. */
  cv::Mat limg1, limg2;
  cv::imread(argv[1], 0).convertTo(limg1, CV_32FC1);
  cv::imread(argv[2], 0).convertTo(limg2, CV_32FC1);
  cv::Mat uimg1 = cv::imread(argv[1], 0);
  cv::Mat uimg2 = cv::imread(argv[2], 0);
  /* Allocate 1280x960 pixel image with device side pitch of 1280 floats. */
  /* Memory on host side already allocated by OpenCV is reused.           */
  CudaImage img1, img2;
  unsigned int w = limg1.cols;
  unsigned int h = limg1.rows;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)limg1.data);
  w = limg2.cols;
  h = limg2.rows;
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)limg2.data);
  /* Download image from host to device */
  img1.Download();
  img2.Download();

  int numOctaves = 5; /* Number of octaves in Gaussian pyramid */
  float initBlur =
      1.0f; /* Amount of initial Gaussian blurring in standard deviations */
  float thresh =
      3.5f; /* Threshold on difference of Gaussians for feature pruning */
  float minScale =
      0.0f; /* Minimum acceptable scale to remove fine-scale features */
  bool upScale = false; /* Whether to upscale image before extraction */
  /* Extract SIFT features */
  ExtractSift(siftData1, img1, numOctaves, initBlur, thresh, minScale, upScale);
  ExtractSift(siftData2, img2, numOctaves, initBlur, thresh, minScale, upScale);

  cv::Mat descriptors_1, descriptors_2;
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

  convert_kpts_descriptors_to_opencv(&siftData1, keypoints_1, descriptors_1);
  convert_kpts_descriptors_to_opencv(&siftData2, keypoints_2, descriptors_2);

  // show the keypoints
  cv::Mat kps_img;
  cv::drawKeypoints(uimg2, keypoints_2, kps_img);
  cv::imshow("image", kps_img);
  while (true) {
    char k = cv::waitKey(33);
    if (k == 'q')
      break;
  }
  cv::destroyAllWindows();

  MatchSiftData(siftData1, siftData2);
  std::vector<cv::DMatch> matches;
  convert_matches_to_opencv(&siftData1, matches);

  cv::Mat out;
  cv::drawMatches(uimg1, keypoints_1, uimg2, keypoints_2, matches, out);

  cv::imshow("image", out);
  while (true) {
    char k = cv::waitKey(33);
    if (k == 'q')
      break;
  }
  cv::destroyAllWindows();

  /* Free space allocated from SIFT features */
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

