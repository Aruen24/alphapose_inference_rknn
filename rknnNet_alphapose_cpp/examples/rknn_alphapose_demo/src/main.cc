// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"
#include "rknnNet.h"
#include <thread>

using namespace std;
using namespace cv;

#define pose_n 1
#define pose_h 64
#define pose_w 48
#define pose_c 17


double GetTickCount1() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000000 + ts.tv_nsec / 1000)/1000.0;
}

static void get_max_pred(float* hms, float* maxvals, int preds[][2]) {
    const int num_joints = pose_c;
    const int width = pose_w;
    int idx[num_joints];

    for (int i = 0; i < num_joints; ++i) {
        float* max_ptr = std::max_element(&(hms[i * pose_h * pose_w]), &(hms[i * pose_h * pose_w]) + pose_h * pose_w);
        maxvals[i] = *max_ptr;
        idx[i] = std::distance(&(hms[i * pose_h * pose_w]), max_ptr);
        //std::cout << "maxvals: " << maxvals[i] << " idx: " << idx[i] << std::endl;
        preds[i][0] = idx[i] % width;
        preds[i][1] = idx[i] / width;
    }
    std::cout << std::endl;
}

static cv::Point2f transform_preds(cv::Point2f pred, cv::Point2f center, cv::Point2f scale, cv::Point2f output_size) {
    float src_w = scale.x;
    float dst_w = output_size.x;
    float dst_h = output_size.y;

    cv::Point2f src_dir(0, src_w * (-0.5));
    cv::Point2f dst_dir(0, dst_w * (-0.5));

    cv::Point2f src[3], dst[3];
    src[0].x = center.x;
    src[0].y = center.y;
    src[1].x = center.x + src_dir.x;
    src[1].y = center.y + src_dir.y;

    dst[0].x = dst_w * 0.5;
    dst[0].y = dst_h * 0.5;
    dst[1].x = dst_w * 0.5 + dst_dir.x;
    dst[1].y = dst_h * 0.5 + dst_dir.y;

    cv::Point2f direct_src(src[0].x - src[1].x, src[0].y - src[1].y);
    src[2].x = src[1].x - direct_src.y;
    src[2].y = src[1].y + direct_src.x;

    cv::Point2f direct_dst(dst[0].x - dst[1].x, dst[0].y - dst[1].y);
    dst[2].x = dst[1].x - direct_dst.y;
    dst[2].y = dst[1].y + direct_dst.x;

    /*
    for(int i=0; i<3; ++i) {
        std::cout << src[i].x << " " << src[i].y << std::endl;
    }
    for(int i=0; i<3; ++i) {
        std::cout << dst[i].x << " " << dst[i].y << std::endl;
    }

    std::cout << std::endl;*/

    cv::Mat trans = cv::getAffineTransform(dst, src);
    cv::Point2f target_coords;
    target_coords.x = trans.at<double>(0, 0) * pred.x + trans.at<double>(0, 1) * pred.y + trans.at<double>(0, 2);
    target_coords.y = trans.at<double>(1, 0) * pred.x + trans.at<double>(1, 1) * pred.y + trans.at<double>(1, 2);

    //std::cout << "pred-----: " << pred.x << " " << pred.y << std::endl;
    /*
    for (int i=0; i<trans.rows ; i++) {
        for (int j=0; j<trans.cols ; j++) {
            std::cout << trans.at<double>(i,j) << " ";
        }
        std::cout << std::endl;
    }*/

    return target_coords;
}

static std::vector<cv::Point2f> heatmap_to_coord_simple(float hms[][pose_h][pose_w], cv::Rect bbox, std::vector<float>* maxvals) {
    float max_vals[pose_c];
    int coords[pose_c][2];
    get_max_pred((float*)hms, max_vals, coords);

    maxvals->clear();
    for (int i = 0; i < pose_c; ++i) {
        maxvals->push_back(max_vals[i]);
    }

    std::vector<cv::Point2f> trans_preds;

    cv::Point2f preds[pose_c];
    for (int i = 0; i < pose_c; ++i) {
        int px = coords[i][0];
        int py = coords[i][1];
        if (px > 1 && px < pose_w - 1 && py>1 && py < pose_h - 1) {
            float diff_x = hms[i][py][px + 1] - hms[i][py][px - 1];
            float diff_y = hms[i][py + 1][px] - hms[i][py - 1][px];
            float sign_x = -0.25;
            float sign_y = -0.25;
            if (diff_x > 0) sign_x = 0.25;
            if (diff_y > 0) sign_y = 0.25;
            preds[i].x = coords[i][0] + sign_x;
            preds[i].y = coords[i][1] + sign_y;
        }
        else {
            preds[i].x = coords[i][0];
            preds[i].y = coords[i][1];
        }
    }

    cv::Point2f center(bbox.x + bbox.width * 0.5, bbox.y + bbox.height * 0.5);
    cv::Point2f scale(bbox.width, bbox.height);

    for (int i = 0; i < pose_c; ++i) {
        trans_preds.push_back(transform_preds(preds[i], center, scale, cv::Point2f(pose_w, pose_h)));
    }

    return trans_preds;
}





/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    const int MODEL_IN_WIDTH = 192;
    const int MODEL_IN_HEIGHT = 256;
    const int MODEL_IN_CHANNELS = 3;

    int ret;


    const char *model_path = argv[1];
    const char *img_path = argv[2];
    auto t0=GetTickCount1();
    
    
    rknnNet net = rknnNet(model_path, 1);
    auto t1=GetTickCount1();
    for(int i =0; i < 1000000; i++){
      auto t2=GetTickCount1();
      
      // Load image
      cv::Mat orig_img = imread(img_path, cv::IMREAD_COLOR);
      if(!orig_img.data) {
          printf("cv::imread %s fail!\n", img_path);
          return -1;
      }
  
      cv::Mat img = orig_img.clone();
      if(orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
          printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
          cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
      }
  
      cv::Mat img_input;
      cv::cvtColor(img, img_input, cv::COLOR_BGR2RGB);

      rknnSession* sess = net.getSession();
      //rknnSession* sess = net.forkSession();
  
      rknn_output* outputs = new rknn_output[sess->io_num.n_output];
      ret = sess->forward(img_input, outputs);
      if(ret < 0) {
          printf("rknn_inference fail! ret=%d\n", ret);
          return -1;
      }
  
      
      printf("rknn_main output_size=%d\n", sess->io_num.n_output);
      std::vector< std::vector<cv::Point2f> > results;
      std::vector< std::vector<float> > max_scores;
      
      int batch_size = 1;
      cv::Rect rects;
      rects.x = -3;
      rects.y = -11;
      rects.width = 188;
      rects.height = 251;
      float threshold = 0.05;
  
      // Post Process
      //for (int i = 0; i < sess->io_num.n_output; i++)
      //{
  
      	float *buffer = (float *)outputs[i].buf;
          uint32_t sz = outputs[i].size/4;
          
          float out[pose_n][pose_c][pose_h][pose_w];//1 17 64 48
          for (int j = 0; j < pose_n; ++j) {
              for (int k = 0; k < pose_c; ++k) {
                  for (int m = 0; m < pose_h; ++m) {
                      for (int n = 0; n < pose_w; ++n) {
                          //out[j][k][m][n] = land_out[0][pose_w*pose_h*pose_c*j+pose_c*pose_w*m+pose_c*n+k];
                          out[j][k][m][n] = buffer[pose_w * pose_h * pose_c * j + pose_h * pose_w * k + pose_w * m + n];
                      }
                  }
              }
  
              //cv::Rect test(0, 0, 192, 256);
              //if (i * batch_size + j < inputs.size()) {
              if (i * batch_size + j < 1) {
                  std::vector<float> maxvals;
                  std::vector<cv::Point2f> trans_preds = heatmap_to_coord_simple(out[j], rects, &maxvals);
                  //heatmap_to_coord_simple(out[j], test, trans_preds);
                  results.push_back(trans_preds);
                  max_scores.push_back(maxvals);
              }
          }
      		
         
      }
      
      
      for(int i=0; i<results.size(); ++i) {
          for(int j=0; j<pose_c; ++j) {
              std::cout << "img_point: " << results[i][j].x << " " << results[i][j].y << std::endl;
              std::cout << "img_score: " << max_scores[i][j] << std::endl;
          }
          if(max_scores[i][0] > threshold && max_scores[i][1] > threshold &&
             max_scores[i][2] > threshold && max_scores[i][3] > threshold &&
             max_scores[i][4] > threshold) {
              std::cout << "draw_point " << std::endl;
              cv::circle(img, results[i][0], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][1], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][2], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][3], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][4], 3, Scalar(255,0,0), -1);
    
              cv::line(img, results[i][0], results[i][1], cv::Scalar(0,255,0));
              cv::line(img, results[i][0], results[i][2], cv::Scalar(0,255,0));
              cv::line(img, results[i][1], results[i][3], cv::Scalar(0,255,0));
              cv::line(img, results[i][2], results[i][4], cv::Scalar(0,255,0));
          }
    
          if(max_scores[i][5] > threshold && max_scores[i][6] > threshold &&
             max_scores[i][7] > threshold && max_scores[i][8] > threshold &&
             max_scores[i][9] > threshold && max_scores[i][10]> threshold) {
              cv::circle(img, results[i][5], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][6], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][7], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][8], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][9], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][10], 3, Scalar(255,0,0), -1);
    
              cv::line(img, results[i][5], results[i][6], cv::Scalar(0,255,0));
              cv::line(img, results[i][5], results[i][7], cv::Scalar(0,255,0));
              cv::line(img, results[i][7], results[i][9], cv::Scalar(0,255,0));
              cv::line(img, results[i][6], results[i][8], cv::Scalar(0,255,0));
              cv::line(img, results[i][8], results[i][10], cv::Scalar(0,255,0));
          }
    
          if(max_scores[i][5] > threshold && max_scores[i][6] > threshold &&
             max_scores[i][11] > threshold && max_scores[i][12] > threshold) {
              cv::circle(img, results[i][11], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][12], 3, Scalar(255,0,0), -1);
    
              cv::line(img, cv::Point2f((results[i][5].x+results[i][6].x)*0.5, 
                  (results[i][5].y+results[i][6].y)*0.5), results[i][11], cv::Scalar(0,255,0));
              cv::line(img, cv::Point2f((results[i][5].x+results[i][6].x)*0.5, 
                  (results[i][5].y+results[i][6].y)*0.5), results[i][12], cv::Scalar(0,255,0));
          }
    
          if(max_scores[i][11] > threshold && max_scores[i][12] > threshold &&
             max_scores[i][13] > threshold && max_scores[i][14] > threshold &&
             max_scores[i][15] > threshold && max_scores[i][16] > threshold) {
              cv::circle(img, results[i][11], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][12], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][13], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][14], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][15], 3, Scalar(255,0,0), -1);
              cv::circle(img, results[i][16], 3, Scalar(255,0,0), -1);
    
              cv::line(img, results[i][11], results[i][13], cv::Scalar(0,255,0));
              cv::line(img, results[i][12], results[i][14], cv::Scalar(0,255,0));
              cv::line(img, results[i][13], results[i][15], cv::Scalar(0,255,0));
              cv::line(img, results[i][14], results[i][16], cv::Scalar(0,255,0));
          }
          std::cout << std::endl;
      }
    
      cv::imwrite("./result.jpg", img);
      rknn_outputs_release(ctx, sess->io_num.n_output, outputs);
      delete[] outputs;
      auto t3=GetTickCount1();
      std::cout << "initNet time: " << t1-t0 << std::endl;
      std::cout << "preprocess data+inference+after deal time: " << t3-t2 << std::endl;
    //}
    
    return 0;
}






/*-------------------------------------------
        Main Function Mliti Thread
-------------------------------------------*/
/*cv::Mat img;
int ret;
void threadFunc(rknnSession* sess, int times, int id) {
    std::cout << "thread id: " << id << " begin" << std::endl;

    for(int i=0; i<times; ++i) {
        rknn_output* outputs = new rknn_output[sess->io_num.n_output];
        ret = sess->forward(img, outputs);
        if(ret < 0) {
            printf("rknn_inference fail! ret=%d\n", ret);
            exit(0);
        }
    
        
        printf("rknn_main output_size=%d\n", sess->io_num.n_output);
        // Post Process
        for (int i = 0; i < sess->io_num.n_output; i++)
        {
            uint32_t MaxClass[5];
        		float fMaxProb[5];
        		float *buffer = (float *)outputs[i].buf;
            uint32_t sz = outputs[i].size/4;
        		
            printf("rknn_main 66666666666 sz=%d\n", sz);
        
        	  rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 5);
        
        		printf(" --- Top5 ---\n");
        		for(int i=0; i<5; i++)
        		{
        			printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
        		}
        }
        rknn_outputs_release(ctx, sess->io_num.n_output, outputs);
        delete[] outputs;
    }

    std::cout << "thread id: " << id << " end" << std::endl;
}

int main(int argc,char *argv[]) {
    const int MODEL_IN_WIDTH = 224;
    const int MODEL_IN_HEIGHT = 224;
    const int MODEL_IN_CHANNELS = 3;

    rknn_context ctx;
    
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    const char *img_path = argv[2];

    // Load image
    cv::Mat orig_img = imread(img_path, cv::IMREAD_COLOR);
    if(!orig_img.data) {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }

    img = orig_img.clone();
    if(orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
        printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
        cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
    }
    
    rknnNet net = rknnNet(model_path);

    thread tasks[10];
    for(int i=0; i<10; ++i) {
        rknnSession* sess = net.forkSession();
        tasks[i] = thread(threadFunc, sess, 10, i);
    }
    for(int i=0; i<10; ++i) {
        tasks[i].join();
    }

    return 0;
}*/
