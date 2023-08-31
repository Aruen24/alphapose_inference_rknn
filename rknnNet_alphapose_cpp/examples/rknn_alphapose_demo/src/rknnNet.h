#ifndef RKNN_NET_H
#define RKNN_NET_H

#include <android/log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "im2d.h"
#include "RgaUtils.h"
#include "rga.h"
#include "rknn_api.h"
//#include "YoloV5.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "rkssd4j", ##__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "rkssd4j", ##__VA_ARGS__);


using namespace std;
using namespace cv;
using std::vector;

class rknnSession;

class rknnNet {
public:


    rknnNet(const char* model_path_, int out_type);
    ~rknnNet();

    rknnSession* getSession();
    rknnSession* forkSession();
    void releaseSession(rknnSession* sess);
    int                     model_size;
    int                     type;
    // init rga context zero copy
    rga_buffer_t            src;
    rga_buffer_t            dst;
    im_rect                 src_rect;
    im_rect                 dst_rect;
    


private:
    
    rknnSession*            sess;
    const char*             model_path; 
    unsigned char*          model;
    
};

class rknnSession {
public:
    int    input_channel;
    int    input_width;                 //网络输入的宽度
    int    input_height;                //网络输入的高度
    int    batch_size;                  //网络的batch
    int    ret=0;
    rknn_context            ctx;        //推理上下文

    rknnSession(unsigned char* model, int model_size, int type);
    ~rknnSession();
    int forward(cv::Mat data, rknn_output* outputs);
                 
    //rknn_sdk_version          version;
    rknn_input_output_num     io_num;     //输入、输出结构体数据
    //rknn_tensor_attr          input_attrs;
    //rknn_tensor_attr          output_attrs; 
    std::vector<float>        out_scales;
    std::vector<int32_t>      out_zps; 
    int                       out_type;
    
    
private:
    rga_buffer_t              src;
    rga_buffer_t              dst;
    im_rect                   dst_rect;
    im_rect                   src_rect;


};


#endif  //SSD_IMAGE_SSD_IMAGE_H

