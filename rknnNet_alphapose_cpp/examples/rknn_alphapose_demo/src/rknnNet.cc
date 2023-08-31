#include "rknnNet.h"

rknnNet::rknnNet(const char* model_path, int type):type(type) {
    FILE *fp = fopen(model_path, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", model_path);
        exit(0);
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", model_path);
        free(model);
        exit(0);
    }
    model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    
    // init rga context zero copy
    /*memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));*/
    
    sess = new rknnSession(model, model_size, type);
}

rknnNet::~rknnNet() {
    if(sess != NULL) delete sess;
    if(model != NULL) free(model);
}

rknnSession* rknnNet::getSession() {
    if(sess == NULL) sess = new rknnSession(model, model_size, type);
    return sess;
}


rknnSession* rknnNet::forkSession() {
    if(sess == NULL) return NULL;
    return new rknnSession(model, model_size, type);
}

void rknnNet::releaseSession(rknnSession* sess) {
    if(sess != NULL) delete sess;
}

rknnSession::rknnSession(unsigned char* model, int model_size, int type) {
    printf("rknn_init...\n");   
    out_type = type;
    ret = rknn_init(&ctx, model, model_size, 0, NULL);
    
    //rknn_context ctx2;
    //ret = rknn_init(&ctx2, model, model_size, 0, NULL);
    printf("rknn_init doing!!!!!!!!!!!!! ret=%d\n", ret);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        exit(0);
    }
    printf("rknn_init successed!!!!");
    // Get Model Input Output Version Info
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(0);
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);

    //rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(0);
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            exit(0);
        }

        printf("input tensor:\n");
        printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            input_attrs[i].index, input_attrs[i].name, input_attrs[i].n_dims, input_attrs[i].dims[0], input_attrs[i].dims[1], input_attrs[i].dims[2], input_attrs[i].dims[3], 
            input_attrs[i].n_elems, input_attrs[i].size, 0, input_attrs[i].type, input_attrs[i].qnt_type, input_attrs[i].fl, input_attrs[i].zp, input_attrs[i].scale);
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    out_scales.clear();
    out_zps.clear();
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
                         
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);

        printf("output tensor:\n");
        printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            output_attrs[i].index, output_attrs[i].name, output_attrs[i].n_dims, output_attrs[i].dims[0], output_attrs[i].dims[1], output_attrs[i].dims[2], output_attrs[i].dims[3], 
            output_attrs[i].n_elems, output_attrs[i].size, 0, output_attrs[i].type, output_attrs[i].qnt_type, output_attrs[i].fl, output_attrs[i].zp, output_attrs[i].scale);
    }
    
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        input_channel = input_attrs[0].dims[1];
        input_width = input_attrs[0].dims[2];
        input_height = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        input_width = input_attrs[0].dims[1];
        input_height = input_attrs[0].dims[2];
        input_channel = input_attrs[0].dims[3];
    }

    
}


rknnSession::~rknnSession() {
    rknn_destroy(ctx);
}

int rknnSession::forward(cv::Mat img, rknn_output* outputs) {
    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = img.cols*img.rows*img.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    
    //yolov5 zero copy
    /*inputs[0].pass_through = 0;
    void *resize_buf = malloc(input_height * input_width * input_channel);

    src = wrapbuffer_virtualaddr((void *)img.data, img.cols, img.rows, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *)resize_buf, input_width, input_height, RK_FORMAT_RGB_888);
    ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }
    //IM_STATUS STATUS = imresize(src, dst);
    cv::Mat resize_img(cv::Size(input_width, input_height), CV_8UC3, resize_buf);
    //cv::imwrite("resize_input.jpg", resize_img);

    inputs[0].buf = resize_buf;*/
    

    inputs[0].buf = img.data;
    
    printf("rknn_forward doing!!!!!!!!!!!!! inputs[0].sizet=%d\n", img.cols);

   
    
    
    //gettimeofday(&start_time, NULL);
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }
    printf("rknn_forward doing 11111111111111111 !!!!!!!!!!!!! \n");

    // Run
    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    printf("rknn_forward doing 22222222222222222 !!!!!!!!!!!!! output_size=%d\n", sizeof(outputs));

    // Get Output one_output 24b   pointerr 8b
    //rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs)*3*io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++)
    {
        //0-8  1-8
        //0-8  1-8  0-int8_t(output)  1-float(output)
        if(out_type==0){
             outputs[i].want_float = 0;
        }else{
            outputs[i].want_float = 1;
        }
    }
    printf("rknn_forward doing 333333333333333333333333 !!!!!!!!!!!!! \n");

    
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    printf("rknn_forward doing 444444444444444 !!!!!!!!!!!!! \n");

    
    //gettimeofday(&stop_time, NULL);
    //printf("once run use %f ms\n",
    //       (__get_us(stop_time) - __get_us(start_time)) / 1000);

    return 1;
}

