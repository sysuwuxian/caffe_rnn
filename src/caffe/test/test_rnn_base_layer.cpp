#include <vector>
#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"
#include "fstream"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/rnn_base_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif


template <typename TypeParam>
class RNNBaseLayerTest : public GPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  RNNBaseLayerTest()
      : blob_bottom_(new Blob<Dtype>(6, 5, 1, 1)), blob_bottom_id_(new Blob<Dtype>()),
        blob_bottom_roi_(new Blob<Dtype>()), blob_top_(new Blob<Dtype>()),
		blob_top1_(new Blob<Dtype>()), blob_top2_(new Blob<Dtype>()),
		blob_top3_(new Blob<Dtype>()){
    // fill the values
    
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);    
    

    blob_top_vec_.push_back(blob_top_);

    blob_top_vec_.push_back(blob_top1_);
    blob_top_vec_.push_back(blob_top2_);
    blob_top_vec_.push_back(blob_top3_);

	//LOG(INFO)<<"enter initalizeation";
  }
  virtual ~RNNBaseLayerTest() {
   delete blob_bottom_;
	 delete blob_bottom_id_;
	 delete blob_bottom_roi_;
    //delete blob_bottom_nobatch_;
   delete blob_top_;

   delete blob_top1_;
   delete blob_top2_;
   delete blob_top3_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_id_;
  Blob<Dtype>* const blob_bottom_roi_;
  //Blob<Dtype>* const blob_bottom_nobatch_;
  Blob<Dtype>* const blob_top_;

  Blob<Dtype>* const blob_top1_;
  Blob<Dtype>* const blob_top2_;
  Blob<Dtype>* const blob_top3_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RNNBaseLayerTest, TestDtypesAndDevices);
/*
TYPED_TEST(RNNBaseLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  
  this->blob_bottom_id_->Reshape(1, 1, 1, 1);
  this->blob_bottom_id_->mutable_cpu_data()[0] = 0;

  this->blob_bottom_roi_->Reshape(7, 5, 1, 1);

  //LOG(INFO)<<"TEST SHOULD FIRST ENNTER HER";
  Dtype* ptr = this->blob_bottom_roi_->mutable_cpu_data();

  //read into the blob
  string name = "/media/WORK/caffe_rnn/caffe/data/box_200/003000.txt";
  // string name = "/media/WORK/caffe_rnn/caffe/src/caffe/test/test_data/box/000000.txt";
  vector<int> roi = ReadFileToVector(name);
  for(int i = 0; i < roi.size() / 4; ++i) {
   ptr[0] = 0;
   ptr[1] = roi[4 * i];
   ptr[2] = roi[4 * i + 1];
   ptr[3] = roi[4 * i + 2];
   ptr[4] = roi[4 * i + 3];

   ptr += 5;
  }

  this->blob_bottom_vec_.push_back(this->blob_bottom_id_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_roi_);


   bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;


    RNNBaseParameter* rnn_base_param =
        layer_param.mutable_rnn_base_param();

    rnn_base_param->mutable_weight_filler()->set_type("gaussian");
    rnn_base_param->mutable_bias_filler()->set_type("constant");

  rnn_base_param->set_loss_per_error(0.05);
  rnn_base_param->set_source("/media/WORK/caffe_rnn/caffe/data/trainval_debug.txt");
  rnn_base_param->set_gt_folder("/media/WORK/caffe_rnn/caffe/data/groundth_200/");
  rnn_base_param->set_adj_folder("/media/WORK/caffe_rnn/caffe/data/adj_200/");
  rnn_base_param->set_class_folder("/media/WORK/caffe_rnn/caffe/data/label_200/");

  
  rnn_base_param->set_source("/media/WORK/caffe_rnn/caffe/src/caffe/test/test_data/val.txt");
  rnn_base_param->set_gt_folder("/media/WORK/caffe_rnn/caffe/src/caffe/test/test_data/groundth/");
  rnn_base_param->set_adj_folder("/media/WORK/caffe_rnn/caffe/src/caffe/test/test_data/adj/");
  rnn_base_param->set_class_folder("/media/WORK/caffe_rnn/caffe/src/caffe/test/test_data/label/");
  
  rnn_base_param->set_batch_size(1);
  rnn_base_param->set_min_iou(0.3);
  rnn_base_param->set_max_iou(0.5);
    //LOG(ERROR)<<"enter here";
  RNNBaseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(layer.get_batch_size(), 1);
  EXPECT_EQ(layer.get_valid(), true);
  EXPECT_EQ(layer.is_tree(), true);
  

  const Dtype* data = this->blob_top_->cpu_data();
  EXPECT_FLOAT_EQ(data[0], -layer.test_score());

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
  




}*/

TYPED_TEST(RNNBaseLayerTest, TestGradient) {

  //LOG(INFO)<<"enter here";
  typedef typename TypeParam::Dtype Dtype;

  /*     
  Dtype* data = this->blob_bottom_->mutable_cpu_data();
  const int count = this->blob_bottom_->count();
  
  for(int i = 0; i < count; i++){
      data[i] = 0.016;
  }*/
  


  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  
  this->blob_bottom_id_->Reshape(1, 1, 1, 1);
  this->blob_bottom_id_->mutable_cpu_data()[0] = 0;
 // this->blob_bottom_id_->mutable_cpu_data()[1] = 0;

  this->blob_bottom_roi_->Reshape(6, 5, 1, 1);

  //LOG(INFO)<<"TEST SHOULD FIRST ENNTER HER";
  Dtype* ptr = this->blob_bottom_roi_->mutable_cpu_data();

  //read into the blob
  string name = "./data/train/box_sz_250/003000.txt";
  //string name = "./src/caffe/test/test_data/box/000000.txt";
  
  vector<int> roi = ReadFileToVector(name);
  for(int i = 0; i < roi.size() / 4; ++i) {
	 ptr[0] = 0;
	 ptr[1] = roi[4 * i];
	 ptr[2] = roi[4 * i + 1];
	 ptr[3] = roi[4 * i + 2];
	 ptr[4] = roi[4 * i + 3];

	 ptr += 5;
  }
  LOG(INFO)<<"roi size is "<<roi.size() / 4;

  this->blob_bottom_vec_.push_back(this->blob_bottom_id_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_roi_);






  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;

      
    RNNBaseParameter* rnn_base_param =
        layer_param.mutable_rnn_base_param();

    rnn_base_param->mutable_weight_filler()->set_type("gaussian");


    //rnn_base_param->mutable_weight_filler()->set_type("constant");
    //rnn_base_param->mutable_weight_filler()->set_value(0.5);   

    //rnn_base_param->mutable_bias_filler()->set_type("constant");
    
    rnn_base_param->mutable_bias_filler()->set_type("gaussian");
    rnn_base_param->mutable_bias_filler()->set_min(1);
    rnn_base_param->mutable_bias_filler()->set_max(2);
    
	rnn_base_param->set_loss_per_error(0.05);

    
	rnn_base_param->set_source("./data/trainval_debug.txt");
	rnn_base_param->set_gt_folder("./data/train/groundth/");
	rnn_base_param->set_adj_folder("./data/train/adj_sz_250/");
	rnn_base_param->set_class_folder("./data/train/label_sz_250/");
  	
 
  /*
	rnn_base_param->set_source("./src/caffe/test/test_data/val.txt");
	rnn_base_param->set_gt_folder("./src/caffe/test/test_data/groundth/");
	rnn_base_param->set_adj_folder("./src/caffe/test/test_data/adj/");
	rnn_base_param->set_class_folder("./src/caffe/test/test_data/label/");
  */
  rnn_base_param->set_batch_size(1);
	rnn_base_param->set_min_iou(0.3);
	rnn_base_param->set_max_iou(0.5);

    //LOG(ERROR)<<"enter here";
    RNNBaseLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);

    checker.CheckGradientSingle(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0, 2, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


}  // namespace caffe
