#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>



namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  //const bool is_color  = this->layer_param_.image_data_param().is_color();
  string feat_folder = this->layer_param_.image_data_param().feat_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  while (infile >> filename) {
    lines_.push_back(filename);
	//used for query!
	tmp_lines_.push_back(filename);
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  //cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_] + ".jpg",
  //                                  new_height, new_width, is_color);
  //CHECK(cv_img.data) << "Could not load " << lines_[lines_id_];
  // Use data_transformer to infer the expected blob shape from a cv_image.
  //vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  //this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  vector<int> top_shape(4, 0);
  top_shape[0] = batch_size;
  top_shape[1] = 315;
  top_shape[2] = 1;
  top_shape[3] = 1;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  top[2]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
	this->prefetch_[i].label1_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ReadFeat(Dtype* feat, string filename, int sz) {
  
  std::ifstream fin(filename.c_str(), std::ios::in);
  std::istringstream istr;

  string str;
  Dtype data;
 
  int id = 0;


  int line = 0;
  while(getline(fin, str)) {
    istr.str(str);
    line += 1;
    int col = 0;

    while(istr >> data){
      col += 1;
      feat[id] = data;
      if(data < 0.0 || data > 1.0)
        LOG(INFO)<<"shut up"<<filename;
      id += 1;
    }
    if(col != 315)
      LOG(INFO)<<"col error "<<filename<<" file not exist ";


    istr.clear();
  }

  if(line != sz)
    LOG(INFO)<<line<<" "<<sz<<" Error "<<filename<<" file not exist !";
  fin.close();
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  //CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  //const int new_height = image_data_param.new_height();
  //const int new_width = image_data_param.new_width();
  //const bool is_color = image_data_param.is_color();
  string roi_folder = image_data_param.roi_folder();
  string feat_folder = image_data_param.feat_folder();
  // Use data_transformer to infer the expected blob shape from a cv_img.
  //vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  vector<int> top_shape(4);

  // we need know roi_num
  int roi_num = 0;
  //batch->label_.Reshape(roi_num, 5, 1, 1);

  Dtype* prefetch_label1 = batch->label1_.mutable_cpu_data();
  vector<vector<int> > batch_roi;

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    //cv::Mat cv_img = ReadImageToFasterCVMat(root_folder + lines_[lines_id_] + ".jpg");
    //CHECK(cv_img.data) << "Could not load " << lines_[lines_id_];
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    //int offset = batch->data_.offset(item_id);
    //this->transformed_data_.set_cpu_data(prefetch_data + offset);
    //this->data_transformer_->Transform_fill(cv_img, &(this->transformed_data_));

	  //print the image
	




    trans_time += timer.MicroSeconds();
	//get prefetch_label
    int id = -1;
	  for(int i = 0; i < tmp_lines_.size(); i++){
		  if(lines_[lines_id_] == tmp_lines_[i]){
		    id = i;
		    break;
		  }
	  }

	  prefetch_label1[item_id] = id;

	  vector<int> roi = ReadFileToVector(roi_folder + lines_[lines_id_] + ".txt");
	  
    assert(roi.size() % 4 == 0);
    roi_num += roi.size() / 4;
	  batch_roi.push_back(roi);

    //prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  
  //get data
  batch->data_.Reshape(roi_num, 315, 1, 1);
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  //get label
  batch->label_.Reshape(roi_num, 5, 1, 1);
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  //int top_index = 0;
  
  for(int item_id = 0; item_id < batch_size; ++item_id) {
	  
    // read the data
    int id = prefetch_label1[item_id];
    
    
    vector<int> roi = batch_roi[item_id];

    // test feat
    ReadFeat(prefetch_data, feat_folder + tmp_lines_[id] + ".txt", roi.size() / 4);

    prefetch_data += roi.size() / 4 * 315; 
	  
    
    for(int i = 0; i < roi.size() / 4; ++i) {
	     prefetch_label[0] = item_id;
		 prefetch_label[1] = roi[4 * i];
		 prefetch_label[2] = roi[4 * i + 1];
		 prefetch_label[3] = roi[4 * i + 2];
		 prefetch_label[4] = roi[4 * i + 3];


		 prefetch_label += 5;
	  }
  }



  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
