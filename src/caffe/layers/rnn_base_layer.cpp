#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include "caffe/util/benchmark.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/rnn_base_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
namespace caffe {

template <typename Dtype>
void RNNBaseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
	   const vector<Blob<Dtype>*>& top){
     num_ = bottom[0]->num();
	 channels_ = bottom[0]->channels();

	 batch_size = this->layer_param_.rnn_base_param().batch_size();
	 min_iou = this->layer_param_.rnn_base_param().min_iou();
	 max_iou = this->layer_param_.rnn_base_param().max_iou();
     
	 gt_folder = this->layer_param_.rnn_base_param().gt_folder();
	 adj_folder = this->layer_param_.rnn_base_param().adj_folder();
     class_folder = this->layer_param_.rnn_base_param().class_folder();

     loss_per_error_ = this->layer_param_.rnn_base_param().loss_per_error();
	 string source = this->layer_param_.rnn_base_param().source();
	 LOG(INFO) <<"Opening File "<< source;
	 std::ifstream infile(source.c_str());
	 string filename;
	 while(infile >> filename){
		  lines_.push_back(filename);
	 }

	 this->blobs_.resize(3);
	 // weight
	 this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, 2*channels_));
	 // bias
	 this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, channels_));
	 //w_score
	 this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, channels_));
		 
	 //fill blob
	 shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>( 
				        this->layer_param_.rnn_base_param().weight_filler()));
	 // init for weight, w_score
	 for(int i = 0; i < 3; ++i){
		if(i == 1) continue;
		weight_filler->Fill(this->blobs_[i].get());	
	 }
	 	
	 // init for bias
	 shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>( this->layer_param_.rnn_base_param().bias_filler())); 
	 bias_filler->Fill(this->blobs_[1].get());

	 //relu_layer	 
	 LayerParameter relu_param;
	 relu_layer_.reset(new ReLULayer<Dtype>(relu_param));

	 LayerParameter sigmoid_param;
	 sigmoid_layer_.reset(new SigmoidLayer<Dtype>(sigmoid_param));

	 //split_layer_.reset(new SplitLayer<Dtype>(split_param));


		 
	 this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void RNNBaseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
	 const vector<Blob<Dtype>*>& top){

	int out_num = 2 * num_ - batch_size;

	top[0]->Reshape(out_num, channels_, 1, 1);
	top[1]->Reshape(out_num, 1, 1, 1);
	top[2]->Reshape(1, 1, 1, 1);
	top[3]->Reshape(4, 1, 1, 1);

	//set up the bias multiplier
	vector<int> bias_shape(1, 1);
	bias_multiplier_.Reshape(bias_shape);
    caffe_set(1, Dtype(1), bias_multiplier_.mutable_cpu_data());                                              
}

template<typename Dtype>
void RNNBaseLayer<Dtype>::Merge_cpu(Node<Dtype>* left, Node<Dtype>* right, Node<Dtype>* top, 
		Dtype* layer_top, Dtype* layer_label, bool correct_tree){

	 //M_:样本的个数  K_:样本的特征维数  N_: 输出的特征维数
	 int left_id = left->is_leaf ? 0: 1;
	 int right_id = right->is_leaf ? 0: 1;

	 const Dtype* left_feature = left->bottom[left_id]->cpu_data(), 
		   *right_feature = right->bottom[right_id]->cpu_data();

     const Dtype *weight = this->blobs_[0]->cpu_data(), *bias = this->blobs_[1]->cpu_data();  
	 const Dtype *w_score = this->blobs_[1]->cpu_data();
	 M_ = 1;
	 N_ = channels_;
	 K_ = 2 * N_;
		 
	 Dtype *joint_feature = new Dtype[K_];
	 caffe_copy(N_, left_feature, joint_feature);
	 caffe_copy(N_, right_feature, joint_feature + N_);

	 //cal 合并之后的bbox[c1, r1, c2, r2]
	 for(int i = 0; i < 2; i++)
	     top->rect.push_back(std::min(left->rect[i], right->rect[i]));
	 for(int i = 2; i < 4; i++)
	     top->rect.push_back(std::max(left->rect[i], right->rect[i]));


	 //cal label	 
     if(correct_tree){
	   top->map_id = map_id_;
	   layer_label[map_id_] = label(gt, top->rect);
	 }

	 //allocate space for the feature
	 for(int i = 0; i < 2; i++){
		top->bottom.push_back(new Blob<Dtype>(1, N_, 1, 1));
	 }

	 //M, N, K. A, B, C;
	 //A = M * K, B = K * N, C = M * N
	 //weight: N * K
	 caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
		   joint_feature, weight, (Dtype)0., top->bottom[0]->mutable_cpu_data());

	 caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
		   bias_multiplier_.cpu_data(), bias, (Dtype)1., top->bottom[0]->mutable_cpu_data());
	
 	 
	 delete joint_feature;
	 // relu
	 relu_bottom_vec_.clear(); 
	 relu_top_vec_.clear();
	 relu_bottom_vec_.push_back(top->bottom[0]);
	 relu_top_vec_.push_back(top->bottom[1]);
	 relu_layer_->Forward(relu_bottom_vec_, relu_top_vec_);


	 //copy to the top data
	 if(correct_tree) {
       layer_top += map_id_ * channels_;
	   caffe_copy(channels_, top->bottom[1]->mutable_cpu_data(), layer_top);
	   map_id_ ++;
	 }
	 //compute merge score
	 //w_score: 1 * N

	 M_ = 1;
	 K_ = channels_;
	 N_ = 1;

	 caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
		   top->bottom[1]->cpu_data(), w_score, (Dtype)0., &top->merge_score);
	 
	 //链接上左右孩子
	 top->left = left; 
	 top->right = right;

	 
}
template<typename Dtype>
void RNNBaseLayer<Dtype>::Backward_cpu(Node<Dtype> *left, Node<Dtype> *right, Node<Dtype> *top, const Dtype* layer_top, bool correct_tree){


    if(top->is_leaf && !correct_tree)
	   return;


	const Dtype* weight = this->blobs_[0]->cpu_data();
	const Dtype* w_score = this->blobs_[2]->cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
	Dtype* w_score_diff = this->blobs_[2]->mutable_cpu_diff();


	//softmax对feature求导数
	if(correct_tree){
	   layer_top += top->map_id * channels_;
	   //__asm__("int $3");
	   if(top->is_leaf) {
	      caffe_add(channels_, layer_top, top->bottom[0]->cpu_diff(), top->bottom[0]->mutable_cpu_diff());
		  return;
	   }
	   else
	      caffe_add(channels_, layer_top, top->bottom[1]->cpu_diff(), top->bottom[1]->mutable_cpu_diff());
	}

	
	// merge_score对feature求导数
	M_ = 1;
	K_ = channels_;
	N_ = 1;

	//loss = (largest_tree_score - correct_tree_score) / batch_size
	Dtype diff = -1.0/batch_size;
	if(!correct_tree) {
	   //__asm__("int $3"); 
	   diff *= -1;
	}
	
	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			        &diff, top->bottom[1]->cpu_data(), (Dtype)1., w_score_diff);

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				    &diff, w_score, (Dtype)1., top->bottom[1]->mutable_cpu_diff());
	

	//relu 
	relu_bottom_vec_.clear();
	relu_top_vec_.clear();
	relu_bottom_vec_.push_back(top->bottom[0]);
	relu_top_vec_.push_back(top->bottom[1]);

	vector<bool> propaget_down(1, true);

	relu_layer_->Backward(relu_top_vec_, propaget_down, relu_bottom_vec_);

    const Dtype *left_feature = left->bottom[0]->cpu_data(), *right_feature = right->bottom[0]->cpu_data();


	Dtype *joint_feature = new Dtype[2*channels_];
	caffe_copy(channels_, left_feature, joint_feature);
	caffe_copy(channels_, right_feature, joint_feature + channels_);
	M_ = 1;
	N_ = channels_;
	K_ = 2 * N_;
	

	//求出对weight以及bias的偏导
	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			top->bottom[0]->cpu_diff(), joint_feature, (Dtype)1., weight_diff);

	caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top->bottom[0]->cpu_diff(),
			bias_multiplier_.cpu_data(), (Dtype)1., bias_diff);
	
	//求出对delta的偏导
	Dtype* delta = new Dtype[K_];
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			top->bottom[0]->cpu_diff(),  weight, (Dtype)0., delta);
	

	//回传给left和right
	//注意是否是叶子节点
	if(left->is_leaf)
	   caffe_add(N_, delta, left->bottom[0]->cpu_diff(), left->bottom[0]->mutable_cpu_diff());
	else
	   caffe_add(N_, delta, left->bottom[1]->cpu_diff(), left->bottom[1]->mutable_cpu_diff());

	if(right->is_leaf)
      caffe_add(N_, delta + N_, right->bottom[0]->cpu_diff(), right->bottom[0]->mutable_cpu_diff());
    else
      caffe_add(N_, delta + N_, right->bottom[1]->cpu_diff(), right->bottom[1]->mutable_cpu_diff());

	delete[] joint_feature;
	delete[] delta;

}

template<typename Dtype>
Dtype RNNBaseLayer<Dtype>::get_score_cpu(const Dtype* left, const Dtype* right){
	  Dtype merge_score = 0.0;
	  const Dtype* weight = this->blobs_[0]->cpu_data();
	  const Dtype* bias = this->blobs_[1]->cpu_data();
	  const Dtype* w_score = this->blobs_[2]->cpu_data();

	  M_ = 1;
	  N_ = channels_;
	  K_ = 2 * N_;
		 
	  Dtype *joint_feature = new Dtype[K_];
	  caffe_copy(N_, left, joint_feature);
	  caffe_copy(N_, right, joint_feature + N_);
	  
      //M, N, K. A, B, C;
	  //A = M * K, B = K * N, C = M * N
	  //weight: N * K
	  Dtype* out_feature = new Dtype[N_];
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
		    joint_feature, weight, (Dtype)0., out_feature);

	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
		    bias_multiplier_.cpu_data(), bias, (Dtype)1., out_feature);
		
	  delete[] joint_feature;

	  for(int i = 0; i < N_; i++) out_feature[i] = MAX((Dtype)0., out_feature[i]);

	  M_ = 1;
	  K_ = channels_;
	  N_ = 1;

	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
		   out_feature, w_score, (Dtype)0., &merge_score);
	  delete[] out_feature;
	  return merge_score;
}

template<typename Dtype>
int RNNBaseLayer<Dtype>::Is_oneclass(int i, int j, const vector<int> vec){
	if(vec[i] == -1 || vec[j] == -1)
	  return 0;
	if(vec[i] == vec[j]) 
	  return 1;
	return 0;
}



template<typename Dtype>
vector<int> RNNBaseLayer<Dtype>::get_recall(const vector<Node<Dtype>*>& tree, bool test) {
	int gt_num = gt.size() / 4;
	bool* hit_1 = new bool[gt_num];
	bool* hit_2 = new bool[gt_num];

	vector<int> hit;
	if(!test){
	CHECK_EQ(tree.size(), total_num + seg_size - 1);
	//string path = "/media/WORK/caffe_rnn/caffe/data/SLIC_result_250/";

	memset(hit_1, false, sizeof(bool) * gt_num);
	memset(hit_2, false, sizeof(bool) * gt_num);

	for(int k = 0; k < gt_num; ++k){
        int x = gt[4*k], y = gt[4*k+1];
	    int w = gt[4*k+2], h = gt[4*k+3];
		Dtype iou = 0.0;
		//int id = -1;
		for(int i = 0; i < tree.size(); i++){
		   if(i >= seg_size && i < total_num) continue;
		   vector<int> rect = tree[i]->rect;

		   int area = (rect[2] - rect[0]) * (rect[3] - rect[1]) + w * h;
		
	       //cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0));
		   int tmp_w = std::max(0, std::min(x + w, rect[2]) - std::max(x, rect[0]));
		   int tmp_h = std::max(0, std::min(y + h, rect[3]) - std::max(y, rect[1]));

		   Dtype cur_iou = (Dtype)1. * tmp_w * tmp_h / (area - tmp_h * tmp_w);
		   iou = MAX(iou, cur_iou);
		} 
		if(iou > 0.5)  hit_1[k] = true;
		if(iou > 0.8)  hit_2[k] = true;
	}
	int cnt_1 = 0, cnt_2 = 0;
	for(int i = 0; i < gt_num; i++){

		if(hit_1[i]) cnt_1++;
		if(hit_2[i]) cnt_2++;
	}
	//LOG(INFO)<<cnt_1<<" "<<cnt_2;

	hit.push_back(cnt_1);
	//__asm__("int $3");
	hit.push_back(cnt_2);
}
	memset(hit_1, false, sizeof(bool) * gt_num);
	memset(hit_2, false, sizeof(bool) * gt_num);

	for(int k = 0; k < gt_num; ++k){
        int x = gt[4*k], y = gt[4*k+1];
	    int w = gt[4*k+2], h = gt[4*k+3];
		Dtype iou = 0.0;
		//int id = -1;
		for(int i = 0; i < total_num; i++){
		   //if(i >= seg_size && i < total_num) continue;
		   vector<int> rect = tree[i]->rect;

		   int area = (rect[2] - rect[0]) * (rect[3] - rect[1]) + w * h;
		
	       //cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0));
		   int tmp_w = std::max(0, std::min(x + w, rect[2]) - std::max(x, rect[0]));
		   int tmp_h = std::max(0, std::min(y + h, rect[3]) - std::max(y, rect[1]));

		   Dtype cur_iou = (Dtype)1. * tmp_w * tmp_h / (area - tmp_h * tmp_w);
		   iou = MAX(iou, cur_iou);
		} 
		if(iou > 0.5)  hit_1[k] = true;
		if(iou > 0.8)  hit_2[k] = true;
	}
	int cnt_1 = 0, cnt_2 = 0;
	for(int i = 0; i < gt_num; i++){
		if(hit_1[i]) cnt_1++;
		if(hit_2[i]) cnt_2++;
	}
	hit.push_back(cnt_1);
	hit.push_back(cnt_2);

	delete[] hit_1;
	delete[] hit_2;
	return hit;

}




template<typename Dtype>
int RNNBaseLayer<Dtype>::label(const vector<int>& gt, const vector<int>& rect){

	//LOG(INFO)<<"iou is "<<iou;
	
	int gt_num = gt.size() / 4;
	Dtype iou = 0.0;
	for(int k = 0; k < gt_num; ++k){
		//cal iou
		int x = gt[4*k], y = gt[4*k+1];
		int w = gt[4*k+2], h = gt[4*k+3];
		int area = (rect[2] - rect[0]) * (rect[3] - rect[1]) + w * h;
		
		int tmp_w = std::max(0, std::min(x + w, rect[2]) - std::max(x, rect[0]));
		int tmp_h = std::max(0, std::min(y + h, rect[3]) - std::max(y, rect[1]));

		Dtype cur_iou = (Dtype)1. * tmp_w * tmp_h / (area - tmp_h * tmp_w);
		iou = MAX(iou, cur_iou);
		//if(rect.size() != 4 || iou > 1)
		//  LOG(INFO)<<"size is "<<rect.size() <<" iou is "<<iou;

	}
	//__asm__("int $3");
	if(iou > max_iou) return 2;
	else if(iou < min_iou) return 1;
	else return 0;
}


template<typename Dtype>
void RNNBaseLayer<Dtype>::build_tree_cpu(vector<Node<Dtype>*> &tree, vector<int> seg_class, vector<bool> adj, 
	 vector<Pair<Dtype> > pair, Dtype* layer_top, Dtype* layer_label, bool correct_tree){
	 //used for swap
	 vector<Pair<Dtype> > pair_1;
	 pair_1.clear();

	 for(int i = 0; i < pair.size(); i++) {
		int left_id = pair[i].l, right_id = pair[i].r;
		if(!correct_tree)
		  pair[i].score += loss_per_error_ * Is_oneclass(left_id, right_id, seg_class);
	 }

	 for(int k = seg_size; k < total_num; ++k) {

		 sort(pair.begin(), pair.end());
		 int max_id = 0;

		 if(correct_tree){
			for(int i = 0; i < pair.size(); i++){
			    if(Is_oneclass(pair[i].l, pair[i].r, seg_class)){
				   max_id = i;
				   break;
				}
			}
	 	 }

	 	 //find the most high pair
		 int left_id = pair[max_id].l, right_id = pair[max_id].r;
		 tree.push_back(new Node<Dtype>());
		 //merge the node
		 Merge_cpu(tree[left_id], tree[right_id], tree[tree.size()-1], layer_top, layer_label, correct_tree);
		 //update current class 
		 if(Is_oneclass(left_id, right_id, seg_class)) 
			seg_class.push_back(seg_class[left_id]);
		 else
			seg_class.push_back(-1);	 
		 //remove nonexist pair
		 for(int i = 0; i < pair.size(); i++){
			if(pair[i].l == left_id || pair[i].l == right_id 
					|| pair[i].r == left_id || pair[i].r == right_id)
			   continue;
			pair_1.push_back(pair[i]);
		 }
		 //update adj matrix
		 for(int i = 0; i < total_num; i++){
			 if(adj[go(left_id, i)] || adj[go(right_id, i)])
				adj[go(k,i)] = adj[go(i,k)] = true;
		 }

		 for(int i = 0; i < total_num; i++){
			 adj[go(left_id,i)] = adj[go(i,left_id)] = false;
			 adj[go(right_id,i)] = adj[go(i,right_id)] = false;
		 }
		 //update merge score
		 for(int i = 0; i < total_num; i++){
			 if(adj[go(k,i)]){
			    int l_id = tree[tree.size()-1]->is_leaf ? 0: 1;
				int r_id = tree[i]->is_leaf ? 0: 1;
				 
				const Dtype* feat1 = tree[tree.size()-1]->bottom[l_id]->cpu_data();
				const Dtype* feat2 = tree[i]->bottom[r_id]->cpu_data();
				Dtype score_1 = get_score_cpu(feat1, feat2);
				Dtype score_2 = get_score_cpu(feat2, feat1);
				if(!correct_tree){
				  score_1 += loss_per_error_ * Is_oneclass(k, i, seg_class);
				  score_2 += loss_per_error_ * Is_oneclass(i, k, seg_class);
				}
				pair_1.push_back(Pair<Dtype>(k, i, score_1));
				pair_1.push_back(Pair<Dtype>(i, k, score_2));
			}
		}

		pair.clear();
		pair = pair_1;
		pair_1.clear();
	}	
}

template<typename Dtype>
void RNNBaseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) { 
    //bottom[0]->[roi_num, dim, 1, 1]---->feature
	//bottom[1]->[batch_size,  1, 1]----->id for each image
	//bottom[2]->[roi_num, 5, 1, 1]------>roi_num coordinate
	
	//合并得到的blob
	//top[0]->[roi, dim, 1, 1]
	//top[1]->gt

	//LOG(INFO)<<"Enter into the RNN layer............";
	int roi_num = bottom[0]->num();
	int dim = bottom[0]->channels();

	int out_num = 2 * roi_num - batch_size;
	//LOG(INFO)<<"batch_size is "<<batch_size;

	top[0]->Reshape(out_num, dim, 1, 1);

	top[1]->Reshape(out_num, 1, 1, 1);
	//__asm__("int $3");


	const Dtype* feature = bottom[0]->cpu_data();
	const Dtype* id = bottom[1]->cpu_data();
	const Dtype* roi = bottom[2]->cpu_data();

	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* top_label = top[1]->mutable_cpu_data();

	//id设置为0
	map_id_ = 0;

	//初始化权重和偏置的diff
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
	Dtype* w_score_diff = this->blobs_[2]->mutable_cpu_diff();

	caffe_set(this->blobs_[0]->count(),  (Dtype)0., weight_diff);
	caffe_set(this->blobs_[1]->count(),  (Dtype)0., bias_diff);
	caffe_set(this->blobs_[2]->count(),  (Dtype)0., w_score_diff);

	for(int batch_id = 0; batch_id < batch_size; batch_id ++){
		//get segmentation size
		int index = id[batch_id];
		//read gt
	    gt = ReadFileToVector(gt_folder + lines_[index] + ".txt");
		//read adj
	    vec_adj = ReadFileToVector(adj_folder + lines_[index] + ".txt");
		//read class
		seg_class = ReadFileToVector(class_folder + lines_[index] + ".txt");
		seg_size = seg_class.size();

		//__asm__("int $3");


		vector<Node<Dtype>*> tree;
		tree.clear();
		CPUTimer timer;
		timer.Start();
		total_num = 2 * seg_size - 1;

		for(int i = 0; i < seg_size; i++){
			tree.push_back(new Node<Dtype>());
		    tree[i]->bottom.push_back(new Blob<Dtype>(1, dim, 1, 1));
			//LOG(INFO)<<"map_id_ should be -1"<<tree[i]->map_id;
			//LOG(INFO)<<"is_leaf should be false"<<tree[i]->is_leaf;
			tree[i]->map_id = map_id_;
			tree[i]->is_leaf = true;
			Dtype* src = tree[i]->bottom[0]->mutable_cpu_data();			
			caffe_copy(dim, feature + i * dim, src);
			caffe_copy(dim, feature + i * dim, top_data + map_id_ * channels_);

			  
			for(int j = 1; j <= 4; ++j) tree[i]->rect.push_back(roi[j]);

			top_label[map_id_] = label(gt, tree[i]->rect);
			roi += bottom[2]->offset(1);
			map_id_ ++;
		}

		vector<bool> adj(total_num * total_num, false);
		// read adj matrix
		for(int i = 0; i < seg_size; i++){
			for(int j = 0; j < seg_size; j++){
				int offset = i * seg_size + j;
				if(vec_adj[offset] == 1) adj[go(i,j)] = true;
			}
		}
        vector<Pair<Dtype> > pair;
		pair.clear();
	    for(int i = 0; i < seg_size; ++i){
		   for(int j = 0; j < seg_size; ++j){
			  const Dtype* feat1 = feature + i * dim;
			  const Dtype* feat2 = feature + j * dim;
			  if(adj[go(i,j)]){
				CPUTimer time;
				time.Start();
				Dtype score = get_score_cpu(feat1,feat2);
				time.Stop();
				LOG(INFO)<<"get score "<<time.MilliSeconds()<<"ms.";
			    pair.push_back(Pair<Dtype>(i, j, score));
			  }
		   }
	    }
		//timer.Stop();
		//LOG(INFO)<<"Initial cal pairs "<<timer.MilliSeconds() << "ms.";
	    //LOG(INFO)<<"modify the tree...............";	
		//结构正确的得分最高的树

		//CPUTimer tree_timer;
		//tree_timer.Start();
		build_tree_cpu(tree, seg_class, adj, pair, top_data, top_label, true);
		//得分最高的树
		build_tree_cpu(tree, seg_class, adj, pair, top_data, top_label, false);
		timer.Stop();
		LOG(INFO)<<"RNN forward cost "<<timer.MilliSeconds() <<"ms.";
		//tree_timer.Stop();
		//LOG(INFO)<<"build tree cost "<<tree_timer.MilliSeconds() << "ms.";
		//merge all score
	    forest.push_back(tree);
	    feature += seg_size * dim;
	}

	//LOG(INFO)<<"Leave the RNN Layer..............";
	//LOG(INFO)<<map_id_<<" should be equal to "<<out_num;

}

template<typename Dtype>
void RNNBaseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	
	//int roi_num = bottom[0]->num();
	int dim = bottom[0]->channels();
    //LOG(INFO)<<"dim is ..."<<dim;
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* top_diff = top[0]->cpu_diff();
	CPUTimer time;
	time.Start();
	for(int i = 0; i < batch_size; i++){
		vector<Node<Dtype>*> tree = forest[i];
		seg_size = (tree.size() + 2) / 3;
		total_num = 2 * seg_size - 1;
		//LOG(INFO)<<"seg_size is "<<seg_size;
		//LOG(INFO)<<"total_num is "<<total_num;

		//first correct tree
		std::stack<Node<Dtype>*> recursive;

		recursive.push(tree[total_num - 1]);

		while(!recursive.empty()){
			Node<Dtype>* top_node = recursive.top();
			Backward_cpu(top_node->left, top_node->right, top_node, top_diff, true);
			//LOG(INFO)<<"enter count ";
			recursive.pop();
			if(top_node->right != NULL) recursive.push(top_node->right);
			if(top_node->left != NULL) recursive.push(top_node->left);
		}
		//LOG(INFO)<<".......correct_tree finished...";

		//high score tree
		recursive.push(tree[total_num + seg_size - 2]);

		while(!recursive.empty()){
			Node<Dtype>* top_node = recursive.top();
			//LOG(INFO)<<"enter ....";
			Backward_cpu(top_node->left, top_node->right, top_node, top_diff, false);
			recursive.pop();
			if(top_node->right != NULL) recursive.push(top_node->right);
			if(top_node->left != NULL) recursive.push(top_node->left);
		}
		//backward 
		//LOG(INFO)<<".......high tree finished...";

		for(int j = 0; j < seg_size; j++){
		   caffe_copy(dim, tree[j]->bottom[0]->cpu_diff(), bottom_diff);
		   bottom_diff += dim;
		}

	}
	time.Stop();

	LOG(INFO)<<"RNN backward cost "<<time.MilliSeconds() <<"ms.";


	//LOG(INFO)<<"delete all space ....";
	//delete all space
	for(int i = 0; i < batch_size; i++){
		vector<Node<Dtype>*> tree = forest[i];
		for(int j = 0; j < tree.size(); j++){
			delete tree[j]->bottom[0];
			if(!tree[j]->is_leaf) 
			  delete tree[j]->bottom[1];
			delete tree[j];
		}
	}
	forest.clear();	
}

#ifdef CPU_ONLY
STUB_GPU(RNNBaseLayer);
#endif

INSTANTIATE_CLASS(RNNBaseLayer);
REGISTER_LAYER_CLASS(RNNBase);
//REGISTER_LAYER_CLASS(RNN);


}
