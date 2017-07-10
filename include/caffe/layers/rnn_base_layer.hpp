#ifndef CAFFE_RNN_BASE_LAYER_HPP_
#define CAFFE_RNN_BASE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
//#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the rnn_base function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
struct Node{
	vector<Blob<Dtype>*> bottom;
	vector<int> rect;
	Dtype merge_score, loss_score;
	bool is_leaf;

	int map_id; //对应的节点编号
	Node<Dtype>* left, *right;
	Node(){
		left = right = NULL;
		map_id = -1;
		rect.clear();
		merge_score = loss_score = 0;
		is_leaf = false;
	}
};
template <typename Dtype>
struct Pair{
	int l, r;
	Dtype score;
	bool operator<(const Pair& A) const {
		return score > A.score;
	}
	Pair(int l_, int r_, Dtype score_){
		 l = l_; r = r_; score = score_;
	}
};


template <typename Dtype>
class RNNBaseLayer : public Layer<Dtype> {
 public:
  explicit RNNBaseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RNNBase"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  virtual inline int ExactNumTopBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  void Merge_cpu(Node<Dtype>* left, Node<Dtype>* right, Node<Dtype>* top, Dtype *layer_top, 
	   Dtype *layer_label, bool correct_tree);

  void Merge_gpu(Node<Dtype>* left, Node<Dtype>* right, Node<Dtype>* top, Dtype *layer_top, 
	   Dtype *layer_label, bool correct_tree, bool test);

  void Backward_cpu(Node<Dtype>* left, Node<Dtype>* right, Node<Dtype>* top, const Dtype *layer_top, bool correct_tree);

  void Backward_gpu(Node<Dtype>* left, Node<Dtype>* right, Node<Dtype>* top, const Dtype *layer_top, bool correct_tree);

  void build_tree_cpu(vector<Node<Dtype>*> &tree, vector<int> seg_class, vector<bool> adj, 
	   vector<Pair<Dtype> > pair, Dtype* feature, Dtype* layer_label, bool correct_tree);

  void build_tree_gpu(vector<Node<Dtype>*> &tree, vector<int> seg_class, vector<bool> adj, 
	   vector<Pair<Dtype> > pair, Dtype* feature, Dtype* layer_label, bool correct_tree, bool test);


  

  int Is_oneclass(int i, int j, const vector<int> seg_class);

  int debug(Node<Dtype>* root);




  inline int go(int i, int j){return i * total_num + j;}
  int label(const vector<int>& gt, const vector<int>& rect);
  //Dtype get_iou(const vector<int>& gt, const vector<int>& rect);
  Dtype get_score_cpu(const Dtype* left, const Dtype *right);
  
  vector<int> get_recall(const vector<Node<Dtype>*>& tree, bool test);

  void get_score_gpu(const Blob<Dtype>* input, Blob<Dtype>* merge_score);  
		  
  vector<vector<Node<Dtype>*> > forest;

				  
  				  
  shared_ptr<Layer<Dtype> > relu_layer_;
  vector<Blob<Dtype>*> relu_bottom_vec_;
  vector<Blob<Dtype>*> relu_top_vec_;
  shared_ptr<Layer<Dtype> > sigmoid_layer_;
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  vector<Blob<Dtype>*> sigmoid_top_vec_;
  int num_, channels_;
  int M_;
  int K_;
  int N_;
  int map_id_;//对应的节点编号
  int seg_size; //seg_size
  int total_num; //total_num
  int batch_size; //batch_size
  Dtype min_iou, max_iou; // min_iou, max_iou
  Dtype loss_per_error_;
  Dtype loss;
  vector<string> lines_;
  vector<int> gt;
  vector<int> vec_adj;
  vector<int> seg_class;
  string gt_folder, adj_folder, class_folder;
  Blob<Dtype> bias_multiplier_;
  vector<Dtype> tree_loss;
  int selected_num;
};

}  // namespace caffe

#endif  // CAFFE_RNN_BASE_LAYE
