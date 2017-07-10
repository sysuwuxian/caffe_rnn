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
namespace caffe {


template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}



template<typename Dtype>
void RNNBaseLayer<Dtype>::Merge_gpu(Node<Dtype>* left, Node<Dtype>* right, Node<Dtype>* top, 
		Dtype* layer_top, Dtype* layer_label, bool correct_tree, bool test){

	 //CPUTimer time;
	 //time.Start();

	 //M_:样本的个数  K_:样本的特征维数  N_: 输出的特征维数
	 int left_id = left->is_leaf ? 0: 1;
	 int right_id = right->is_leaf ? 0: 1;

	 const Dtype* left_feature = left->bottom[left_id]->gpu_data(); 
	 const Dtype* right_feature = right->bottom[right_id]->gpu_data();

   const Dtype* weight = this->blobs_[0]->gpu_data();
	 const Dtype* bias = this->blobs_[1]->gpu_data();  
	 const Dtype* w_score = this->blobs_[2]->gpu_data();
	 M_ = 1;
	 N_ = channels_;
	 K_ = 2 * N_;
	

   Blob<Dtype>* joint = new Blob<Dtype>(1, K_, 1, 1);

	 Dtype *joint_feature = joint->mutable_gpu_data();

	 caffe_copy(N_, left_feature, joint_feature);
	 caffe_copy(N_, right_feature, joint_feature + N_);

	 //cal 合并之后的bbox[c1, r1, c2, r2]
	 for(int i = 0; i < 2; i++)
	     top->rect.push_back(std::min(left->rect[i], right->rect[i]));
	 for(int i = 2; i < 4; i++)
	     top->rect.push_back(std::max(left->rect[i], right->rect[i]));


	 //cal label	 
     if(correct_tree && !test){
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
	 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
		   joint_feature, weight, (Dtype)0., top->bottom[0]->mutable_gpu_data());

	 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
		   bias_multiplier_.gpu_data(), bias, (Dtype)1., top->bottom[0]->mutable_gpu_data());
	
 	 
	 delete joint;
	 // relu
	 relu_bottom_vec_.clear(); 
	 relu_top_vec_.clear();
	 relu_bottom_vec_.push_back(top->bottom[0]);
	 relu_top_vec_.push_back(top->bottom[1]);
	 relu_layer_->Forward(relu_bottom_vec_, relu_top_vec_);



	 //sigmoid

//	 sigmoid_bottom_vec_.clear(); 
//	 sigmoid_top_vec_.clear();
//	 sigmoid_bottom_vec_.push_back(top->bottom[0]);
//	 sigmoid_top_vec_.push_back(top->bottom[1]);
//	 sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

	 //copy to the top data
	 if(correct_tree && !test) {
       layer_top += map_id_ * channels_;
	   caffe_copy(channels_, top->bottom[1]->mutable_gpu_data(), layer_top);
	   map_id_ ++;
	 }
	 //compute merge score
	 //w_score: 1 * N

	 M_ = 1;
	 K_ = channels_;
	 N_ = 1;
	 Blob<Dtype> score;
	 score.Reshape(1,1,1,1);

	 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
		   top->bottom[1]->gpu_data(), w_score, (Dtype)0., score.mutable_gpu_data());

	 //sigmoid layer
	 

	 top->merge_score = score.cpu_data()[0];	 
	 //链接上左右孩子
	 top->left = left; 
	 top->right = right;

}
template<typename Dtype>
void RNNBaseLayer<Dtype>::Backward_gpu(Node<Dtype> *left, Node<Dtype> *right, Node<Dtype> *top, const Dtype* layer_top, bool correct_tree){


	//CPUTimer time;
	//time.Start();
    if(top->is_leaf && !correct_tree)
	   return;

	const Dtype* weight = this->blobs_[0]->gpu_data();
	const Dtype* w_score = this->blobs_[2]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
	Dtype* w_score_diff = this->blobs_[2]->mutable_gpu_diff();


	//softmax对feature求导数
	if(correct_tree){
	   //layer_top += top->map_id * channels_;
	   //__asm__("int $3");
	   if(top->is_leaf) {
	      //caffe_gpu_add(channels_, layer_top, top->bottom[0]->gpu_diff(), top->bottom[0]->mutable_gpu_diff());
		  return;
	   }
	   /*
	   else
	      caffe_gpu_add(channels_, layer_top, top->bottom[1]->gpu_diff(), top->bottom[1]->mutable_gpu_diff());
	   */
	}

	
	// merge_score对feature求导数
	M_ = 1;
	K_ = channels_;
	N_ = 1;

	//loss = (largest_tree_score - correct_tree_score) / batch_size

	//Dtype score = top->merge_score;
	Dtype value = -1.0/batch_size;

	//Dtype tmp = 1.0;

	/*tmp = tmp * exp(tree_loss[selected_num]);

	tmp = tmp /(1 + tmp);*/

	//tmp = 100 * tmp / (seg_size - 1);

/*	value = value * tmp;

	if(rand() % 20000 == 0)
		LOG(INFO) << tmp;*/

	if(!correct_tree) {
	   //__asm__("int $3"); 
	   value *= -1;
	}

	//value = value + 0.01 * score / batch_size;

	Blob<Dtype> Diff;
	Diff.Reshape(1, 1, 1, 1);


	Dtype* diff = Diff.mutable_cpu_data();
	diff[0] = value;


	//may be accelerated by other function
	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			        Diff.gpu_data(), top->bottom[1]->gpu_data(), (Dtype)1., w_score_diff);

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				    Diff.gpu_data(), w_score, (Dtype)1., top->bottom[1]->mutable_gpu_diff());
	

	//relu 
	relu_bottom_vec_.clear();
	relu_top_vec_.clear();
	relu_bottom_vec_.push_back(top->bottom[0]);
	relu_top_vec_.push_back(top->bottom[1]);
	vector<bool> propaget_down(1, true);
	relu_layer_->Backward(relu_top_vec_, propaget_down, relu_bottom_vec_);

//	sigmoid_bottom_vec_.clear();
//	sigmoid_top_vec_.clear();
//	sigmoid_bottom_vec_.push_back(top->bottom[0]);
//	sigmoid_top_vec_.push_back(top->bottom[1]);
//	vector<bool> propaget_down(1, true);

//	sigmoid_layer_->Backward(sigmoid_top_vec_, propaget_down, sigmoid_bottom_vec_);
	//now backward to two son node
	//careful for others
    int left_id = left->is_leaf ? 0: 1;
	int right_id = right->is_leaf ? 0: 1;


    const Dtype* left_feature = left->bottom[left_id]->gpu_data();
    const Dtype* right_feature = right->bottom[right_id]->gpu_data();


	Blob<Dtype>* joint = new Blob<Dtype>(1, 2*channels_, 1, 1);

	Dtype *joint_feature = joint->mutable_gpu_data();


	//Dtype *joint_feature = new Dtype[2*channels_];
	caffe_copy(channels_, left_feature, joint_feature);
	caffe_copy(channels_, right_feature, joint_feature + channels_);
	M_ = 1;
	N_ = channels_;
	K_ = 2 * N_;
    //LOG(INFO)<<"ENTER .................";	

	//求出对weight以及bias的偏导
	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			top->bottom[0]->gpu_diff(), joint_feature, (Dtype)1., weight_diff);

	caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top->bottom[0]->gpu_diff(),
			bias_multiplier_.gpu_data(), (Dtype)1., bias_diff);
	
	//求出对delta的偏导
	//Dtype* delta = new Dtype[K_];
	Dtype* delta = joint->mutable_gpu_diff();


	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			top->bottom[0]->gpu_diff(),  weight, (Dtype)0., delta);


    caffe_gpu_add(N_, delta, left->bottom[left_id]->gpu_diff(), left->bottom[left_id]->mutable_gpu_diff());

    caffe_gpu_add(N_, delta + N_, right->bottom[right_id]->gpu_diff(), right->bottom[right_id]->mutable_gpu_diff());

	delete joint;

}

template<typename Dtype>
void  RNNBaseLayer<Dtype>::get_score_gpu(const Blob<Dtype>* input, Blob<Dtype>* merge_score) {

	//input blobs 

      CPUTimer time;
	  time.Start();


	  const Dtype* weight = this->blobs_[0]->gpu_data();
	  //__asm__("int $3");
	  const Dtype* bias = this->blobs_[1]->gpu_data();
	  const Dtype* w_score = this->blobs_[2]->gpu_data();

	  M_ = input->num();
	  N_ = channels_;
	  K_ = 2 * N_;


	  Blob<Dtype>* out = new Blob<Dtype>(M_, N_, 1, 1);



	  const Dtype* joint_feature = input->gpu_data();
	  Dtype* out_feature = out->mutable_gpu_data();

      //M, N, K. A, B, C;
	  //A = M * K, B = K * N, C = M * N
	  //weight: N * K

	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
		    joint_feature, weight, (Dtype)0., out_feature);

	  //notice the matrix dimension
	  Blob<Dtype> matrix_bias_multiplier_;
	  matrix_bias_multiplier_.Reshape(1, 1, 1, M_);
	  caffe_set(matrix_bias_multiplier_.count(), Dtype(1), matrix_bias_multiplier_.mutable_cpu_data()); 

	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
		    matrix_bias_multiplier_.gpu_data(), bias, (Dtype)1., out_feature);
	

      relu_bottom_vec_.clear(); 
	  relu_top_vec_.clear();
	  relu_bottom_vec_.push_back(out);
	  relu_top_vec_.push_back(out);
	  relu_layer_->Forward(relu_bottom_vec_, relu_top_vec_);

//    sigmoid_bottom_vec_.clear(); 
//	  sigmoid_top_vec_.clear();
//	  sigmoid_bottom_vec_.push_back(out);
//	  sigmoid_top_vec_.push_back(out);
//	  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
	  M_ = input->num();
	  K_ = channels_;
	  N_ = 1;

	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
		   out_feature, w_score, (Dtype)0., merge_score->mutable_gpu_data());
	  delete out;
	  time.Stop();

	  //LOG(INFO)<<"get score per cost "<<time.MilliSeconds()<<".ms";
	  return ;
}



template<typename Dtype>
void RNNBaseLayer<Dtype>::build_tree_gpu(vector<Node<Dtype>*> &tree, vector<int> seg_class, vector<bool> adj, 
	   vector<Pair<Dtype> > pair, Dtype* layer_top, Dtype* layer_label, bool correct_tree, bool test) {

     CPUTimer time;
	 vector<Pair<Dtype> > pair_1;
	 pair_1.clear();
     //time.Start();
     CHECK_EQ(seg_class.size(), seg_size);
     //__asm__("int $3");
	 //LOG(INFO)<<"loss per error is "<<loss_per_error_;
	 
	 for(int i = 0; i < pair.size(); i++) {
		int l_id = pair[i].l, r_id = pair[i].r;
		if(!correct_tree){
		   pair[i].score += loss_per_error_ * (1 - Is_oneclass(l_id, r_id, seg_class));
		   // LOG(INFO)<<i<<" "<<pair[i].score;
		}
	 }


	 for(int k = seg_size; k < total_num; ++k) {
        // time.Start();

		 sort(pair.begin(), pair.end());
		 int max_id = 0;


		 if(correct_tree && !test){
			for(int i = 0; i < pair.size(); i++){
			    if(Is_oneclass(pair[i].l, pair[i].r, seg_class)){
				   max_id = i;
				   break;
				}
			}
	 	 }

		 if(correct_tree) loss -= pair[max_id].score;
		 else loss += pair[max_id].score;
		 int left_id = pair[max_id].l, right_id = pair[max_id].r;
		 
		 
	 	 //find the most high pair
		 tree.push_back(new Node<Dtype>());
		 //merge the node
		 if(correct_tree)
		    Merge_gpu(tree[left_id], tree[right_id], tree[tree.size()-1], layer_top, layer_label, correct_tree, test);
		 else{
			 //To use the adj space and others 
			 //we obey the order first correct tree, second highest tree
			 //must noticed the l_id, r_id.
			 int l_id = left_id >= seg_size && left_id < total_num ? left_id + seg_size - 1 : left_id;
			 int r_id = right_id >= seg_size && right_id < total_num ? right_id + seg_size - 1 : right_id;
			 Merge_gpu(tree[l_id], tree[r_id], tree[tree.size()-1], layer_top, layer_label, correct_tree, test);
		 }

		 //update current class 
		 if(Is_oneclass(left_id, right_id, seg_class)) 
			seg_class.push_back(seg_class[left_id]);
		 else
			seg_class.push_back(-1);


		 /*if(rand() % 5000 == 0 && !test){

			if(seg_class[seg_class.size()-1] == -1 && !correct_tree)
		       LOG(INFO)<<"Wrong merge "<<pair[max_id].score<<" loss per error "<<loss_per_error_;
			else
			   LOG(INFO)<<"Correct merge "<<pair[max_id].score<<" loss per error "<<loss_per_error_;
		 }*/

		 //remove nonexist pair
		 for(int i = 0; i < pair.size(); i++){
			if(pair[i].l == left_id || pair[i].l == right_id 
					|| pair[i].r == left_id || pair[i].r == right_id)
			   continue;
			pair_1.push_back(pair[i]);
		 }
		 //update adj matrix
		 vector<int> tmp;
		 for(int i = 0; i < total_num; i++){
			 if(adj[go(left_id, i)] || adj[go(right_id, i)])
				tmp.push_back(i);
		 }

		 for(int i = 0; i < tmp.size(); i++){
		 	 adj[go(k,tmp[i])] = adj[go(tmp[i],k)] = true;
		 }

		 for(int i = 0; i < total_num; i++){
			 adj[go(left_id,i)] = adj[go(i,left_id)] = false;
			 adj[go(right_id,i)] = adj[go(i,right_id)] = false;
		 }

		 //update merge score
		time.Start();

		int cnt = 0;
		for(int i = 0; i < total_num ; i++){
			if(adj[go(k,i)])
			  cnt ++;
		}
		//for 2 direction
		cnt *= 2;

		if(cnt == 0) continue;

		Blob<Dtype> input, merge_score;
		//__asm__("int $3");
		input.Reshape(cnt, 2*channels_, 1, 1);
		merge_score.Reshape(cnt, 1, 1, 1);

		Dtype* input_feature = input.mutable_gpu_data();
		for(int i = 0; i < total_num; ++i) {
			 if(adj[go(k,i)]){

				int id = i;
				if(!correct_tree)
                  id = i  >= seg_size && i < total_num ? i + seg_size - 1 : i;

			    int l_id = tree[tree.size()-1]->is_leaf ? 0: 1;
				int r_id = tree[id]->is_leaf ? 0: 1;
				 
				const Dtype* feat1 = tree[tree.size()-1]->bottom[l_id]->gpu_data();
				const Dtype* feat2 = tree[id]->bottom[r_id]->gpu_data();

				caffe_copy(channels_, feat1, input_feature);
				caffe_copy(channels_, feat2, input_feature + channels_);

				input_feature += input.offset(1);

				caffe_copy(channels_, feat2, input_feature);
				caffe_copy(channels_, feat1, input_feature + channels_);

				input_feature += input.offset(1);
			 }

		}
		//__asm__("int $3");
		get_score_gpu(&input, &merge_score);
		const Dtype* score = merge_score.cpu_data();
		int id = 0;

		for(int i = 0; i < total_num; ++i) {

			if(adj[go(k,i)]){
			  	Dtype score_1 = score[id++], score_2 = score[id++];
				//time.Stop();
				//LOG(INFO)<<"get score cost "<<time.MilliSeconds()<<"ms.";
					
				if(!correct_tree){
				   score_1 += loss_per_error_ * (1 - Is_oneclass(k, i, seg_class));
				   score_2 += loss_per_error_ * (1 - Is_oneclass(i, k, seg_class));
				}
				pair_1.push_back(Pair<Dtype>(k, i, score_1));
				pair_1.push_back(Pair<Dtype>(i, k, score_2));
			
			}

		}

		time.Stop();
		//LOG(INFO)<<"update get_score is "<<time.MilliSeconds()<<"ms.";

		pair.clear();
		pair = pair_1;
		pair_1.clear();

	}
	//__asm__("int $3");
	/*	 
	if(!correct_tree) {
		for(int k = 0; k < seg_class.size(); ++k){
			LOG(INFO)<<seg_class[k];
		}
		__asm__("int $3");
	}*/
	
}

template<typename Dtype>
void RNNBaseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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


	const Dtype* feature = bottom[0]->gpu_data();
	const Dtype* id = bottom[1]->cpu_data();
	const Dtype* roi = bottom[2]->cpu_data();


	Dtype* top_data = top[0]->mutable_gpu_data();

	//note for cpu
	Dtype* top_label = top[1]->mutable_cpu_data();

	//id设置为0
	map_id_ = 0;
	//tree loss is set to 0
	loss = (Dtype)0.;
	int hit_1 = 0, hit_2 = 0, hit_3 = 0, hit_4 = 0, total_gt = 0;
    // __asm__("int $3");
	Dtype total_loss = 0;
	tree_loss.clear();
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
		//CPUTimer timer;
		//timer.Start();
		total_num = 2 * seg_size - 1;
		DLOG(INFO)<<"Forward seg_size is "<<seg_size;

		for(int i = 0; i < seg_size; i++){
			tree.push_back(new Node<Dtype>());
		    tree[i]->bottom.push_back(new Blob<Dtype>(1, dim, 1, 1));
			//LOG(INFO)<<"map_id_ should be -1"<<tree[i]->map_id;
			//LOG(INFO)<<"is_leaf should be false"<<tree[i]->is_leaf;
			tree[i]->map_id = map_id_;
			tree[i]->is_leaf = true;
			Dtype* src = tree[i]->bottom[0]->mutable_gpu_data();			
			caffe_copy(dim, feature + i * dim, src);
			caffe_copy(dim, feature + i * dim, top_data + map_id_ * channels_);

			for(int j = 1; j <= 4; ++j) tree[i]->rect.push_back(roi[j]);

			top_label[map_id_] = label(gt, tree[i]->rect);


			roi += bottom[2]->offset(1);
			map_id_ ++;
		}

		vector<bool> adj(total_num * total_num, false);

		int cnt = 0;
		// read adj matrix
		for(int i = 0; i < seg_size; i++){
			for(int j = 0; j < seg_size; j++){
				int offset = i * seg_size + j;
				if(vec_adj[offset] == 1) {
				   adj[go(i,j)] = true;
				   cnt ++;
				}
			}
		}
        vector<Pair<Dtype> > pair;
		pair.clear();
		//__asm__("int $3");

		CPUTimer time;
		time.Start();

		Blob<Dtype>* input = new Blob<Dtype>(cnt, 2 * channels_, 1, 1);
		Blob<Dtype>* merge_score = new Blob<Dtype>(cnt, 1, 1, 1);

		Dtype* input_feature = input->mutable_gpu_data();
	    for(int i = 0; i < seg_size; ++i){
		   for(int j = 0; j < seg_size; ++j){
			  const Dtype* feat1 = feature + i * dim;
			  const Dtype* feat2 = feature + j * dim;
			  if(adj[go(i,j)]){
				caffe_copy(channels_, feat1, input_feature);
				caffe_copy(channels_, feat2, input_feature + channels_);
				//__asm__("int $3");
				input_feature += 2 * channels_;
			    pair.push_back(Pair<Dtype>(i, j, 0));
			  }
		   }
	    }

	    get_score_gpu(input, merge_score);
		//check for eq
		CHECK_EQ(pair.size(), merge_score->count());


		for(int i = 0; i < pair.size(); ++i) {
			pair[i].score = merge_score->cpu_data()[i];
		}

		//__asm__("int $3");

		delete input;
		delete merge_score;


		time.Stop();
		DLOG(INFO)<<"Initial cal pairs "<<time.MilliSeconds() << "ms.";
		//__asm__("int $3");
	    //LOG(INFO)<<"modify the tree...............";	
		//结构正确的得分最高的树

		CPUTimer timer;
		timer.Start();

		if(this->layer_param_.phase() == caffe::TEST){

		   build_tree_gpu(tree, seg_class, adj, pair, top_data, top_label, true, true);

		   feature += seg_size * dim;

		   vector<int> ret = get_recall(tree, true);
		   //__asm__("int $3");
		   hit_1 += ret[0]; 
		   hit_2 += ret[1];
		   total_gt += gt.size() / 4;

		   //destroy this tree
		   for(int j = 0; j < tree.size(); j++){
			   delete tree[j]->bottom[0];
			   if(!tree[j]->is_leaf) 
			      delete tree[j]->bottom[1];
			   delete tree[j];
		   }




		   continue;
		}
		loss = 0.0;
		//得分最高的树
		build_tree_gpu(tree, seg_class, adj, pair, top_data, top_label, true, false);
		//LOG(INFO)<<"BUILD CORRECT TREE FINISHED!";
		DLOG(INFO)<<debug(tree[total_num-1])<<" should equal to "<<total_num;
		//__asm__("int $3");
	    	build_tree_gpu(tree, seg_class, adj, pair, top_data, top_label, false, false);	
		DLOG(INFO)<<debug(tree[tree.size()-1])<<" should equal to "<<total_num;
		timer.Stop();
		DLOG(INFO)<<"RNN forward cost "<<timer.MilliSeconds() <<"ms.";

		tree_loss.push_back(loss / (seg_size - 1));
		//total_loss += log(1 + exp(loss / (seg_size - 1)));
		total_loss += loss;
		vector<int> ret = get_recall(tree, false);
		//__asm__("int $3");
		hit_1 += ret[0]; 
		hit_2 += ret[1];
		hit_3 += ret[2];
		hit_4 += ret[3];
		total_gt += gt.size() / 4;


		forest.push_back(tree);
	    feature += seg_size * dim;
	}

	if(this->layer_param_.phase() == TEST)
	   LOG(INFO)<<"Test "<<total_loss / batch_size << " " <<hit_1<<" "<<hit_2<<" "<<" "<<hit_3<<" "<<hit_4<<" "<<total_gt;
	else
	   LOG(INFO)<<"TRAIN "<<total_loss / batch_size <<" "<<hit_1<<" "<<hit_2<<" "<<" "<<hit_3<<" "<<hit_4<<" "<<total_gt;

		
	top[2]->mutable_cpu_data()[0] = total_loss / batch_size / 2;

	top[3]->mutable_cpu_data()[0] = (Dtype)1.0 * hit_1 / total_gt;
	top[3]->mutable_cpu_data()[1] = (Dtype)1.0 * hit_2 / total_gt;

	top[3]->mutable_cpu_data()[2] = (Dtype)1.0 * hit_3 / total_gt;
	top[3]->mutable_cpu_data()[3] = (Dtype)1.0 * hit_4 / total_gt;

}

template<typename Dtype>
int RNNBaseLayer<Dtype>::debug(Node<Dtype>* root){

	if(root->is_leaf) return 1;

	int num = 1;

	num += debug(root->left);
	num += debug(root->right);
	return num;

}

template<typename Dtype>
void RNNBaseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	
	//int roi_num = bottom[0]->num();
	int dim = bottom[0]->channels();
    //LOG(INFO)<<"dim is ..."<<dim;
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* top_diff = top[0]->gpu_diff();
	//CPUTimer time;
	//time.Start();
	
	for(int i = 0; i < batch_size; i++){
		selected_num = i;
		vector<Node<Dtype>*> tree = forest[i];
		seg_size = (tree.size() + 2) / 3;
		total_num = 2 * seg_size - 1;
		//first correct tree
		std::stack<Node<Dtype>*> recursive;
		//__asm__("int $3");
		recursive.push(tree[total_num - 1]);
		while(!recursive.empty()){
			Node<Dtype>* top_node = recursive.top();

			Backward_gpu(top_node->left, top_node->right, top_node, top_diff, true);
            recursive.pop();
			if(top_node->right != NULL)	recursive.push(top_node->right);
			
			if(top_node->left != NULL) recursive.push(top_node->left);
		}

		//LOG(INFO)<<"should enter count is "<<cnt<<" "<<total_num;
		

		//high score tree
		recursive.push(tree[tree.size() - 1]);

		while(!recursive.empty()){
			Node<Dtype>* top_node = recursive.top();
			//LOG(INFO)<<"enter ....";
			Backward_gpu(top_node->left, top_node->right, top_node, top_diff, false);
			recursive.pop();
			if(top_node->right != NULL) recursive.push(top_node->right);

			if(top_node->left != NULL) recursive.push(top_node->left);
		}
		//backward 
		//LOG(INFO)<<".......high tree finished...";


		for(int j = 0; j < seg_size; j++){
		   caffe_copy(dim, tree[j]->bottom[0]->gpu_diff(), bottom_diff);
		   bottom_diff += dim;
		}

	}

	const Dtype* test_weight = this->blobs_[0]->cpu_data();
	//__asm__("int $3");
	//time.Stop();

    //LOG(INFO)<<"RNN backward cost "<<time.MilliSeconds() <<"ms.";

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
	tree_loss.clear();	
}

INSTANTIATE_LAYER_GPU_FUNCS(RNNBaseLayer);

}

