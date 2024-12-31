#include "MLP/SoftMaxLayer.hpp"
#include "CommonLib/basicFuncs.hpp"


void SoftMaxLayer::forward(const MatrixXf& input){
  FeedForwardLayer::forward(input);
  const E::RowVectorXf maxCoeff=output.colwise().maxCoeff();
  // Subtract for numerical stability and exp
  const MatrixXf exps=(output.rowwise()-maxCoeff).array().exp();
  // Get sum of each column 
  const E::RowVectorXf col_sum=exps.colwise().sum();
  output=exps.array().rowwise()/col_sum.array();
  //std::cout<<"Soft max: "<<output.sum()<<std::endl;
}


void SoftMaxLayer::forward(){
  const MatrixXf& input=previous->outputRef();
  forward(input);
}


float SoftMaxLayer::loss(const VectorXi& labels){
  int sample_size=labels.size();
  VectorXf loss_array(sample_size);
  #pragma omp parallel for
  for(int i=0;i<sample_size;i++){
    loss_array(i)=log(output(labels(i),i));
  }
  return -loss_array.sum();
}


void SoftMaxLayer::backward(const MatrixXf& input,
                            const VectorXi& labels){
  MatrixXf error=output;
  const int sample_size=output.cols();
  #pragma omp parallel for
  for(int i=0;i<sample_size;i++){
    error(labels(i),i)--;
  }

  
  const E::MatrixXf& in=(previous==nullptr)?input:previous->outputRef();
  MatFunction func=previous->getFDot();
  input_error=(weights.transpose()*error).cwiseProduct(func(in));
  updateWeights(in, error);
  /**
  std::cout<<"Soft limits: "<<input_error.minCoeff()<<" "<<input_error.maxCoeff()<<std::endl;
  std::cout<<"Soft dims: "<<input_error.rows()<<" "<<input_error.cols()<<std::endl;
  std::cout<<"Soft error div: "<<input_error.array().pow(2).mean()<<std::endl;
  */
}


int SoftMaxLayer::prediction_success(const VectorXi& labels){
  int cnt=0;
  int size=labels.size();
  for(int i=0;i<size;i++){
    E::MatrixXf::Index idx;
    output.col(i).maxCoeff(&idx);
    cnt+=(labels(i)==idx);
  }
  return cnt;
}
