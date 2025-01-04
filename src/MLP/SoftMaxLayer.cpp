#include "MLP/SoftMaxLayer.hpp"
#include "CommonLib/basicFuncs.hpp"


void SoftMaxLayer::forward(const PassContext& context){
  FeedForwardLayer::forward(context);
  MatrixXf& output=output_interface->forward_signal;

  const E::RowVectorXf maxCoeff=output.colwise().maxCoeff();
  // Subtract for numerical stability and exp
  const MatrixXf exps=(output.rowwise()-maxCoeff).array().exp();
  // Get sum of each column 
  const E::RowVectorXf col_sum=exps.colwise().sum();
  output=exps.array().rowwise()/col_sum.array();
  //std::cout<<"Soft max: "<<output.sum()<<std::endl;
}


float SoftMaxLayer::loss(const PassContext& context){
  int sample_size=context.labels.size();
  MatrixXf& output=output_interface->forward_signal;
  VectorXf loss_array(sample_size);
  #pragma omp parallel for
  for(int i=0;i<sample_size;i++){
    loss_array(i)=log(output(context.labels(i),i));
  }
  return -loss_array.sum();
}


void SoftMaxLayer::backward(const PassContext& context){
  MatrixXf& output=output_interface->forward_signal;
  MatrixXf error=output;
  const int sample_size=output.cols();
  #pragma omp parallel for
  for(int i=0;i<sample_size;i++){
    error(context.labels(i),i)--;
  }
  const E::MatrixXf& in=(input_interface->type==Input)?
    context.input:input_interface->forward_signal;
  // Pass backward if needed
  if(input_interface->type!=Input){
    MatFunction func=input_interface->f_dot;
    input_interface->backward_signal=
      (weights.transpose()*error).cwiseProduct(func(in));
  } 
  if(!lockParams){
    updateWeights(in, error);
  }
  /**
  std::cout<<"Soft limits: "<<input_error.minCoeff()<<" "<<input_error.maxCoeff()<<std::endl;
  std::cout<<"Soft dims: "<<input_error.rows()<<" "<<input_error.cols()<<std::endl;
  std::cout<<"Soft error div: "<<input_error.array().pow(2).mean()<<std::endl;
  */
}


int SoftMaxLayer::prediction_success(const PassContext& context){
  int cnt=0;
  int size=context.labels.size();
  MatrixXf& output=output_interface->forward_signal;
  for(int i=0;i<size;i++){
    E::MatrixXf::Index idx;
    output.col(i).maxCoeff(&idx);
    cnt+=(context.labels(i)==idx);
  }
  return cnt;
}
