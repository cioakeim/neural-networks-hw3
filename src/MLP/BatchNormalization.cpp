#include "MLP/BatchNormalization.hpp"


void BatchNormHandler::init(OptimizerConfig opt_config){
  mean=0; var=1;
  scale=E::MatrixXf(2,1);
  opt_config.adam.rate/=10;
  opt_config.sgd.rate/=10;
  opt.configure(opt_config, scale);
  scale(0,0)=0;
  scale(1,0)=1;
}


void BatchNormHandler::normalize(const MatrixXf& activation,
                                 bool is_training){
  float m,v;
  if(is_training){
    m=activation.mean();
    v=(activation.array()-m).array().pow(2).mean();
    mean=ALPHA*mean+(1-ALPHA)*m;
    var=ALPHA*var+(1-ALPHA)*v;
  }
  else{
    m=mean;
    v=var;
  }
  u_norm=(activation.array()-m)/sqrt(var+EPSILON);
}


void BatchNormHandler::update(const MatrixXf& error){
  E::MatrixXf scale_grad(2,1);
  scale_grad(0)=error.colwise().sum().mean();
  scale_grad(1)=error.cwiseProduct(u_norm).colwise().sum().mean();
}
