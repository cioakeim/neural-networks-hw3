#include <iostream>
#include "CommonLib/basicFuncs.hpp"
#include "CommonLib/basicStructs.hpp"
#include "CommonLib/eventTimer.hpp"
#include "CommonLib/cifarHandlers.hpp"
#include "MLP/ActivationFunctions.hpp"
#include "MLP/MLP.hpp"

#define INPUT_SIZE 3072

namespace E=Eigen;

int main(){
  EventTimer et;
  std::string dataset_path="../data/cifar-10-batches-bin";
  int training_size=2000;
  int test_size=1000;
  int batch_size=50;

  et.start("Load dataset");
  Cifar10Handler c10=Cifar10Handler(dataset_path);
  SampleMatrix training_set=c10.getTrainingMatrix(training_size);
  SampleMatrix test_set=c10.getTestMatrix(test_size);
  /*
  normalizeDataset(training_set.vectors,
                   test_set.vectors);
  */
  normalizeImageDataset(training_set.vectors, test_set.vectors, 3);
  et.stop();

  std::cout<<"Cool"<<std::endl;

  MLP mlp=MLP(training_set,test_set,batch_size);
  std::cout<<"Cool"<<std::endl;

  LayerConfig config;
  std::vector<int> layer_sizes={1024,512,124};
  config.batch_size=50;
  config.layer_type=FeedForward;
  config.f=reLU;
  config.f_dot=reLUder;
  config.input_size=INPUT_SIZE;
  config.optimizer_mode=Adam;
  config.adam_config.rate=1e-3;
  config.adam_config.beta_1=0.9;
  config.adam_config.beta_2=0.999;


  for(auto l_size:layer_sizes){
    config.ff_config.feedforward_output=l_size;
    mlp.addLayer(config);
  }
  std::cout<<"Cool"<<std::endl;
  config.ff_config.feedforward_output=10;
  config.layer_type=SoftMax;
  mlp.addLayer(config);

  float loss;

  for(int i=0;i<10;i++){
    loss=mlp.runEpoch();
    std::cout<<"Loss: "<<loss<<std::endl;
  }


}
