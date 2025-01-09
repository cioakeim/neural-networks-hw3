#include <iostream>
#include "CommonLib/basicFuncs.hpp"
#include "CommonLib/basicStructs.hpp"
#include "CommonLib/eventTimer.hpp"
#include "CommonLib/cifarHandlers.hpp"
#include "MLP/ActivationFunctions.hpp"
#include "MLP/MLP.hpp"
#include "AutoEncoder/AutoEncoder.hpp"
#include "AutoEncoder/Config.hpp"

#define INPUT_DIM 32
#define INPUT_CHANNEL 3

namespace E=Eigen;
using InterfacePtr=std::shared_ptr<LayerInterface>;


int main(int argc,char* argv[]){
  EventTimer et;
  std::string config_filepath="../data/configs/test";
  // Config
  // Defaults
  GeneralConfig gen_config;
  gen_config.dataset_path="../data/cifar-10-batches-bin";
  gen_config.run_path="../data/AutoEncoder/default_run";
  gen_config.training_size=10000;
  gen_config.test_size=2000;
  gen_config.batch_size=100;
  gen_config.epochs=15;
  OptimizerConfig opt_config;
  opt_config=opt_config;
  opt_config.type=Adam;
  opt_config.adam.rate=5e-4;
  opt_config.adam.beta_1=0.9;
  opt_config.adam.beta_2=0.999;
  AutoEncoderConfig aenc_config;
  aenc_config.stack_sizes={1024,512,124};
  aenc_config.stack_types={FeedForward,FeedForward};
  aenc_config.lock_weights=true;
  aenc_config.f=linear;aenc_config.f_dot=linearder;
  if(argc!=1){
    std::string arg=argv[1];
    if(arg!="-d"){
      std::cout<<"Config found!"<<std::endl;
      config_filepath=argv[1];
    }
    configGeneral(gen_config,config_filepath+"/general.txt");
    configAutoEncoder(aenc_config,config_filepath+"/aenc.txt");
    configOptimizer(opt_config,config_filepath+"/opt.txt");
  }

  et.start("Load dataset");
  Cifar10Handler c10=Cifar10Handler(gen_config.dataset_path);
  SampleMatrix training_set=c10.getTrainingMatrix(gen_config.training_size);
  SampleMatrix test_set=c10.getTestMatrix(gen_config.test_size);
  std::vector<NormalizationParams> params;
  normalizeImageDataset(training_set.vectors, test_set.vectors, 3,params);
  et.stop();


  AutoEncoder aenc=AutoEncoder(training_set,test_set,gen_config.batch_size);
  aenc.setWeightsLockable(aenc_config.lock_weights);
  std::cout<<"Cool"<<std::endl;

  LayerProperties properties;
  properties.opt_config=opt_config;
  properties.layer_type=MSE;
  properties.batch_normalization=true;

  InterfacePtr input=std::make_shared<LayerInterface>();
  input->width=input->height=32;
  input->channels=3;
  input->f=aenc_config.f;
  input->f_dot=aenc_config.f_dot;
  aenc.addInterfaceStack(input);
  std::cout<<"Cool"<<std::endl;


  // Layer-wise training
  std::string log_path=config_filepath+"/logs";
  int stack_idx=0;
  ensure_a_path_exists(log_path);
  for(unsigned int i=0;i<aenc_config.stack_sizes.size();i++){
    const int layer_size=aenc_config.stack_sizes[i];
    const LayerType layer_type=aenc_config.stack_types[i];
    std::ofstream log(log_path+"/run_"+std::to_string(layer_size)+".csv");
    if(!log.is_open()){
      std::cerr<<"CAN'T OPEN LOG: "<<log_path+"/run_"+std::to_string(layer_size)+".csv"<<std::endl;
      exit(1);
    }
    InterfacePtr interface=std::make_shared<LayerInterface>(*input);
    interface->height=layer_size;
    interface->width=interface->channels=1;
    aenc.addInterfaceStack(interface);
    std::cout<<"Cool"<<std::endl;
    aenc.addLayerStack(properties);
    std::cout<<"Cool"<<std::endl;
    float J_train,J_test;
    float accuracy;
    et.start("Run epochs");
    std::cout<<"Epochs: "<<std::endl;
    for(int i=0;i<gen_config.epochs;i++){
      std::cout<<"Epoch: "<<i<<std::endl;
      J_train=aenc.runEpoch();
      std::cout<<"Loss: "<<J_train<<std::endl;
      aenc.testModel(test_set, J_test, accuracy);
      std::cout<<"Test loss: "<<J_test<<std::endl;
      log<<i<<","<<J_train<<","<<J_test<<"\n";
    }
    log.close();
    properties.layer_type=layer_type;
    et.stop();
    aenc.setStorePath(config_filepath+"/network_fine_tuned"+
                      std::to_string(stack_idx++));
    aenc.store();
  }
  // If weights were locked, time to fine-tune
  if(aenc_config.lock_weights){
    std::ofstream log(log_path+"/run_fine_tune.csv");
    if(!log.is_open()){
      std::cerr<<"CAN'T OPEN LOG: "<<log_path+"/run_fine_tune.csv"<<std::endl;
      exit(1);
    }
    float rate=(opt_config.type==Adam)?
      (opt_config.adam.rate):(opt_config.sgd.rate);
    aenc.setLearningRate(rate/10);
    aenc.unlockAll();
    float J_train,J_test;
    float accuracy;
    et.start("Run epochs");
    std::cout<<"Epochs: "<<std::endl;
    const int fine_tune_epochs=gen_config.epochs*aenc_config.stack_sizes.size()/3;
    for(int i=0;i<fine_tune_epochs;i++){
      J_train=aenc.runEpoch();
      std::cout<<"Loss: "<<J_train<<std::endl;
      aenc.testModel(test_set, J_test, accuracy);
      std::cout<<"Test loss: "<<J_test<<std::endl;
      log<<i<<","<<J_train<<","<<J_test<<"\n";
    }
    log.close();
    et.stop();
    aenc.setStorePath(config_filepath+"/network_fine_tuned"+
                      std::to_string(stack_idx++));
    aenc.store();
  }

  et.displayIntervals();
  et.writeToFile(log_path+"/time_info.txt");
}
