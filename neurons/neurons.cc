// hello.cc
#include <iostream>
#include <sstream>
#ifdef _WIN32
    #include <Windows.h>
#else
    #include <unistd.h>
#endif
#include <node.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "neuronengine.h"
namespace mc {

using v8::FunctionCallbackInfo;
using v8::Isolate;
using v8::Local;
using v8::Object;
using v8::String;
using v8::Value;

NeuronEngine myEngine;

  int * toIntArray(char *data,int length) { //字符数组转int数组
      const char * split = ",";
      char *token;
      int *intArray;
      intArray = new int[length];
      int arrayLen = 0;
      token = strtok(data,split);

      while(token!=NULL) {
          intArray[arrayLen] = atoi(token);
          arrayLen++;
          token = strtok(NULL,split);
      }

      for(int i=0;i<arrayLen;i++){
          //std::cout<<"vec"<<intArray[i]<<std::endl;
      }

      return intArray;
  }

void Learn(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    String::Utf8Value utf8Value(args[0]->ToString());
    std::string sourceData = std::string(*utf8Value);


    char strData[1024];
    //bzero(strData, 1024);
    strcpy(strData, sourceData.c_str());

    int * data;
    data = toIntArray(strData,sourceData.length());

    if (data[0] == 1) {
        std::cout<<"\n--------------Learn-------------"<<std::endl;
        int length = data[1];
        uint8_t *vec = new uint8_t[length]; //data[1]:length
        int j = 3;
        std::cout<<"vec:"<<std::endl;
        for (int i = 0;i < length;i++) {
            vec[i] =  data[j];
            std::cout<<i<<":"<<data[j]<<std::endl;
            j++;
        }
        

        int result =  myEngine.Learn(data[2],vec,length); //data[2]:category
        std::cout<<" learn result: (neurons size) "<<result<<std::endl;

        std::string result_int = std::to_string(result);
        args.GetReturnValue().Set(String::NewFromUtf8(isolate, result_int.c_str()));
    }
    else{
        args.GetReturnValue().Set(String::NewFromUtf8(isolate, "Wrong arguments"));
    }
    /*
    NeuronEngine myEngine;
    uint8_t vec[5] =  {10,11,12,13,14};
    int r =  myEngine.Learn(1,vec,5);
    std::string result = std::to_string(r)+input;
    args.GetReturnValue().Set(String::NewFromUtf8(isolate, result.c_str()));*/
}


void Classify(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    String::Utf8Value utf8Value(args[0]->ToString());
    std::string sourceData = std::string(*utf8Value);


    char strData[1024];
    strcpy(strData, sourceData.c_str());

    int * data;
    data = toIntArray(strData,sourceData.length());

    if (data[0] == 0) {
        cout<<"\n--------------Classify-------------"<<endl;
        int length = data[1];
        uint8_t *vec = new uint8_t[length]; //data[1]:length
        int j = 2;
        for (int i = 0;i < data[1];i++) {
            vec[i] =  data[j];
            std::cout<<i<<":"<<data[j]<<std::endl;
            j++;

        }

        int result =  myEngine.Classify(vec,length);
        if(result > 0){
           std::cout<<" Classify result category:"<<result<<std::endl;
        }
        else{
            std::cout<<" nukonw:"<<result<<std::endl;
        }

        std::string result_int = std::to_string(result);
        args.GetReturnValue().Set(String::NewFromUtf8(isolate, result_int.c_str()));
    }
    else{
        args.GetReturnValue().Set(String::NewFromUtf8(isolate, "Wrong arguments"));
    }
}

void init(Local<Object> exports) {
  NODE_SET_METHOD(exports, "Learn", Learn);
  NODE_SET_METHOD(exports, "Classify", Classify);
}

NODE_MODULE(neurons, init)

}  // namespace demo