/***************************************************************************
**
** Autohr : Yunpeng Song
** Contact: 413740951@qq.com
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program.  If not, see <http://www.gnu.org/licenses/>.
**
****************************************************************************/
#include "neuronengine.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <string.h>

static bool neuron_compare(Neuron* a,Neuron* b);

NeuronEngine::NeuronEngine():NeuronEngine(CONTEXT_START){}
NeuronEngine::NeuronEngine(uint8_t context):
  NeuronEngine(context,
               MODE_RBF,NORM_L1,
               DEFAULT_NEURON_AIF_MIN,DEFAULT_NEURON_AIF_MAX,
               DEFAULT_NEURON_NUMBER,DEFAULT_NEURON_MEM_LENGTH)
{

}

NeuronEngine::NeuronEngine(uint8_t context,int mode):
  NeuronEngine(context,
               mode,NORM_L1,
               DEFAULT_NEURON_AIF_MIN,DEFAULT_NEURON_AIF_MAX,
               DEFAULT_NEURON_NUMBER,DEFAULT_NEURON_MEM_LENGTH)
{

}

NeuronEngine::NeuronEngine(uint8_t context,
                    int mode,int norm,
                    uint16_t minaif,uint16_t maxaif):
  NeuronEngine(context,
               MODE_RBF,NORM_L1,
               minaif,maxaif,
               DEFAULT_NEURON_NUMBER,DEFAULT_NEURON_MEM_LENGTH)

{

}

NeuronEngine::NeuronEngine(uint8_t context,
                         int mode,int norm,
                         uint16_t minaif,uint16_t maxaif,
                         uint16_t max_neuron_memlen,uint16_t max_neuron_number)
{
  if(context<CONTEXT_START){context=CONTEXT_START;}
  context_index_ = context;

  Begin(mode,norm,minaif,maxaif,max_neuron_memlen,max_neuron_number);
}

NeuronEngine::~NeuronEngine(){
  ResetEngine();
}


void NeuronEngine::Begin(int mode,int norm, uint16_t minaif,uint16_t maxaif, uint16_t memlen,uint16_t maxneuronnumber)
{
  if(minaif>maxaif){
    minaif = DEFAULT_NEURON_AIF_MIN;
    maxaif = DEFAULT_NEURON_AIF_MAX;
  }

  //step1.reset the engine
  ResetEngine();

  //step2 norm
  switch(norm){
  case NORM_L1:
  case NORM_LSUP:
    norm_ = (int)norm;
      break;
  default:
    norm_ = NORM_L1;
  }

  //step3 mode
  SetMode((enum MODE)mode);

  //step4 aif
  min_aif_ = minaif;
  max_aif_ = maxaif;
  max_neuron_number_ = maxneuronnumber;
  max_neuron_memlen_ = memlen;
}
void NeuronEngine::SetMode(MODE mode){

  switch(mode){
  case MODE_RBF:
  case MODE_KNN:
    mode_ = mode;
    break;
  default:
    mode_ = MODE_RBF;
  }
}

int NeuronEngine::Mode(){
  return mode_;
}
void NeuronEngine::ResetEngine(){
  last_neuron_index_ = 0;

  //clear neuronlist
  ClearNeuronList();
  firing_list_.clear();

  is_loading_neuron_=false;
}
const Neuron* NeuronEngine::ReadNeuron(int index){
  return neuron_list_.at(index);
}
int NeuronEngine::Learn(uint16_t cat, uint8_t *vec, int len){
  if(is_loading_neuron_)return -1;//loading neurons, DONOT LEARN!
  int ret = 0;

  if(vec==NULL || len==0){
    return -1;
  }
  if(neuron_list_.size()>=max_neuron_number_){
    return neuron_list_.size();
  }

  switch(mode_){
  case MODE_RBF:
  case MODE_KNN:
    /* TODO: according to the document and test cases,
     * when engine learing in KNN mode, it is the same
     * as RBF. AIF is working.
     * But in the Recognition mode, AIF will be disabled.
     * FIXME: try more test cases on hw to figure out if it works.
     */
    ret = LearnRBF(cat,vec,len);
    break;
  }
  return ret;
}

int NeuronEngine::Classify(uint8_t vec[], int len, uint16_t nid[], int *nidlen)
{
  if(is_loading_neuron_)return -1;//loading neurons, DONOT LEARN!

  vector<Neuron* >::iterator iter;
  int index=0;
  int ret = 0;

  if(vec==NULL || len==0){
      return 0;
  }

  switch(mode_){
  case MODE_RBF:
      ret = ClassifyRBF(vec,len);
      break;
  case MODE_KNN:
      ret = ClassifyKNN(vec,len);
      break;
  }

  if(nid && nidlen){
    for(iter = firing_list_.begin();iter<firing_list_.end();iter++){
      Neuron* ptr = *iter;
      nid[index] = ptr->Index();
      index++;
    }
    *nidlen = firing_list_.size();
  }

  return ret;
}

int NeuronEngine::Classify(uint8_t vec[], int len){
  return Classify(vec,len,NULL,NULL);
}

int NeuronEngine::neuronSize() const{
  return neuron_list_.size();
}

int NeuronEngine::LearnRBF(int cat, uint8_t *vec, int len){
  vector<Neuron*>::iterator iter = neuron_list_.begin();
  int dist=0;
  int minDist = DEFAULT_NEURON_AIF_MAX;
  bool isExist=false;

  //step1 clear the firing list
  firing_list_.clear();

  //step2 traversing all neurons
  for(iter = neuron_list_.begin();iter<neuron_list_.end();iter++){
    Neuron * ptr = *iter;
    if(!ptr){
        continue;
    }

    //calc the distance
    if(ValidateNeuronFiring(ptr,vec,len,&dist))
    {/**Firing!!!**/
      firing_list_.push_back(ptr);
      if(ptr->Cat() == cat){
        //if it exists in its own neuron. do not create new neuron
        isExist=true;
      }else{
        //since you are firing, shrink your AIF.
        ptr->SetAIF(dist);
      }
    }
    if(dist<minDist){
      minDist = dist;
    }
  }

  //step3. process the new input
  if(!isExist ){
    CreateNeuron(cat,minDist,min_aif_,max_aif_,vec,len);
  }
  return neuron_list_.size();
}

int NeuronEngine::ClassifyRBF(uint8_t vec[], int len)
{
  vector<Neuron*>::iterator iter = neuron_list_.begin();
  int dist=0;
  int minDist = DEFAULT_NEURON_AIF_MAX;
  int retCat=0;

  //step1 clear the firing list
  firing_list_.clear();

  //step2 traversing all neurons
  for(iter = neuron_list_.begin();iter<neuron_list_.end();iter++){
    Neuron * ptr = *iter;
    if(!ptr){
        continue;
    }

    //calc the distance
    if(ValidateNeuronFiring(ptr,vec,len,&dist))
    {//firing
      if(dist<minDist){//find the most close neuron
          minDist = dist;
          retCat = ptr->Cat();
      }
      firing_list_.push_back(ptr);
      ptr->firing_ = dist;//update the firing distance
    }
  }

  return retCat;
}
int NeuronEngine::ClassifyKNN(uint8_t vec[], int len){
  vector<Neuron*>::iterator iter = neuron_list_.begin();
  int dist=0;
  int minDist = DEFAULT_NEURON_AIF_MAX;
  int retCat=0;

  //step1 clear the firing list
  firing_list_.clear();

  //step2 traversing all neurons
  for(iter = neuron_list_.begin();iter<neuron_list_.end();iter++){
    Neuron * ptr = *iter;
    if(!ptr){
      continue;
    }

    //calc the distance
    ValidateNeuronFiring(ptr,vec,len,&dist);//no firing checking.
    if(dist<minDist){//find the most close neuron
      minDist = dist;
      retCat = ptr->Cat();
    }
    ptr->firing_ = dist;//save the dist
    firing_list_.push_back(ptr);
  }

  /*TODO: add a post processing function to sort the result*/
  sort(firing_list_.begin(),firing_list_.end(),neuron_compare);

  return retCat;
}
bool NeuronEngine::ValidateNeuronFiring(Neuron *nptr, uint8_t *vec, int len,int* ptrDist)
{
  int dist=0;

  switch(norm_){
  case NORM_L1:
    dist = nptr->CalcDistanceL1(vec,len);
    break;
  case NORM_LSUP:
    dist = nptr->CalcDistanceLsup(vec,len);
    break;
  }

  *ptrDist = dist;
  if(dist<nptr->Aif()){
    return true;
  }
  return false;
}
bool NeuronEngine::CreateNeuron(int cat, int aif, int minAif, int maxAif,uint8_t vec[], int len){
  if(neuron_list_.size()>=(unsigned int)max_neuron_number_){
    return false;
  }
  //create neurons only if not exist.
  Neuron * nptr = new Neuron(last_neuron_index_,minAif,maxAif,max_neuron_memlen_);
  last_neuron_index_++;

  //init the neuron
  nptr->WriteNeuronMem(cat,vec,len);
  nptr->SetAIF(aif);

  //append the new neuron
  neuron_list_.push_back(nptr);
  return true;
}

void NeuronEngine::ClearNeuronList()
{
  vector<Neuron*>::iterator iter = neuron_list_.begin();
  for(iter = neuron_list_.begin();iter<neuron_list_.end();iter++){
    Neuron * ptr = *iter;
    if(ptr){
      delete(ptr);
    }
  }
  neuron_list_.clear();
}

void NeuronEngine::BeginRestoreMode(){
  ResetEngine();
  is_loading_neuron_ = true;
}
void NeuronEngine::EndRestoreMode(){
  is_loading_neuron_ = false;
}
int NeuronEngine::RestoreNeuron(uint8_t *buffer, uint16_t len, uint16_t cat, uint16_t aif, uint16_t minAIF){
  bool ret = CreateNeuron(cat,aif,minAIF,max_aif_,buffer,len);
  if(ret){
    return this->neuron_list_.size();
  }
  return -1;
}

bool neuron_compare(Neuron* a,Neuron* b){
  return a->Firing() < b->Firing();
}

Neuron::Neuron(uint16_t index,uint16_t minaif,uint16_t maxaif,uint16_t max_mem_length):
  max_mem_len_(max_mem_length),max_aif_(maxaif),min_aif_(minaif)
{
  index_=index;
  neuron_mem_ = NULL;
  neuron_mem_len_ = 0;
  aif_=0;
  firing_=0;
  cat_=0;
}

Neuron::~Neuron()
{
  Reset();
}

void Neuron::Reset(){
  //clear the memory
  if(neuron_mem_!=NULL){
    free(neuron_mem_);
  }
  neuron_mem_=NULL;
  neuron_mem_len_=0;

  firing_ = 0;
  cat_=0;

  //set aif;
  SetAIF(0);
}
bool Neuron::WriteNeuronMem(uint16_t cat, uint8_t src_vector[], int src_len){
  if(src_vector==NULL ||
     src_len>max_mem_len_ ||
     src_len<=0 ){
    return false;
  }

  //reset the neuron
  Reset();

  //malloc the memory
  neuron_mem_ = (uint8_t*)malloc(max_mem_len_);
  if(neuron_mem_==NULL){
    return false;
  }
  memset(neuron_mem_,0,max_mem_len_);

  //copy the vector to my vector.
  memcpy(neuron_mem_,src_vector,src_len);
  neuron_mem_len_ = src_len;
  cat_ = cat;

  return true;
}

void Neuron::SetAIF(uint16_t aif){
  if(aif_<min_aif_){
    aif_ = min_aif_;
  }else if(aif_>max_aif_){
    aif_ = max_aif_;
  }else{
    aif_ = aif;
  }
}
uint16_t Neuron::CalcDistanceL1(uint8_t *src_vector, int src_len){
  int dist = 0;

  //check input
  if(src_vector==NULL ||
    src_len>max_mem_len_ || src_len<0){
    return -1; //distance cannot be smaller than zero.
  }

  //check the array size.
  for(int i=0;i<src_len;i++){
    dist = dist + abs(src_vector[i]-neuron_mem_[i]);
  }
  return dist;
}
uint16_t Neuron::CalcDistanceLsup(uint8_t *src_vector, int src_len){
  int dist = 0;
  int val=0;

  //check input
  if(src_vector==NULL ||
    src_len>max_mem_len_ ||  src_len<=0){
    return -1; //distance cannot be smaller than zero.
  }

  //check the array size.
  for(int i=0;i<src_len;i++){
    val = abs(src_vector[i]-neuron_mem_[i]);
    if(val>dist){
      dist = val;
    }
  }
  return dist;
}

uint16_t Neuron::ReadNeuronMem(uint8_t buffer[]) const{
  if(buffer){
    memcpy(buffer,neuron_mem_,neuron_mem_len_);
    return neuron_mem_len_;
  }
  return -1;
}
