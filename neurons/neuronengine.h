/***************************************************************************
**
** Autohr : Yunpeng Song
** Contact: 413740951@qq.com
** Date:    2016-11-01
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
#ifndef NEURONENGINE_H
#define NEURONENGINE_H
#include <stdint.h>
#include <vector>

using namespace std;

class Neuron;
class NeuronEngine
{
public:

  enum NORM{
    NORM_L1=0,
    NORM_LSUP=1
  };
  enum MODE{
    MODE_RBF=0,
    MODE_KNN=1
  };

  static constexpr int16_t DEFAULT_NEURON_NUMBER=128;
  static constexpr int16_t DEFAULT_NEURON_MEM_LENGTH=128;
  static constexpr int16_t DEFAULT_NEURON_AIF_MIN=0x02;
  static constexpr int16_t DEFAULT_NEURON_AIF_MAX=0x4000;
  static constexpr int16_t CONTEXT_START=1;

  explicit NeuronEngine();
  explicit NeuronEngine(uint8_t context);
  explicit NeuronEngine(uint8_t context,int mode);
  explicit NeuronEngine(uint8_t context,
                        int mode,int norm,
                        uint16_t minaif,uint16_t maxaif);
  explicit NeuronEngine(uint8_t context,
                        int mode,int norm,
                        uint16_t minAIF,uint16_t maxAIF,
                        uint16_t maxNeuronMemLen,
                        uint16_t maxNeuronNumber);
  ~NeuronEngine();

  virtual void ResetEngine();//reset the whole engine

  void SetMode(enum MODE Mode);
  int Mode();

  /*purpose:learn the input vectors
   *input:
   * @cat,the input vector's category
   * @vec, the input vector
   * @len, the input length
   *return:
   * @the current neruon number
   */
  virtual int Learn(uint16_t cat,uint8_t * vec,int len);
  const Neuron * ReadNeuron(int index);// Read one data
  int neuronSize() const;//how many neurons in the engine

  //purpose:begin loading neurons status. The engine STOPs learning and classify
  void BeginRestoreMode();
  /*purpose:load the neuron to engine. NO RBF OR KNN CHECK!
   *input:
   * @buffer,the neuron buffer
   * @len, the length of the buffer
   * @cat, the cat of this neuron
   * @aif, the aif of this neuron
   * @minAIF, the minAIF of this neuron
   *return:
   * @the current neruon number
   */
  int RestoreNeuron(uint8_t* buffer, uint16_t len,
                     uint16_t cat,
                     uint16_t aif,uint16_t minAIF);
  //purpose:end loading neurons status.  The engine STARTs learning and classify
  void EndRestoreMode();

  /*purpose:classify the input vectors
   *input:
   * @vec, the input vector
   * @len, the input length
   *output:
   * @nid, the neurons' index firing in this classify ops
   * @nidLen, the length of the output NID array
   *return:
   * @return the recognized category
   * @return 0 if not recognized
   */
  virtual int Classify(uint8_t vec[],int len,uint16_t nid[],int *nidLen);
  virtual int Classify(uint8_t vec[],int len);

  //maximum memory length of each neuron
  int MaxNeuronMemLength() const{return max_neuron_memlen_;}
  int MaxNeuronNumber() const {return max_neuron_number_;}
  void ClearNeuronList();//clear the neuron list, including the ram.

protected:
  /* purpose:init the neuron engine with customized settings
   * input:
   * @mode,the classification mode. RBF or KNN
   * @norm, the distance calculation mode. L1 Lsup
   * @minAif, the minimum AIF of all neurons
   * @maxAIF, the max AIF of all neurons
   * @memLen, the memory size of each neuron
   * @maxNeuronNumber, the max number of neurons.
   * return:
   * true, if success
   */
  virtual void Begin(int mode,
                     int norm,
                     uint16_t minaif,uint16_t maxaif,
                     uint16_t max_neuron_memlen,
                     uint16_t max_neuron_number);

  bool CreateNeuron(int cat, int aif, int minAif, int maxAif,uint8_t vec[],int len);//create a new neuron and insert to neuron List.
  /*purpose:read the firing distances, call this after the classify functoion.
   *input:
   * @nptr, the neuron
   *output:
   * @ptrDist,the distance
   *return:
   * true, firing
   */
  bool ValidateNeuronFiring(Neuron* nptr,uint8_t *vec, int len, int * ptrDist);

  //learn the pattern in RBF mode.
  virtual int LearnRBF(int cat,uint8_t * vec,int len);

  virtual int ClassifyRBF(uint8_t vec[],int len);
  virtual int ClassifyKNN(uint8_t vec[],int len);
protected:
  vector<Neuron*> firing_list_;//saves all fired neurons;
  vector<Neuron*> neuron_list_;//saves all generated neuron data

  uint16_t max_aif_;
  uint16_t min_aif_;
  int norm_; //the distance calculation method: L1/Lsup
  int mode_; //the classify mode:KNN/RBF

  uint16_t max_neuron_number_;//default max neuron numbers
  uint16_t max_neuron_memlen_;//default max neuron memory length

  uint16_t last_neuron_index_;
  uint8_t context_index_;//the context id of this neuron.

  bool is_loading_neuron_;
};


class Neuron{
  friend class NeuronEngine;
public:
  Neuron(uint16_t index,uint16_t minaif,uint16_t maxaif,uint16_t memlen);
  ~Neuron();

  uint16_t Index()const {return index_;} //the index of this neuron
  uint16_t Aif() const{return aif_;}//return the aif of this neuron
  uint16_t MinAIF() const{return min_aif_;}
  uint16_t MaxAIF() const{return max_aif_;}

  uint16_t Firing() const{return firing_;}//the last calculated firing distance
  uint16_t Cat() const{return cat_;}//category of this neuron

  uint16_t ReadNeuronMem(uint8_t buffer[]) const;
  uint16_t NeuronMemLength() const{return neuron_mem_len_;}
protected:

  /*purpose:init this neuron's memory
   *input:
   * @cat, the category
   * @vec, the memory buffer
   * @len, the length of the input buffer
   *return:
   * @true, true if success
   *MUST BE CALLED BY NEURONENGINE!
   */
  bool WriteNeuronMem(uint16_t Cat,uint8_t vec[],int len);

  /*set the AIF of this neuron
   */
  void SetAIF(uint16_t Aif);

  /*reset this neuron*/
  void Reset();

  /* calc the L1 distance of this neuron
   * @n, input neuron,
   * @src_vector,the input vector for this neuron
   * @src_len, the length of the vector
   * return:
   * @the L1 distance of this neuron
   */
  uint16_t CalcDistanceL1(uint8_t *src_vector,int src_len);

  /* calc the L_sup distance of this neuron
   * @n, input neuron,
   * @src_vector,the input vector for this neuron
   * @src_len, the length of the vector
   * return:
   * @the L_sup distance of this neuron
   */
  uint16_t CalcDistanceLsup(uint8_t *src_vector,int src_len);

protected:
  const uint16_t max_aif_;
  const uint16_t min_aif_;
  const uint16_t max_mem_len_;

private:
  uint8_t *neuron_mem_;//vecotr memory buffer
  uint16_t neuron_mem_len_;

  uint16_t cat_;
  uint16_t aif_;
  uint16_t index_;

  uint16_t firing_;//the last calculated firing distance.
};

#endif // NEURONENGINE_H
