#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialSubSampling.cu"
#else

#include "../common.h"

void THNN_(SpatialSubSampling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           int kW, int kH,
           int dW, int dH)
{
  real *weight_data = THCTensor_(data)(state, weight);
  real *bias_data = THCTensor_(data)(state, bias);
  real *output_data;
  real *input_data;

  int nInputPlane = THCTensor_(size)(state, weight, 0);

  THCUNN_assertSameGPU(state, 4, input, output, weight, bias);
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;

    THArgCheck(input->size[0] == nInputPlane, 2, "invalid number of input planes");
    THArgCheck(nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THCTensor_(newContiguous)(state, input);
    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize3d)(state, output, nInputPlane, nOutputRows, nOutputCols);
    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run subsample kernel
    subsample<real, accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      input_data, output_data, weight_data, bias_data,
      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
    THCudaCheck(cudaGetLastError());
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;

    THArgCheck(input->size[1] == nInputPlane, 2, "invalid number of input planes");
    THArgCheck(nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THCTensor_(newContiguous)(state, input);
    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize4d)(state, output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run subsample kernel
    subsample<real, accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      input_data, output_data, weight_data, bias_data,
      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
    THCudaCheck(cudaGetLastError());
  }

  // clean
  THCTensor_(free)(state, input);

}

void THNN_(SpatialSubSampling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           int kW, int kH,
           int dW, int dH)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, weight, gradInput);

  int nInputPlane = THCTensor_(size)(state, weight, 0);

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];

    real *weight_data = THCTensor_(data)(state, weight);
    real *gradOutput_data = THCTensor_(data)(state, gradOutput);
    real *gradInput_data;

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    if (kH <= dH && kW <= dW) {
      subgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
    } else {
      subgradinputAtomic <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
    }
    THCudaCheck(cudaGetLastError());
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];

    real *weight_data = THCTensor_(data)(state, weight);
    real *gradOutput_data = THCTensor_(data)(state, gradOutput);
    real *gradInput_data;

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    if (kH <= dH && kW <= dW) {
      subgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
    } else {
      subgradinputAtomic <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
    }
    THCudaCheck(cudaGetLastError());
  }
}

void THNN_(SpatialSubSampling_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           int kW, int kH,
           int dW, int dH,
           float scale)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, gradWeight, gradBias);

  int nInputPlane = THCTensor_(size)(state, gradWeight, 0);

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];

    real *gradWeight_data = THCTensor_(data)(state, gradWeight);
    real *gradBias_data = THCTensor_(data)(state, gradBias);
    real *gradOutput_data = THCTensor_(data)(state, gradOutput);
    real *input_data;

    input = THCTensor_(newContiguous)(state, input);
    input_data = THCTensor_(data)(state, input);

    // cuda blocks & threads:
    dim3 blocks(nInputPlane);
    dim3 threads(32,8);

    // run gradweight kernel
    subgradweight<real, accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      input_data, gradOutput_data, gradWeight_data, gradBias_data,
      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, scale);
    THCudaCheck(cudaGetLastError());
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];

    real *gradWeight_data = THCTensor_(data)(state, gradWeight);
    real *gradBias_data = THCTensor_(data)(state, gradBias);
    real *gradOutput_data = THCTensor_(data)(state, gradOutput);
    real *input_data;

    input = THCTensor_(newContiguous)(state, input);
    input_data = THCTensor_(data)(state, input);

    // cuda blocks & threads:
    dim3 blocks(nInputPlane);
    dim3 threads(32,8);

    // run gradweight kernel
    long sl;
    for (sl=0; sl<nbatch; sl++) {
      subgradweight<real, accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        input_data + sl*input->stride[0],
        gradOutput_data + sl*gradOutput->stride[0],
        gradWeight_data, gradBias_data,
        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, scale);
    }
    THCudaCheck(cudaGetLastError());
  }

  // clean
  THCTensor_(free)(state, input);

}

#endif
