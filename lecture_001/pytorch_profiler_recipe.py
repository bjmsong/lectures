import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


##################################### 3. analyze execution time #####################################

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# PyTorch profiler is enabled through the context manager
# ProfilerActivity.CPU: PyTorch operators, TorchScript functions and user-defined code labels (record_function)
# record_shapes: whether to record shapes of the operator inputs
# record_function context manager: label arbitrary code ranges with user provided names (model_inference is used as a label)
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# operators can call other operators, self cpu time excludes time spent in children operator calls, while total cpu time includes it
"""
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  model_inference         4.54%       5.489ms       100.00%     120.845ms     120.845ms             1  
                     aten::conv2d         0.14%     164.000us        78.47%      94.833ms       4.742ms            20  
                aten::convolution         0.36%     432.000us        78.34%      94.669ms       4.733ms            20  
               aten::_convolution         0.27%     329.000us        77.98%      94.237ms       4.712ms            20  
         aten::mkldnn_convolution        77.39%      93.517ms        77.71%      93.908ms       4.695ms            20  
                 aten::batch_norm         0.07%      79.000us         8.20%       9.904ms     495.200us            20  
     aten::_batch_norm_impl_index         0.16%     199.000us         8.13%       9.825ms     491.250us            20  
          aten::native_batch_norm         7.74%       9.350ms         7.94%       9.599ms     479.950us            20  
                 aten::max_pool2d         0.05%      57.000us         6.59%       7.965ms       7.965ms             1  
    aten::max_pool2d_with_indices         6.54%       7.908ms         6.54%       7.908ms       7.908ms             1  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 120.845ms
"""

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# most of the time is spent in convolution(aten::mkldnn_convolution)
"""
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
         aten::mkldnn_convolution        77.39%      93.517ms        77.71%      93.908ms       4.695ms            20  
          aten::native_batch_norm         7.74%       9.350ms         7.94%       9.599ms     479.950us            20  
    aten::max_pool2d_with_indices         6.54%       7.908ms         6.54%       7.908ms       7.908ms             1  
                  model_inference         4.54%       5.489ms       100.00%     120.845ms     120.845ms             1  
                       aten::add_         0.65%     787.000us         0.65%     787.000us      28.107us            28  
                 aten::clamp_min_         0.53%     640.000us         0.53%     640.000us      37.647us            17  
                aten::convolution         0.36%     432.000us        78.34%      94.669ms       4.733ms            20  
                      aten::empty         0.34%     409.000us         0.34%     409.000us       2.045us           200  
                      aten::relu_         0.30%     358.000us         0.83%     998.000us      58.706us            17  
                      aten::addmm         0.30%     358.000us         0.32%     389.000us     389.000us             1  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 120.845ms

"""

# analyze performance of models executed on GPUs
model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

"""
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference         0.00%       0.000us         0.00%       0.000us       0.000us     118.992ms        92.48%     118.992ms      59.496ms             2  
                                        model_inference         1.75%       3.663ms        99.99%     209.209ms     209.209ms       0.000us         0.00%      11.578ms      11.578ms             1  
                                           aten::conv2d         0.06%     118.000us        75.57%     158.110ms       7.905ms       0.000us         0.00%       9.939ms     496.950us            20  
                                      aten::convolution         0.17%     362.000us        75.51%     157.992ms       7.900ms       0.000us         0.00%       9.939ms     496.950us            20  
                                     aten::_convolution         0.14%     286.000us        75.34%     157.630ms       7.881ms       0.000us         0.00%       9.939ms     496.950us            20  
                                aten::cudnn_convolution        39.45%      82.548ms        75.20%     157.344ms       7.867ms       8.039ms         6.25%       9.939ms     496.950us            20  
cudnn_infer_volta_scudnn_winograd_128x128_ldg1_ldg4_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.563ms         4.32%       5.563ms     427.923us            13  
                                 cudaDeviceGetAttribute         0.09%     191.000us         0.09%     191.000us       5.026us       1.449ms         1.13%       1.449ms      38.132us            38  
                                       aten::batch_norm         0.03%      69.000us         4.99%      10.449ms     522.450us       0.000us         0.00%     929.000us      46.450us            20  
                           aten::_batch_norm_impl_index         0.08%     167.000us         4.96%      10.380ms     519.000us       0.000us         0.00%     929.000us      46.450us            20  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 209.221ms
Self CUDA time total: 128.670ms
"""

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

"""
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference         0.00%       0.000us         0.00%       0.000us       0.000us     118.992ms        92.48%     118.992ms      59.496ms             2  
                                aten::cudnn_convolution        39.45%      82.548ms        75.20%     157.344ms       7.867ms       8.039ms         6.25%       9.939ms     496.950us            20  
cudnn_infer_volta_scudnn_winograd_128x128_ldg1_ldg4_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.563ms         4.32%       5.563ms     427.923us            13  
                                 cudaDeviceGetAttribute         0.09%     191.000us         0.09%     191.000us       5.026us       1.449ms         1.13%       1.449ms      38.132us            38  
                                 aten::cudnn_batch_norm         1.33%       2.788ms         4.88%      10.213ms     510.650us     929.000us         0.72%     929.000us      46.450us            20  
cudnn_infer_volta_scudnn_128x64_relu_xregs_large_nn_...         0.00%       0.000us         0.00%       0.000us       0.000us     815.000us         0.63%     815.000us     407.500us             2  
cudnn_infer_volta_scudnn_128x64_3dconv_fprop_small_n...         0.00%       0.000us         0.00%       0.000us       0.000us     626.000us         0.49%     626.000us     626.000us             1  
void cudnn::bn_fw_tr_1C11_singleread<float, 512, tru...         0.00%       0.000us         0.00%       0.000us       0.000us     536.000us         0.42%     536.000us      35.733us            15  
void cudnn::winograd::generateWinogradTilesKernel<0,...         0.00%       0.000us         0.00%       0.000us       0.000us     444.000us         0.35%     444.000us      34.154us            13  
                              cudaStreamCreateWithFlags         1.00%       2.091ms         1.00%       2.091ms     130.688us     397.000us         0.31%     397.000us      24.812us            16  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 209.221ms
Self CUDA time total: 128.670ms
"""


##################################### 4. analyze memory consumption #####################################

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# the amount of memory (used by the model’s tensors) that was allocated (or released) during the execution of the model’s operators
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

"""
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                      aten::empty         0.60%     528.000us         0.60%     528.000us       2.640us      94.84 Mb      94.84 Mb           200  
    aten::max_pool2d_with_indices         7.81%       6.848ms         7.81%       6.848ms       6.848ms      11.48 Mb      11.48 Mb             1  
                      aten::addmm         0.12%     106.000us         0.14%     121.000us     121.000us      19.53 Kb      19.53 Kb             1  
                       aten::mean         0.04%      37.000us         0.16%     136.000us     136.000us      10.00 Kb      10.00 Kb             1  
              aten::empty_strided         0.01%       5.000us         0.01%       5.000us       5.000us           4 b           4 b             1  
                     aten::conv2d         0.17%     148.000us        77.33%      67.774ms       3.389ms      47.37 Mb           0 b            20  
                aten::convolution         0.47%     416.000us        77.16%      67.626ms       3.381ms      47.37 Mb           0 b            20  
               aten::_convolution         0.31%     270.000us        76.68%      67.210ms       3.361ms      47.37 Mb           0 b            20  
         aten::mkldnn_convolution        75.91%      66.534ms        76.37%      66.940ms       3.347ms      47.37 Mb           0 b            20  
                aten::as_strided_         0.14%     123.000us         0.14%     123.000us       6.150us           0 b           0 b            20  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 87.648ms
"""

#######################   5. Using tracing functionality  #######################
# You can examine the sequence of profiled operators and CUDA kernels in Chrome trace viewer (chrome://tracing):
model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")


#######################  6. Examining stack traces  #######################
# what?
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True,) as prof:
    model(inputs)

# Print aggregated stats
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))

"""
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                aten::cudnn_convolution        21.51%       1.699ms        27.46%       2.169ms     108.450us       8.031ms        82.90%       8.031ms     401.550us            20  
cudnn_infer_volta_scudnn_winograd_128x128_ldg1_ldg4_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.564ms        57.43%       5.564ms     428.000us            13  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 7.900ms
Self CUDA time total: 9.688ms
"""

#######################  7. Using profiler to analyze long-running jobs  #######################
"""
PyTorch profiler offers an additional API to handle long-running jobs (such as training loops). 
Tracing all of the execution can be slow and result in very large trace files. 
To avoid this, use optional arguments: 
  schedule:
  - specifies a function that takes an integer argument (step number) as an input and returns an action for the profiler
  - the best way to use this parameter is to use torch.profiler.schedule helper function that can generate a schedule for you;
  on_trace_ready:
  - specifies a function that takes a reference to the profiler as an input and is called by the profiler each time the new trace is ready
"""
from torch.profiler import schedule

# Profiler assumes that the long-running job is composed of steps, numbered starting from zero. 
# profiler will skip the first 15 steps, spend the next step on the warm up, actively record the next 3 steps, skip another 5 steps, spend the next step on the warm up, actively record another 3 steps. Since the repeat=2 parameter value is specified, the profiler will stop the recording after the first two cycles.
my_schedule = schedule(
    skip_first=10, # ignore the first 10 steps
    wait=5, # idling 5 steps, during this phase profiler is not active
    warmup=1, # warmup 1 steps, during this phase profiler starts tracing, but the results are discarded
    active=3, # active=3 steps, during this phase profiler traces and records data;
    repeat=2)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    # At the end of each cycle profiler calls the specified on_trace_ready function and passes itself as an argument. 
    # This function is used to process the new trace - either by obtaining the table output or by saving the output on disk as a trace file.
    on_trace_ready=trace_handler
) as p:
    for idx in range(8):
        model(inputs)
        p.step()