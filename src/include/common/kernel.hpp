/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef GUARD_RPP_KERNEL_HPP
#define GUARD_RPP_KERNEL_HPP

#include <string>
#include <vector>

#ifdef HIP_COMPILE

#include "hipoc_kernel.hpp"
namespace rpp {
std::string GetKernelSrc(std::string name);
std::string GetKernelInc(std::string key);
std::vector<std::string> GetKernelIncList();
using Kernel       = HIPOCKernel;
using KernelInvoke = HIPOCKernelInvoke;
using Program      = HIPOCProgram;
}    // namespace rpp

#elif defined(OCL_COMPILE)

#include "clhelper.hpp"
#include "oclkernel.hpp"
namespace rpp {
std::string GetKernelSrc(std::string name);
std::string GetKernelInc(std::string key);
std::vector<std::string> GetKernelIncList();
using Kernel       = OCLKernel;
using KernelInvoke = OCLKernelInvoke;
using Program      = SharedProgramPtr;
} // namespace rpp

#endif    // BACKEND

#endif    // GUARD_RPP_KERNEL_HPP
