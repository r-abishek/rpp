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

/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#pragma once
#ifndef GUARD_RPP_KERNEL_CACHE_HPP_
#define GUARD_RPP_KERNEL_CACHE_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "rpp.h"
#include "handle.hpp"
#include "kernel.hpp"
#include "simple_hash.hpp"

#ifdef GPU_SUPPORT

namespace rpp {

// The KernelCache class - Build and cache kernels
class KernelCache
{

    public:
    using Key        = std::pair<std::string, std::string>;
    using KernelMap  = std::unordered_map<Key, std::vector<Kernel>, SimpleHash>;
    using ProgramMap = std::unordered_map<Key, Program, SimpleHash>;

    Kernel AddKernel(Handle& h,
                     const std::string& algorithm,
                     const std::string& network_config,
                     const std::string& program_name,
                     const std::string& kernel_name,
                     const std::vector<size_t>& vld,
                     const std::vector<size_t>& vgd,
                     std::string params            = "",
                     std::size_t cache_index       = 0,
                     bool is_kernel_str            = false,
                     const std::string& kernel_src = "");

    void AddKernel(Key key, Kernel k, std::size_t cache_index);

    void ClearKernels(const std::string& algorithm, const std::string& network_config);

    const std::vector<Kernel>& GetKernels(const std::string& algorithm,
                                          const std::string& network_config);

    bool HasKernels(const std::string& algorithm, const std::string& network_config) const;

    KernelCache();

    private:
    KernelMap kernel_map;
    ProgramMap program_map;
};

} // namespace rpp

#endif // GPU_SUPPORT

#endif // GUARD_RPP_KERNEL_CACHE_HPP_
