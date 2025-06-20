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

#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
float gaussian(int x,int y, float stdDev) 
{
    float res,pi=3.14;
    res= 1 / (2 * pi * stdDev * stdDev);
    float exp1,exp2;
    exp1= - (x*x) / (2*stdDev*stdDev);
    exp2= - (y*y) / (2*stdDev*stdDev);
    exp1= exp1+exp2;
    exp1=exp(exp1);
    res*=exp1;
	return res;
}
__kernel void vignette_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float stdDev
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * channel * width + id_x * channel + id_z;

    int x = (id_x - (width / 2));
    int y = (id_y - (height / 2));
    float gaussianvalue = gaussian(x, y, stdDev) / gaussian(0.0, 0.0, stdDev);
    float res = ((float)input[pixIdx]) * gaussianvalue ;
    output[pixIdx] = saturate_8u(res) ;
}
__kernel void vignette_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float stdDev
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_z * width * height + id_y * width + id_x;

    int x = (id_x - (width / 2));
    int y = (id_y - (height / 2));
    float gaussianvalue=gaussian(x, y, stdDev) / gaussian(0.0, 0.0, stdDev);
    float res = ((float)input[pixIdx]) * gaussianvalue ;
    output[pixIdx] = saturate_8u(res) ;
}

__kernel void vignette_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global float* stdDev,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    float tempstdDev = stdDev[id_z];
    int indextmp=0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {   
        int x = (id_x - (width[id_z] >> 1));
        int y = (id_y - (height[id_z] >> 1));
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            float gaussianvalue=gaussian(x, y, tempstdDev) / gaussian(0.0, 0.0, tempstdDev);
            float res = ((float)input[pixIdx]) * gaussianvalue ;
            output[pixIdx] = saturate_8u((int)res);
            pixIdx += inc[id_z];
        }
    }
}
