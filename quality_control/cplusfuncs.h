#pragma once
/*
   Created by SEU-BME-LBMD-chl, SEU-BME-LBMD-zzy, SEU-BME-LBMD-scj
*/
#include <io.h>
#include <math.h>
#include <direct.h>
#include <windows.h>

#include <stack>
#include <ctime>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cassert>
#include <iostream>



//获取该文件夹内所有文件的信息
void getFilepaths(std::string indir, std::vector<std::string>& filepaths);

// 在vector中找到对应对象的索引（取最小值）
int get_vector_idx_int(std::vector<int> vector_input, int find_val);

// 在字符串中找到子字符串的索引（取最小值）
int get_vector_idx_str(std::string vector_input, std::string find_val);