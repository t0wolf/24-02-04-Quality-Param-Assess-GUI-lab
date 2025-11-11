/*
   Created by SEU-BME-LBMD-chl, SEU-BME-LBMD-zzy, SEU-BME-LBMD-scj
*/
#include "cplusfuncs.h"



//获取该文件夹内所有文件的信息
//输入：文件夹路径
//输出：文件夹内所有文件路径的字符串向量组filepaths
void getFilepaths(std::string indir, std::vector<std::string>& filepaths) {
	//文件句柄  
	intptr_t hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(indir).append("/*").c_str(), &fileinfo)) != -1) {
		do {
			//如果是目录,进行迭代；如果不是,加入列表  
			if ((fileinfo.attrib & _A_SUBDIR)) {
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFilepaths(p.assign(indir).append("/").append(fileinfo.name), filepaths);
			}
			else {
				filepaths.push_back(p.assign(indir).append("/").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}




// 在vector中找到对应对象的索引（取最小值）
int get_vector_idx_int(std::vector<int> vector_input, int find_val)
{
	int vector_idx;
	std::vector<int>::iterator find_num = std::find(vector_input.begin(), vector_input.end(), find_val);
	vector_idx = (find_num != vector_input.end()) ? std::distance(vector_input.begin(), find_num) : -1;

	return vector_idx;
}


// 在字符串中找到子字符串的索引（取最小值）
int get_vector_idx_str(std::string vector_input, std::string find_val)
{
	int vector_idx;
	std::string::size_type find_num = vector_input.find(find_val);
	vector_idx = (find_num == std::string::npos) ? -1 : 1;

	return vector_idx;
}