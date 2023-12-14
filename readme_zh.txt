# TinyML板端部署样例

## 功能描述
	该样例主要是基于TinyML输出的om模型在板端上实现图片目标检测的功能。

## 目录结构
	此处work_dir为配置文件配置的路径
	├── work_dir
	│   ├── deploy
	│   │   ├── board
	│   │   │   ├── data
	│   │   │   │   ├── best.om //模型权重文件
	│   │   │   │   ├── model.cfg //参数配置文件（例如图片路径、模型路径、输入分辨率）
	│   │   │   │   ├── sample_input.yuv //测试图片
	│   │   │   │   └── draw_box.py //画框脚本
	│   │   │   ├── code
	│   │   │   │   ├── CMakeLists.txt //编译脚本
	│   │   │   │   ├── main.cpp //主函数，图片目标检测功能实现接口
	│   │   │   │   ├── DetectCfgParse.h //声明检测config解析函数的头文件
	│   │   │   │   ├── DetectCfgParse.cpp //检测config解析函数的实现文件
	│   │   │   │   ├── ForwardEngine.h //声明模型前处理和推理相关函数头文件
	│   │   │   │   ├── ForwardEngine.cpp //模型前处理和推理相关函数实现文件
	│   │   │   │   ├── DetectPostprocess.h //声明检测数据后处理相关函数头文件
	│   │   │   │   ├── DetectPostprocess.cpp //检测数据后处理相关函数实现文件
	│   │   │   │   ├── Utils.h //声明公共函数（例如：文件读取函数）的头文件
	│   │   │   │   └── Utils.cpp //公共函数（例如：文件读取函数）的实现文件
	│   │   │   ├── output //PC端编译代码后生成
	│   │   │   │   ├── main //可执行程序（PC端编译代码后生成）
	│   │   │   │   └── result.txt //检测结果（板端运行程序后生成）
	│   │   │   ├── build.sh //编译脚本，调用code目录下的CMakeLists文件
	│   │   │   └── readme_zh.txt //中文readme文件

## 环境要求
	1.  SDK包编译环境
	2.  TinyML部署环境

## PC端编译代码
    1.  配置SDK依赖（${SDKPATH}为SDK开发包路径），执行命令
	    export SDK_DIR=${SDKPATH}/amp/a55_linux/mpp/out
	2.  cd ${work_dir}/deploy/board
		bash build.sh
	3.  命令执行完后，在${work_dir}/deploy/board/output下生成main可执行程序

## 板端运行程序
	1.  配置svp动态库(如已配置则跳过此步)：
	    cd /; mkdir -p sdk
		mount -t nfs -o nolock -o tcp -o rsize=32768,wsize=32768 ${SERVERIP}:${SDKPATH} /sdk
		export LD_LIBRARY_PATH=/sdk/amp/a55_linux/mpp/out/lib:/sdk/amp/a55_linux/mpp/out/lib/svp_npu
	2.  在板端将PC环境的${work_dir}/deploy/board目录挂载到到单板：
		cd /; mkdir -p board
		mount -t nfs -o nolock -o tcp -o rsize=32768,wsize=32768 ${SERVERIP}:${work_dir}/deploy/board /board
	3.  运行可执行文件。执行以下命令：
	    cd /board;
		chmod a+r -R data;
		cd output;chmod a+x main;./main
		chmod a+r result.txt 
	4. result.txt说明
		320  192  2  //原图宽 原图高 检测类别数
		pedestrian  0         0.85278314    8.95   33.53   95.77  177.11  //类别名 类别ID 置信度 左上点_X 左上点_Y 右下点_X 右下点_Y
		face  1         0.83430398   39.25   46.18   59.87   74.09 	//同上
	5. 若更换测试图片，请同步更新${work_dir}/deploy/board/data/model.cfg中的imageShape。
		imageShape = [测试图片的高，测试图片的宽]
		备注：务必保证${work_dir}/deploy/board/data/model.cfg中的imageShape与inputShape的宽高比一致，若不一致，可能影响模型效果

## PC端进行推理结果可视化(TinyML运行环境下)，<image_path>为${work_dir}/deploy/pc/data/sample_input.jpg的路径，结果图片保存路径为${work_dir}/deploy/board/data/output_sample_input.jpg。
	cd ${work_dir}/deploy/board/data
	python draw_box.py -i <image_path> -r ../output/result.txt




