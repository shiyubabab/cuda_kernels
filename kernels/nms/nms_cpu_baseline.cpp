/*************************************************************************
	> File Name: nms_cpu_baseline.cpp
	> Author: mlxh
	> Mail: mlxh_gto@163.com 
	> Created Time: Fri 16 May 2025 10:32:50 AM CST
 ************************************************************************/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>


struct BoundingBox{
	float x1;
	float y1;
	float x2;
	float y2;
	float confidence;
};

float calculate_IOU(const BoundingBox &box1,const BoundingBox &box2){
	float x_left   = std::max(box1.x1,box2.x1); //左边
	float y_top    = std::max(box1.y1,box2.y1);
	float x_right  = std::min(box1.x2,box2.x2);	//右边
	float y_bottom = std::min(box1.y2,box2.y2);

	if(x_right < x_left || y_bottom < y_top){
		return 0.0f; //没有重叠
	}

	float intersection_area = (x_right - x_left) * (y_bottom - y_top);

	float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
	float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
	float union_area = area1 + area2 - intersection_area;

	if (union_area == 0.0f){
		return 0.0f;
	}
	return intersection_area / union_area;
}

std::vector<BoundingBox> non_maximum_suppression(std::vector<BoundingBox> boxes, float iou_threshold){
	if(boxes.empty()){
		return {};
	}
	std::sort(boxes.begin(),boxes.end(),[](const BoundingBox &a,const BoundingBox &b){
		return a.confidence > b.confidence;
	});

	std::vector<BoundingBox> result;
	std::vector<bool> suppressed(boxes.size(),false);

	for(size_t i = 0;i<boxes.size();++i){
		if(suppressed[i]){
			continue;
		}
		BoundingBox current_box = boxes[i];
		result.push_back(current_box);
		for(size_t j = i+1;j < boxes.size();++j){
			if(suppressed[j]){
				continue;
			}

			float iou = calculate_IOU(current_box,boxes[j]);
			if(iou>iou_threshold){
				suppressed[j] = true;
			}
		}
	}
	return result;
}

int main() {
	std::vector<BoundingBox> predicted_boxes = {
		{100, 100, 200, 200, 0.9f},
		{120, 110, 210, 210, 0.85f}, // 与第一个框重叠
		{50, 50, 150, 150, 0.7f},
		{180, 180, 280, 280, 0.92f},
		{190, 190, 290, 290, 0.88f}  // 与第四个框重叠
	};

	float nms_threshold = 0.5f;
	std::vector<BoundingBox> final_boxes = non_maximum_suppression(predicted_boxes, nms_threshold);

	std::cout << "原始预测框数量: " << predicted_boxes.size() << std::endl;
	std::cout << "NMS 处理后的框数量: " << final_boxes.size() << std::endl;

	std::cout << "NMS 处理后的边界框:" << std::endl;
	for (const auto& box : final_boxes) {
		std::cout << "x1: " << box.x1 << ", y1: " << box.y1 << ", x2: " << box.x2 << ", y2: " << box.y2 << ", confidence: " << box.confidence << std::endl;
	} 
	return 0;
}
