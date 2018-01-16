#include "semantic_map_builder.h"

using namespace std;

namespace semantic_map_builder{

SemanticMapBuilder::SemanticMapBuilder(){
    _K = Eigen::Matrix3f::Zero();
    _depth_cloud = NULL;
    _camera_height = 0.5;
    _focal_length = 277.1273455955935;
    _raw_depth_scale = 1e-3;
    _depth_camera_transform = Eigen::Isometry3f::Identity();
    _inverse_depth_camera_transform = Eigen::Isometry3f::Identity();
}

void SemanticMapBuilder::setImages(const cv::Mat &rgb_image_, const cv::Mat &depth_image_){
    _rgb_image = rgb_image_;
    _depth_image = depth_image_;
}

void SemanticMapBuilder::computeWorldBoundingBoxes(BoundingBoxes3D &bounding_boxes,
                                                   const Eigen::Isometry3f &transform,
                                                   const Models &models){
    for(int i=0; i<models.size(); ++i){
        const lucrezio_logical_camera::Model &model = models[i];

        tf::Transform model_pose;
        tf::poseMsgToTF(model.pose,model_pose);

        std::vector<Eigen::Vector3f> points;
        points.push_back(transform*tfTransform2eigen(model_pose)*Eigen::Vector3f(model.min.x,model.min.y,model.min.z));
        points.push_back(transform*tfTransform2eigen(model_pose)*Eigen::Vector3f(model.max.x,model.max.y,model.max.z));

        float x_min=100000,x_max=-100000,y_min=100000,y_max=-100000,z_min=100000,z_max=-100000;
        for(int i=0; i < 2; ++i){
            if(points[i].x()<x_min)
                x_min = points[i].x();
            if(points[i].x()>x_max)
                x_max = points[i].x();
            if(points[i].y()<y_min)
                y_min = points[i].y();
            if(points[i].y()>y_max)
                y_max = points[i].y();
            if(points[i].z()<z_min)
                z_min = points[i].z();
            if(points[i].z()>z_max)
                z_max = points[i].z();
        }

        bounding_boxes[i] = std::make_pair(Eigen::Vector3f(x_min,y_min,z_min),Eigen::Vector3f(x_max,y_max,z_max));
        _detections[i].type = model.type;
    }
}

void SemanticMapBuilder::computeImageBoundingBoxes(const BoundingBoxes3D &bounding_boxes){
    for(int i=0; i < _depth_cloud->size(); ++i){
        for(int j=0; j < bounding_boxes.size(); ++j){
            if(inRange(_depth_cloud->points[i],bounding_boxes[j])){
                Eigen::Vector3f image_point = _K*Eigen::Vector3f(_depth_cloud->points[i].x,
                                                                 _depth_cloud->points[i].y,
                                                                 _depth_cloud->points[i].z);

                const float& z=image_point.z();
                image_point.head<2>()/=z;
                int r = image_point.x();
                int c = image_point.y();
                int &r_min = _detections[j].p_min.x();
                int &c_min = _detections[j].p_min.y();
                int &r_max = _detections[j].p_max.x();
                int &c_max = _detections[j].p_max.y();

                if(r < r_min)
                    r_min = r;
                if(r > r_max)
                    r_max = r;

                if(c < c_min)
                    c_min = c;
                if(c > c_max)
                    c_max = c;

                _detections[j].pixels.push_back(Eigen::Vector2i(c,r));

                break;
            }
        }
    }
}

void SemanticMapBuilder::detectObjects(const lucrezio_logical_camera::LogicalImage::ConstPtr &logical_image_msg){
    ROS_INFO("DETECTION----------------------------------------");
    cerr << endl;

    int num_models = logical_image_msg->models.size();
    cerr << "num_models: " << num_models;

    _detections.clear();
    _detections.resize(num_models);

    //Compute world bounding boxes
    double cv_wbb_time = (double)cv::getTickCount();
    BoundingBoxes3D bounding_boxes(num_models);
    tf::StampedTransform logical_camera_pose;
    tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);
    computeWorldBoundingBoxes(bounding_boxes,
                              _inverse_depth_camera_transform*tfTransform2eigen(logical_camera_pose),
                              logical_image_msg->models);
    ROS_INFO("Computing WBB took: %f",((double)cv::getTickCount() - cv_wbb_time)/cv::getTickFrequency());

    //Compute image bounding boxes
    double cv_ibb_time = (double)cv::getTickCount();
    computeImageBoundingBoxes(bounding_boxes);
    ROS_INFO("Computing IBB took: %f",((double)cv::getTickCount() - cv_ibb_time)/cv::getTickFrequency());

    cerr << endl;

    //print result
    for(int i=0; i < num_models; ++i){

        if(_detections[i].pixels.empty())
            continue;

        cerr << "-" << _detections[i].type << ":" << endl;
        cerr << "\t>>World Bounding Box: [";
        cerr << "(" << bounding_boxes[i].first.x() << "," << bounding_boxes[i].first.y() << "," << bounding_boxes[i].first.z() << "),";
        cerr << "(" << bounding_boxes[i].second.x() << "," << bounding_boxes[i].second.y() << "," << bounding_boxes[i].second.z() << ")]" << endl;
        cerr << "\t>>Image Bounding Box: [";
        cerr << "(" << _detections[i].p_min.x() << "," << _detections[i].p_min.y() << "),";
        cerr << "(" << _detections[i].p_max.x() << "," << _detections[i].p_max.y() << ")]" << endl;
        cerr << "\t>>Number of Pixels: " << _detections[i].pixels.size() << endl;
        cerr << endl;
    }
}

Objects SemanticMapBuilder::extractBoundingBoxes(const cv::Mat& depth_image){
    ROS_INFO("EXTRACTION---------------------------------------");
    cerr << endl;

    Objects objects;

    for(int i=0; i < _detections.size(); ++i){

        const Detection& detection = _detections[i];

        cerr << detection.type << ": [(";
        cerr << detection.p_min.transpose() << ") - (" << detection.p_max.transpose() << ")]" << endl;

        //cerr << "Depth image type: " << depth_image.type() << endl;

        //        cerr << "Depth camera transform: " << endl;
        //        cerr << _depth_camera_transform.translation().transpose() << endl;
        //        Eigen::Quaternionf q(_depth_camera_transform.linear());
        //        cerr << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << endl;
        //        cerr << "inverse K: " << endl;
        //        cerr << _invK << endl;

        PointCloudType::Ptr cloud (new PointCloudType);
        PointCloudType::Ptr cloud_filtered (new PointCloudType);
        const std::vector<Eigen::Vector2i>& pixels = detection.pixels;
        for(int idx=0; idx < pixels.size(); ++idx){
            const Eigen::Vector2i& pixel = pixels[idx];
            int r = pixel.x();
            int c = pixel.y();
            const unsigned short& depth = depth_image.at<const unsigned short>(r,c);
            float d = depth * _raw_depth_scale;

            if(d <= 0.02)
                continue;

            if(d >= 5.0)
                continue;

            Eigen::Vector3f camera_point = _invK * Eigen::Vector3f(c*d,r*d,d);
            Eigen::Vector3f world_point = _depth_camera_transform * camera_point;

            cloud->push_back(pcl::PointXYZ (world_point.x(),world_point.y(),world_point.z()));

        }

        pcl::io::savePCDFileASCII ("cloud.pcd", *cloud);

        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud (cloud);
        sor.setMeanK (10);
        sor.setStddevMulThresh (1.0);
        sor.filter (*cloud_filtered);
        pcl::io::savePCDFileASCII ("cloud_filtered.pcd", *cloud_filtered);

        pcl::PointXYZ min_pt,max_pt;
        pcl::getMinMax3D(*cloud_filtered,min_pt,max_pt);

        Eigen::Vector3f centroid = Eigen::Vector3f((min_pt.x+max_pt.x)/2.0f,
                                                   (min_pt.y+max_pt.y)/2.0f,
                                                   (min_pt.z+max_pt.z)/2.0f);
        Eigen::Vector3f size = Eigen::Vector3f(max_pt.x-min_pt.x,
                                               max_pt.y-min_pt.y,
                                               max_pt.z-min_pt.z);

        cerr << endl;
        cerr << "Centroid: " << centroid.transpose() << endl;
        cerr << "Size: " << size.transpose() << endl << endl;

        std::string object_type = detection.type.substr(0,detection.type.find_first_of("_"));
        objects.push_back(Object(object_type,
                                 centroid,
                                 size));

    }
    cerr << "-------------------------------------------------" << endl << endl;

    return objects;
}

Eigen::Isometry3f SemanticMapBuilder::tfTransform2eigen(const tf::Transform &p){
    Eigen::Isometry3f iso;
    iso.translation().x()=p.getOrigin().x();
    iso.translation().y()=p.getOrigin().y();
    iso.translation().z()=p.getOrigin().z();
    Eigen::Quaternionf q;
    tf::Quaternion tq = p.getRotation();
    q.x()= tq.x();
    q.y()= tq.y();
    q.z()= tq.z();
    q.w()= tq.w();
    iso.linear()=q.toRotationMatrix();
    return iso;
}

tf::Transform SemanticMapBuilder::eigen2tfTransform(const Eigen::Isometry3f &T){
    Eigen::Quaternionf q(T.linear());
    Eigen::Vector3f t=T.translation();
    tf::Transform tft;
    tft.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));
    tft.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
    return tft;
}

}
