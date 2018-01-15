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

PointCloudType::Ptr SemanticMapBuilder::transformCloud(const PointCloudType::ConstPtr& in_cloud,
                                                       const Eigen::Isometry3f& transform,
                                                       const string &frame_id){
    PointCloudType::Ptr out_cloud (new PointCloudType ());
    pcl::transformPointCloud (*in_cloud, *out_cloud, transform);
    out_cloud->header.frame_id = frame_id;
    out_cloud->width  = in_cloud->width;
    out_cloud->height = in_cloud->height;
    out_cloud->is_dense = false;

    return out_cloud;
}

PointCloudType::Ptr SemanticMapBuilder::modelBoundingBox(const lucrezio_logical_camera::Model& model,
                                                         const Eigen::Isometry3f& transform){

    PointCloudType::Ptr bounding_box_cloud (new PointCloudType ());

    Eigen::Vector3f box_min (model.min.x,model.min.y,model.min.z);

    Eigen::Vector3f box_max (model.max.x,model.max.y,model.max.z);

    float x_range = box_max.x()-box_min.x();
    float y_range = box_max.y()-box_min.y();
    float z_range = box_max.z()-box_min.z();

    for(int k=0; k <= 1; k++)
        for(int j=0; j <= 1; j++)
            for(int i=0; i <= 1; i++){
                bounding_box_cloud->points.push_back (pcl::PointXYZ(box_min.x() + i*x_range,
                                                                    box_min.y() + j*y_range,
                                                                    box_min.z() + k*z_range));
            }

    PointCloudType::Ptr transformed_bounding_box_cloud (new PointCloudType ());
    tf::Transform model_pose;
    tf::poseMsgToTF(model.pose,model_pose);
    Eigen::Isometry3f model_transform = transform*tfTransform2eigen(model_pose);
    pcl::transformPointCloud (*bounding_box_cloud, *transformed_bounding_box_cloud, model_transform);

    return transformed_bounding_box_cloud;

}

PointCloudType::Ptr SemanticMapBuilder::filterCloud(const PointCloudType::ConstPtr &in_cloud,
                                                    const pcl::PointXYZ& min_pt,
                                                    const pcl::PointXYZ& max_pt){

    PointCloudType::Ptr cloud_filtered_x (new PointCloudType ());
    PointCloudType::Ptr cloud_filtered_xy (new PointCloudType ());
    PointCloudType::Ptr cloud_filtered_xyz (new PointCloudType ());

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (in_cloud);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (min_pt.x,max_pt.x);
    pass.filter (*cloud_filtered_x);

    pass.setInputCloud (cloud_filtered_x);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (min_pt.y,max_pt.y);
    pass.filter (*cloud_filtered_xy);

    pass.setInputCloud (cloud_filtered_xy);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (min_pt.z,max_pt.z);
    pass.filter (*cloud_filtered_xyz);

    return cloud_filtered_xyz;
}

void SemanticMapBuilder::computeDetection(Eigen::Vector2i &p_min,
                                          Eigen::Vector2i &p_max,
                                          std::vector<Eigen::Vector2i> &pixels,
                                          const PointCloudType::Ptr &in_cloud){
    pixels.resize(in_cloud->points.size());
    for(int i=0; i<in_cloud->points.size(); i++){

        pcl::PointXYZ camera_point = pcl::transformPoint(in_cloud->points[i],_inverse_depth_camera_transform);
        Eigen::Vector3f image_point = _K*Eigen::Vector3f(camera_point.x,camera_point.y,camera_point.z);

        const float& z=image_point.z();
        image_point.head<2>()/=z;
        int r = image_point.x();
        int c = image_point.y();

        if(r < p_min.x())
            p_min.x() = r;
        if(r > p_max.x())
            p_max.x() = r;

        if(c < p_min.y())
            p_min.y() = c;
        if(c > p_max.y())
            p_max.y() = c;

        pixels[i] = Eigen::Vector2i(c,r);

        cv::circle(_rgb_image,
                   cv::Point(r,c),
                   1,
                   cv::Scalar(0,0,255));
    }


}


void SemanticMapBuilder::detectObjects(const lucrezio_logical_camera::LogicalImage::ConstPtr& logical_image_msg){
    cerr << "DETECTION----------------------------------------" << endl;
    _detections.clear();

    double cv_transformation_time = (double)cv::getTickCount();
    PointCloudType::Ptr local_map_cloud = transformCloud(_depth_cloud, _depth_camera_transform, "/map");
    ROS_INFO("Transforming cloud took: %f",((double)cv::getTickCount() - cv_transformation_time)/cv::getTickFrequency());

    tf::StampedTransform logical_camera_pose;
    tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);

    for(int i=0; i < logical_image_msg->models.size(); i++){

        cerr << "Detected model: " << logical_image_msg->models.at(i).type << endl;

        double cv_bb_time = (double)cv::getTickCount();
        PointCloudType::Ptr transformed_bounding_box_cloud = modelBoundingBox(logical_image_msg->models.at(i),
                                                                              tfTransform2eigen(logical_camera_pose));
        ROS_INFO("Computing BB took: %f",((double)cv::getTickCount() - cv_bb_time)/cv::getTickFrequency());

        double cv_minmax_time = (double)cv::getTickCount();
        pcl::PointXYZ min_pt,max_pt;
        pcl::getMinMax3D(*transformed_bounding_box_cloud,min_pt,max_pt);
        cerr << "Min: " << min_pt << "Max: " << max_pt << endl;
        ROS_INFO("Computing MinMax took: %f",((double)cv::getTickCount() - cv_minmax_time)/cv::getTickFrequency());

        double cv_filter_time = (double)cv::getTickCount();
        PointCloudType::Ptr cloud_filtered = filterCloud(local_map_cloud,min_pt,max_pt);
        ROS_INFO("Filtering cloud took: %f",((double)cv::getTickCount() - cv_filter_time)/cv::getTickFrequency());

        if(!cloud_filtered->points.empty()){

            double cv_detection_time = (double)cv::getTickCount();
            Eigen::Vector2i p_min(10000,10000);
            Eigen::Vector2i p_max(-10000,-10000);
            std::vector<Eigen::Vector2i> pixels;
            computeDetection(p_min,p_max,pixels,cloud_filtered);
            _detections.push_back(Detection(logical_image_msg->models.at(i).type,
                                           p_min,
                                           p_max,
                                           pixels));
            ROS_INFO("Computing detection took: %f",((double)cv::getTickCount() - cv_detection_time)/cv::getTickFrequency());
            cerr << "Number of pixels: " << pixels.size() << endl;
            cerr << "Min x: " << p_min.x() << " - Min y: " << p_min.y();
            cerr << " - Max x: " << p_max.x() << " - Max y: " << p_max.y() << endl;
            cv::rectangle(_rgb_image,
                          cv::Point(p_min.x(),p_min.y()),
                          cv::Point(p_max.x(),p_max.y()),
                          cv::Scalar(255,0,0));
            cerr << endl;
        }
    }
    cv::imwrite("detections.png",_rgb_image);
    cerr << "-------------------------------------------------" << endl << endl;
}

Objects SemanticMapBuilder::extractBoundingBoxes(const cv::Mat& depth_image){
    cerr << "EXTRACTION---------------------------------------" << endl;
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

        objects.push_back(Object(detection.type,
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
