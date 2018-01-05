#include "semantic_map_builder.h"

using namespace std;

namespace semantic_map_builder{

SemanticMapBuilder::SemanticMapBuilder(){
    _K = Eigen::Matrix3f::Zero();
    _robot_pose = Eigen::Isometry3f::Identity();
    _rgb_image = NULL;
    _depth_cloud = NULL;
}

Detections SemanticMapBuilder::detectObjects(const Eigen::Isometry3f &depth_camera_transform,
                                             const lucrezio_logical_camera::LogicalImage::ConstPtr &logical_image_msg,
                                             const PointCloudType::ConstPtr &depth_cloud_msg){
    Detections detections;

    PointCloudType::Ptr local_map_cloud (new PointCloudType ());
    pcl::transformPointCloud (*depth_cloud_msg, *local_map_cloud, depth_camera_transform);
    local_map_cloud->header.frame_id = "/map";
    local_map_cloud->width  = depth_cloud_msg->width;
    local_map_cloud->height = depth_cloud_msg->height;
    local_map_cloud->is_dense = false;

    tf::StampedTransform logical_camera_pose;
    tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);
    for(int i=0; i < logical_image_msg->models.size(); i++){

        Eigen::Vector3f box_min (logical_image_msg->models.at(i).min.x,
                                 logical_image_msg->models.at(i).min.y,
                                 logical_image_msg->models.at(i).min.z);

        Eigen::Vector3f box_max (logical_image_msg->models.at(i).max.x,
                                 logical_image_msg->models.at(i).max.y,
                                 logical_image_msg->models.at(i).max.z);

        float x_range = box_max.x()-box_min.x();
        float y_range = box_max.y()-box_min.y();
        float z_range = box_max.z()-box_min.z();

        PointCloudType::Ptr bounding_box_cloud (new PointCloudType ());
        for(int k=0; k <= 1; k++)
            for(int j=0; j <= 1; j++)
                for(int i=0; i <= 1; i++){
                    bounding_box_cloud->points.push_back (pcl::PointXYZ(box_min.x() + i*x_range,
                                                                        box_min.y() + j*y_range,
                                                                        box_min.z() + k*z_range));
                }
        PointCloudType::Ptr transformed_bounding_box_cloud (new PointCloudType ());
        tf::Transform model_pose;
        tf::poseMsgToTF(logical_image_msg->models.at(i).pose,model_pose);
        Eigen::Isometry3f model_transform = tfTransform2eigen(logical_camera_pose)*tfTransform2eigen(model_pose);
        pcl::transformPointCloud (*bounding_box_cloud, *transformed_bounding_box_cloud, model_transform);
        pcl::PointXYZ min_pt,max_pt;
        pcl::getMinMax3D(*transformed_bounding_box_cloud,min_pt,max_pt);

        PointCloudType::Ptr cloud_filtered_x (new PointCloudType ());
        PointCloudType::Ptr cloud_filtered_xy (new PointCloudType ());
        PointCloudType::Ptr cloud_filtered_xyz (new PointCloudType ());

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud (local_map_cloud);
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

        if(!cloud_filtered_xyz->points.empty()){
            cv::Point2i p_min(10000,10000);
            cv::Point2i p_max(-10000,-10000);

            for(int i=0; i<cloud_filtered_xyz->points.size(); i++){
                Eigen::Vector3f camera_point = depth_camera_transform.inverse()*
                        Eigen::Vector3f(cloud_filtered_xyz->points[i].x,
                                        cloud_filtered_xyz->points[i].y,
                                        cloud_filtered_xyz->points[i].z);
                Eigen::Vector3f image_point = _K*camera_point;

                const float& z=image_point.z();
                image_point.head<2>()/=z;
                int r = image_point.x();
                int c = image_point.y();

                if(r < p_min.x)
                    p_min.x = r;
                if(r > p_max.x)
                    p_max.x = r;

                if(c < p_min.y)
                    p_min.y = c;
                if(c > p_max.y)
                    p_max.y = c;
            }
            detections.push_back(Detection(logical_image_msg->models.at(i).type,
                                           (float)(p_min.y+p_max.y)/2.0f,
                                           (float)(p_min.x+p_max.x)/2.0f,
                                           (float)(p_max.y-p_min.y),
                                           (float)(p_max.x-p_min.x)));
        }
    }
    return detections;
}

Objects SemanticMapBuilder::extractBoundingBoxes(const Detections &detections, const cv::Mat& depth_image, const Eigen::Isometry3f &depth_camera_transform){
    Objects objects;

    for(int i=0; i < detections.size(); ++i){

        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Vector3f size = Eigen::Vector3f::Zero();

        const Detection& detection = detections[i];

        cv::Mat roi = depth_image(cv::Rect(detection.x(),
                                           detection.y(),
                                           detection.width(),
                                           detection.height()));
        int roi_rows = roi.rows;
        int roi_cols = roi.cols;
        cv::Mat roi_without_floor;
        roi_without_floor.create(roi_rows,roi_cols,CV_16UC1);
        roi_without_floor = std::numeric_limits<ushort>::max();

        float min_distance=100;
        int closest_r=0;
        int closest_c=0;
        for(int r = 0; r < roi_rows; ++r){
            const ushort* roi_ptr  = roi.ptr<ushort>(r);
            ushort* out_ptr  = roi_without_floor.ptr<ushort>(r);
            for(int c=0; c < roi_cols; ++c, ++roi_ptr, ++out_ptr) {
                const ushort& depth = *roi_ptr;
                ushort& out = *out_ptr;
                float d = depth * _raw_depth_scale;
                Eigen::Vector3f point = _invK * Eigen::Vector3f(c*d,r*d,d);

                if(point.y() < _camera_height){
                    out = depth;
                    float distance = point.squaredNorm();
                    if(distance<min_distance){
                        min_distance = distance;
                        closest_r = r;
                        closest_c = c;
                    }
                }
            }
        }

        cv::Mat mask;
        mask.create(roi_rows+2,roi_cols+2,CV_8UC1);
        mask = 0;
        cv::Point closest_point(closest_r,closest_c);
        cv::Rect ccomp;
        cv::floodFill(roi_without_floor,
                      mask,
                      closest_point,
                      255,
                      &ccomp,
                      cv::Scalar(0.025),
                      cv::Scalar(0.025));

        int k=0;
        float x_min=1000000,x_max=-1000000,y_min=1000000,y_max=-1000000,z_min=1000000,z_max=-1000000;
        for(int r = 0; r < roi_rows; ++r){
            const ushort* roi_ptr  = roi_without_floor.ptr<ushort>(r);
            for(int c=0; c < roi_cols; ++c, ++roi_ptr) {
                const ushort& depth = *roi_ptr;
                float d = depth * _raw_depth_scale;
                Eigen::Vector3f world_point = depth_camera_transform*_invK * Eigen::Vector3f(c*d,r*d,d);
                centroid += world_point;

                if(world_point.x()<x_min)
                    x_min = world_point.x();
                if(world_point.x()>x_max)
                    x_max = world_point.x();
                if(world_point.y()<y_min)
                    y_min = world_point.y();
                if(world_point.y()>y_max)
                    y_max = world_point.y();
                if(world_point.z()<z_min)
                    z_min = world_point.z();
                if(world_point.z()>z_max)
                    z_max = world_point.z();

                k++;
            }
        }
        centroid = centroid/(float)k;
        size = Eigen::Vector3f(x_max-x_min,y_max-y_min,z_max-z_min);

        objects.push_back(Object(detection.type(),
                                    centroid,
                                    size));

    }
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
