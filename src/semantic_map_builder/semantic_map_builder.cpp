#include "semantic_map_builder.h"

using namespace std;

namespace semantic_map_builder{

SemanticMapBuilder::SemanticMapBuilder(){
    _K = Eigen::Matrix3f::Zero();
    _rgb_image = NULL;
    _depth_cloud = NULL;
    _camera_height = 0.5;
    _focal_length = 277.1273455955935;
    _raw_depth_scale = 1e-3;
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


Detections SemanticMapBuilder::detectObjects(cv::Mat rgb_image,
                                             const Eigen::Isometry3f& depth_camera_transform,
                                             const lucrezio_logical_camera::LogicalImage::ConstPtr& logical_image_msg,
                                             const PointCloudType::ConstPtr& depth_cloud_msg){
    cerr << "DETECTION----------------------------------------" << endl;
    Detections detections;

    PointCloudType::Ptr local_map_cloud = transformCloud(depth_cloud_msg, depth_camera_transform, "/map");

    tf::StampedTransform logical_camera_pose;
    tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);

    for(int i=0; i < logical_image_msg->models.size(); i++){

        cerr << "Detected model: " << logical_image_msg->models.at(i).type << endl;

        PointCloudType::Ptr transformed_bounding_box_cloud = modelBoundingBox(logical_image_msg->models.at(i),
                                                                              tfTransform2eigen(logical_camera_pose));

        pcl::PointXYZ min_pt,max_pt;
        pcl::getMinMax3D(*transformed_bounding_box_cloud,min_pt,max_pt);
        cerr << "Min: " << min_pt << "Max: " << max_pt << endl;

        PointCloudType::Ptr cloud_filtered = filterCloud(local_map_cloud,min_pt,max_pt);

        if(!cloud_filtered->points.empty()){
            Eigen::Vector2i p_min(10000,10000);
            Eigen::Vector2i p_max(-10000,-10000);

            for(int i=0; i<cloud_filtered->points.size(); i++){

                Eigen::Vector3f camera_point = depth_camera_transform.inverse()*
                        Eigen::Vector3f(cloud_filtered->points[i].x,
                                        cloud_filtered->points[i].y,
                                        cloud_filtered->points[i].z);
                Eigen::Vector3f image_point = _K*camera_point;

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

                cv::circle(rgb_image,
                           cv::Point(r,c),
                           1,
                           cv::Scalar(0,0,255));
            }

            detections.push_back(Detection(logical_image_msg->models.at(i).type,
                                           p_min,
                                           p_max));

            cerr << "Min x: " << p_min.x() << " - Min y: " << p_min.y();
            cerr << " - Max x: " << p_max.x() << " - Max y: " << p_max.y() << endl;
            cv::rectangle(rgb_image,
                          cv::Point(p_min.x(),p_min.y()),
                          cv::Point(p_max.x(),p_max.y()),
                          cv::Scalar(255,0,0));
        }
    }
    cv::imwrite("detections.png",rgb_image);
    cerr << "-------------------------------------------------" << endl << endl;
    return detections;
}

Objects SemanticMapBuilder::extractBoundingBoxes(const Detections &detections,
                                                 const cv::Mat& depth_image,
                                                 const Eigen::Isometry3f &depth_camera_transform){
    cerr << "EXTRACTION---------------------------------------" << endl;
    Objects objects;

    for(int i=0; i < detections.size(); ++i){

        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Vector3f size = Eigen::Vector3f::Zero();

        const Detection& detection = detections[i];

        cerr << detection.type << ": [(";
        cerr << detection.p_min.transpose() << ") - (" << detection.p_max.transpose() << ")]" << endl;

        cerr << "Depth image type: " << depth_image.type() << endl;
        cv::imwrite("depth.png",depth_image);

        //        cv::Mat_<float> depth_converted;
        //        convert_16UC1_to_32FC1(depth_converted, depth_image);

        cv::Mat roi = depth_image(cv::Rect(cv::Point(detection.p_min.x(),detection.p_min.y()),
                                           cv::Point(detection.p_max.x(),detection.p_max.y())));
        int roi_rows = roi.rows;
        int roi_cols = roi.cols;
        cerr << "ROI: " << roi_rows << "x" << roi_cols << endl;
        cerr << "ROI image type: " << roi.type() << endl;
        cv::imwrite("roi.png",roi*5.0);

        cv::Mat roi_without_floor;
        roi_without_floor.create(roi_rows,roi_cols,CV_16UC1);
        roi_without_floor = std::numeric_limits<unsigned short>::max();
        cerr << "ROI_without_floor image type: " << roi_without_floor.type() << endl;

        float min_depth=std::numeric_limits<float>::max();
        int closest_r=-1;
        int closest_c=-1;
        for(int r = 0; r < roi_rows; ++r){
            const unsigned short* roi_ptr  = roi.ptr<unsigned short>(r);
            unsigned short* out_ptr  = roi_without_floor.ptr<unsigned short>(r);
            for(int c=0; c < roi_cols; ++c, ++roi_ptr, ++out_ptr) {
                const unsigned short& depth = *roi_ptr;
                unsigned short& out = *out_ptr;
                float d = depth * _raw_depth_scale;

                //                if(d <= 0)
                //                    continue;

                //                Eigen::Vector3f point = _invK * Eigen::Vector3f(c*d,r*d,d);

                //                if(point.y() > -_camera_height){
                //                    out = depth;
                //                    if(d<min_depth){
                //                        min_depth = d;
                //                        closest_r = r;
                //                        closest_c = c;
                //                    }
                //                }

                float value = (0.5 * depth_image.rows - (detection.p_min.x() + r)) * d / _focal_length;
                if(value > -_camera_height){
                    out = depth;
                    if(d<min_depth){
                        min_depth = d;
                        closest_r = r;
                        closest_c = c;
                    }
                }
            }
        }
        cerr << endl;
        cerr << "Min depth: " << min_depth << endl;
        cerr << "Closest Point: " << closest_r << "," << closest_c << endl;

        cv::Mat roi_without_floor_gray;
        cv::Mat temp;
        roi_without_floor.convertTo(roi_without_floor_gray,CV_8UC1,0.00390625);
        cv::cvtColor(roi_without_floor_gray,temp,CV_GRAY2BGR);
        cv::circle(temp,
                   cv::Point(closest_c,closest_r),
                   3,
                   cv::Scalar(255,0,0),
                   3);


        cv::Mat mask;
        mask.create(roi_rows+2,roi_cols+2,CV_8UC1);
        mask = 0;
        cv::Point closest_point(closest_c,closest_r);
        cv::Rect ccomp;
        cv::floodFill(roi_without_floor_gray,
                      mask,
                      closest_point,
                      255,
                      &ccomp,
                      cv::Scalar(0.025),
                      cv::Scalar(0.025));

        cerr << "FloodFill output: " << endl;
        cerr << "X: " << ccomp.x << " -Y: " << ccomp.y << " -W: " << ccomp.width << " -H: " << ccomp.height << endl;
        cerr << "TopLeft: " << ccomp.tl() << " -BottomRight: " << ccomp.br() << endl;
        cv::rectangle(temp,ccomp,cv::Scalar(0,0,255));

        cv::imwrite("roi_without_floor_gray.png",roi_without_floor_gray*5.0);
        cv::imwrite("roi_without_floor.png",temp*5.0);

        double min_value,max_value;
        cv::minMaxLoc(roi_without_floor(ccomp), &min_value, &max_value);

        cerr << "max_value: " << max_value << " -min_value: " << min_value << endl;

        cv::Point im_size = ccomp.br() - ccomp.tl();

        float obj_y_size = im_size.x * (min_value*1e-3) / _focal_length;
        float obj_x_size = im_size.y * (min_value*1e-3) / _focal_length;
        float obj_depth_size = (max_value - min_value)*1e-3;

        int k=0;
        float x_min=1000000,x_max=-1000000,y_min=1000000,y_max=-1000000,z_min=1000000,z_max=-1000000;

        centroid = centroid/(float)k;
        size = Eigen::Vector3f(x_max-x_min,y_max-y_min,z_max-z_min);

        //        cerr << "Centroid: " << centroid.transpose() << endl;
        //        cerr << "Size: " << size.transpose() << endl;

        cerr << "Size: " << obj_x_size << "," << obj_y_size << "," << obj_depth_size << endl;
        //cerr << "Max: " << x_max << "," << y_max << "," << z_max << endl;

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

void SemanticMapBuilder::convert_16UC1_to_32FC1(cv::Mat& dest, const cv::Mat& src, float scale) {
    assert(src.type() != CV_16UC1 && "convert_16UC1_to_32FC1: source image of different type from 16UC1");
    const unsigned short* sptr = (const unsigned short*)src.data;
    int size = src.rows * src.cols;
    const unsigned short* send = sptr + size;
    dest.create(src.rows, src.cols, CV_32FC1);
    dest.setTo(cv::Scalar(0.0f));
    float* dptr = (float*)dest.data;
    while(sptr < send) {
        if(*sptr == 0) { *dptr = 1e9f; }
        else { *dptr = scale * (*sptr); }
        ++dptr;
        ++sptr;
    }
}


}

//        for(int r = 0; r < roi_rows; ++r){
//            const unsigned short* roi_ptr  = roi.ptr<unsigned short>(r);
//            const unsigned char* mask_ptr = roi_without_floor_gray.ptr<unsigned char>(r);
//            for(int c=0; c < roi_cols; ++c, ++roi_ptr, ++mask_ptr) {
//                if((*mask_ptr) != 200)
//                    continue;

//                const unsigned short& depth = *roi_ptr;
//                float d = depth * _raw_depth_scale;

//                if (d > 5.0f)
//                    continue;

//                Eigen::Vector3f world_point = depth_camera_transform*_invK * Eigen::Vector3f(c*d,r*d,d);
//                centroid += world_point;

//                if(world_point.x()<x_min)
//                    x_min = world_point.x();
//                if(world_point.x()>x_max)
//                    x_max = world_point.x();
//                if(world_point.y()<y_min)
//                    y_min = world_point.y();
//                if(world_point.y()>y_max)
//                    y_max = world_point.y();
//                if(world_point.z()<z_min)
//                    z_min = world_point.z();
//                if(world_point.z()>z_max)
//                    z_max = world_point.z();

//                k++;
//            }
//        }
